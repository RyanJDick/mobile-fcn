'''
Ryan Dick
December 9, 2017

A script to load the trained weights and test IoU metric of the network on the
test dataset.
'''
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import scipy
import sys
import json
from kitti_dataset_factory import build_test_dataset
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.python.client import timeline

sys.path.insert(0, '../models')
from mobilenet import MOBILENET
from vgg16 import VGG_16
from fcn_separable import FCN_SEP_EXTENSION
from fcn import FCN_EXTENSION
from cross_entropy import get_valid_entries_indices_from_annotation_batch, get_one_hot_labels_from_annotation_batch

flags = tf.app.flags

flags.DEFINE_string('config', None, 'Path to configuration .json file.')


def save_output_image(predictions, input_file_path, cfg):
    # Determine output file path
    output_image_folder_path = os.path.dirname(cfg['test_output_image_folder'])
    os.makedirs(output_image_folder_path, exist_ok=True) # Create ouput image folder if it does not already exist

    input_file_name = os.path.basename(input_file_path[0].decode('utf-8'))
    input_file_name_split = os.path.splitext(input_file_name)
    output_file_name = input_file_name_split[0] + '_pred' + input_file_name_split[1]
    output_file_path = os.path.join(output_image_folder_path, output_file_name)

    output_img = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
    classes = cfg['classes']
    for row in range(predictions.shape[0]):
        for col in range(predictions.shape[1]):
            output_img[row][col] = classes[int(predictions[row][col])]

    scipy.misc.imsave(output_file_path, output_img)
    print('Saved "' + output_file_path + '"')

def main(_):
    ########################
    ### Load config file ###
    ########################
    if tf.app.flags.FLAGS.config is None:
        logging.error("No config file is provided.")
        logging.info("Usage: python train.py --config config.json")
        exit(1)

    with open(tf.app.flags.FLAGS.config) as config_file:
        cfg = json.load(config_file)

    ###############################
    ### Setup dataset iterators ###
    ###############################
    test_dataset = build_test_dataset(cfg)

    test_dataset = test_dataset.batch(1) # Process one image at a time

    # Setup test dataset iterators
    test_iterator = test_dataset.make_one_shot_iterator()
    next_element = test_iterator.get_next()

    ############################
    ### Initialize the model ###
    ############################
    if cfg['model'] == 'vgg16_fcn':
        conv7_features, pool4_features, pool3_features = VGG_16(image_batch_tensor = next_element[0],
                                                                is_training = False)

        output_logits_dict = FCN_EXTENSION( conv7_features,
                                            pool4_features,
                                            pool3_features,
                                            len(cfg['classes']))
    elif cfg['model'] == 'mobilenet_fcn':
        conv13_features, conv11_features, conv5_features = MOBILENET(image_batch_tensor = next_element[0],
                                                                is_training = False)
        output_logits_dict = FCN_EXTENSION( conv13_features,
                                            conv11_features,
                                            conv5_features,
                                            len(cfg['classes']))
    elif cfg['model'] == 'mobilenet_fcn_sep':
        conv13_features, conv11_features, conv5_features = MOBILENET(image_batch_tensor = next_element[0],
                                                                is_training = False)
        output_logits_dict = FCN_SEP_EXTENSION( conv13_features,
                                                conv11_features,
                                                conv5_features,
                                                len(cfg['classes']))


    #######################################
    ### Process annotation/label tensor ###
    #######################################

    annotation_batch_tensor = tf.squeeze(next_element[1], axis=-1) # (b x h x w x 1) -> (b x h x w)after squeeze
    # Later, resize the predictions to match the annotation tensor. Unfortunately,
    # the KITTI dataset has slightly different dimensions for some images, so this
    # is necessary:
    annotation_shape = tf.shape(annotation_batch_tensor)

    # Get (b, w, h) indices of pixel that are not masked out
    valid_batch_indices = get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor=annotation_batch_tensor)

    # Keep only valid pixels:
    valid_labels_batch_tensor = tf.gather_nd(params=annotation_batch_tensor,
                                                indices=valid_batch_indices)

    ##########################
    ### Add evaluation ops ###
    ##########################
    # Note: We use FCN-8s output for evaluation because all weights were
    # zero-initialized, so the untrained layers will not affect the output. In
    # other words, initially (before training FCN-16s and FCN-8s), the FCN-8s
    # output is the same as the FCN-32s output.

    # Resize logits to original image size
    orig_size_logits = tf.image.resize_images(output_logits_dict['fcn8'], [annotation_shape[1], annotation_shape[2]],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    orig_size_pred = tf.argmax(orig_size_logits, axis=3) # Predictions for all pixels (including masked)

    # Flattened array of logits for non-masked pixels
    valid_logits_batch_tensor = tf.gather_nd(params=orig_size_logits, indices=valid_batch_indices)

    # Calculate accuracy over non-masked pixels:
    valid_pred_batch_tensor = tf.argmax(valid_logits_batch_tensor, axis = 1)
    iou_metric_calculate, iou_metric_update = tf.metrics.mean_iou(valid_labels_batch_tensor, valid_pred_batch_tensor, len(cfg['classes']), name='iou_metric')
    iou_metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='iou_metric')
    iou_metric_initializer = tf.variables_initializer(var_list=iou_metric_vars)

    ##################################
    ### Initialize model variables ###
    ##################################
    global_vars_init_op = tf.global_variables_initializer()
    local_vars_init_op = tf.local_variables_initializer()
    combined_init_op = tf.group(local_vars_init_op, global_vars_init_op)

    # Saver is used to load full model checkpoints
    model_variables = slim.get_model_variables()
    saver = tf.train.Saver(model_variables)

    #######################
    ### Test the network ###
    #######################
    with tf.Session() as sess:
        ###### Initialize variables #####
        sess.run(combined_init_op)
        # Always restore from trained checkpoint when testing. Base checkpoint would produce awful performance.
        saver.restore(sess, cfg['trained_checkpoint_filename'])

        ###### Test ######
        sess.run(iou_metric_initializer) # Reset the iou metric
        while True: # Iterate over entire validation set
            try:
                class_predictions, input_file_name, _ = sess.run([orig_size_pred, next_element[2], iou_metric_update])
                iou = sess.run(iou_metric_calculate) # IoU to date
                print('Cumulative mean IoU: ' + str(iou))
                save_output_image(class_predictions[0], input_file_name, cfg)
            except tf.errors.OutOfRangeError: # Thrown when end of dataset is reached
                break
        print('Final Mean IoU: ' + str(iou))

if __name__ == "__main__":
    tf.app.run()
