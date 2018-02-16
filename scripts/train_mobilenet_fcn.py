'''
Ryan Dick

November 7, 2017
'''
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import scipy
import sys
import json
from kitti_dataset_factory import build_train_val_datasets
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, '../models')
from mobilenet import MOBILENET
from fcn import FCN_EXTENSION
from cross_entropy import get_valid_entries_indices_from_annotation_batch, get_one_hot_labels_from_annotation_batch

flags = tf.app.flags

flags.DEFINE_string('config', None, 'Path to configuration .json file.')

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
    train_dataset, val_dataset = build_train_val_datasets(cfg)

    train_dataset = train_dataset.batch(1) # Batch size of 1 is really like a large batch size, because we are updating based on the loss at every pixel
    val_dataset = val_dataset.batch(1)
    # Create handle to control which dataset is fed into the model:
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    next_element = iterator.get_next()

    # Setup training and validation dataset iterators
    train_iterator = train_dataset.make_initializable_iterator()
    val_iterator = val_dataset.make_initializable_iterator()

    ############################
    ### Initialize the model ###
    ############################
    conv13_features, conv11_features, conv5_features = MOBILENET(image_batch_tensor = next_element[0],
                                                            is_training = True)
    output_logits_dict = FCN_EXTENSION( conv13_features,
                                        conv11_features,
                                        conv5_features,
                                        len(cfg['classes']))

    #######################################
    ### Process annotation/label tensor ###
    #######################################
    annotation_batch_tensor = tf.squeeze(next_element[1], axis=-1) # (b x h x w x 1) -> (b x h x w)after squeeze

    # Convert annotation tensor to one-hot encoded labels for comparison against upsampled logits
    labels_one_hot_batch_tensor = get_one_hot_labels_from_annotation_batch(annotation_batch_tensor=annotation_batch_tensor,
                                                                            num_classes=len(cfg['classes']))
    # Get (b, w, h) indices of pixel that are not masked out
    valid_batch_indices = get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor=annotation_batch_tensor)
    # Keep only valid pixels from the labels, one-hot labels, and logits
    valid_labels_one_hot_batch_tensor = tf.gather_nd(params=labels_one_hot_batch_tensor,
                                                        indices=valid_batch_indices)
    valid_labels_batch_tensor = tf.gather_nd(params=annotation_batch_tensor,
                                                indices=valid_batch_indices)

    training_stages = ['fcn32', 'fcn16', 'fcn8']
    learning_rates = {'fcn32':1, 'fcn16':0.1, 'fcn8':0.01}
    #####################################
    ### FCN-32s, FCN-16s, FCN-8s Loss ###
    #####################################
    train_steps = {}
    cross_entropy_means = {}
    iou_metric_calculators = {}
    iou_metric_updaters = {}
    iou_metric_initializers = {}
    # Each batchnorm layer has an update op that must run during training to
    # update the moving averages tracked by the batchnorm layer:
    batch_norm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    for stage in training_stages:
        valid_logits_batch_tensor = tf.gather_nd(params=output_logits_dict[stage],
                                                    indices=valid_batch_indices)
        cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                            labels=valid_labels_one_hot_batch_tensor)

        # Normalize the cross entropy -- the number of elements is different during
        # each step due to masked out regions
        cross_entropy_means[stage] = tf.reduce_mean(cross_entropies)

        with tf.variable_scope(stage + '_sgd'):
            with tf.control_dependencies(batch_norm_update_ops): # Ensure that batchnorm statistics get updated at every train step
                train_steps[stage] = tf.train.MomentumOptimizer(learning_rate=cfg['learning_rate'] * learning_rates[stage],
                                                            momentum=cfg['momentum']).minimize(cross_entropy_means[stage])

        valid_pred_batch_tensor = tf.argmax(valid_logits_batch_tensor, axis = 1)

        iou_metric_calculators[stage], iou_metric_updaters[stage] = tf.metrics.mean_iou(valid_labels_batch_tensor, valid_pred_batch_tensor, len(cfg['classes']), name=stage+'_iou_metric')
        iou_metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=stage+'_iou_metric')
        iou_metric_initializers[stage] = tf.variables_initializer(var_list=iou_metric_vars)

    ##########################
    ### Add prediction ops ###
    ##########################
    # Note: We use FCN-8s output for evaluation because all weights were
    # zero-initialized, so the untrained layers will not affect the output. In
    # other words, initially (before training FCN-16s and FCN-8s), the FCN-8s
    # output is the same as the FCN-32s output.
    pred = tf.argmax(output_logits_dict['fcn8'], axis=3) # Predictions for all pixels (including masked)
    probabilities = tf.nn.softmax(output_logits_dict['fcn8']) # Probabilities for each class for each pixel (including masked)

    ##################################
    ### Initialize model variables ###
    ##################################
    if not cfg['restore_from_trained_checkpoint'] : # Load pretrained mobilenet Imagenet weights
        variables_to_restore = slim.get_model_variables('MobilenetV1')
        init_mobilenet_fn = slim.assign_from_checkpoint_fn(model_path=cfg['base_checkpoint_filename'], var_list=variables_to_restore)

    global_vars_init_op = tf.global_variables_initializer()
    local_vars_init_op = tf.local_variables_initializer()
    combined_init_op = tf.group(local_vars_init_op, global_vars_init_op)

    # Create initializer that can be run at the start of each stage to ensure
    # that the next stage is initialized as desired
    stage_reinitializers = {}
    global_variable_list = tf.global_variables()

    fcn16_fc_weights = [v for v in global_variable_list if v.name == "fcn16/downsampled_by_16_fc/weights:0"][0]
    fcn16_upsample_by_2_filter = [v for v in global_variable_list if v.name == "fcn16/upsample_by_2:0"][0]
    stage_reinitializers['fcn16'] = tf.variables_initializer([fcn16_fc_weights, fcn16_upsample_by_2_filter])

    fcn8_fc_weights = [v for v in global_variable_list if v.name == "fcn8/downsampled_by_8_fc/weights:0"][0]
    fcn8_upsample_by_2_filter = [v for v in global_variable_list if v.name == "fcn8/upsample_by_2:0"][0]
    stage_reinitializers['fcn8'] = tf.variables_initializer([fcn8_fc_weights, fcn8_upsample_by_2_filter])

    # We need this to save only model variables and omit
    # optimization-related and other variables.
    model_variables = slim.get_model_variables()
    saver = tf.train.Saver(model_variables) # saver is used to save and load full model checkpoints

    #######################
    ### Train the model ###
    #######################
    with tf.Session() as sess:
        # Get training and validation iterator handles to feed into the model
        training_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(val_iterator.string_handle())

        ###### Initialize variables #####
        sess.run(combined_init_op)
        if cfg['restore_from_trained_checkpoint'] :
            saver.restore(sess, cfg['trained_checkpoint_filename'])
        else :
            init_mobilenet_fn(sess) # Load base checkpoint (ex: mobilenet weights before training FCN32 weights)

        # Best performance achieved on the validation set so far:
        best_validation_metric = None

        for stage in training_stages : # Train FCN-32s, then FCN-16s, then FCN-8s
            print('***** Starting training of ' + stage + ' model. *****')

            # If this is the fcn16 or fcn8 training stage
            if stage != 'fcn32':
                print('Reverting to checkpoint of best model so far')
                saver.restore(sess, cfg['trained_checkpoint_filename'])
                # Reinitialize the weights for the next layer (to zeros for the
                # 'fully connected'-like layer and to bilinear interpolation for the upsampling layer)
                sess.run(stage_reinitializers[stage])

            # Number of consecutive times that validation set performance did not
            # improve. If this reaches 4, continue to next training stage:
            validation_did_not_improve_count = 0

            # Set to True if validation performance has not improved for 4 consecutive tests
            done_stage = False

            # Training step count. Every 30 steps, test on validation.
            train_count = 0

            epoch = 0
            # Train for at most 30 epochs in each stage
            # (we will continue to next stage sooner if performance on the
            # validation set is not improving)
            while epoch < 30 and not done_stage:
                epoch += 1
                sess.run(train_iterator.initializer) # Re-initialize the training iterator

                while True: # Iterate over entire training set
                    try:
                        ##### Training Step #####
                        train_count += 1
                        cross_entropy, _ = sess.run([cross_entropy_means[stage], train_steps[stage]],
                                                            feed_dict={handle: training_handle})
                        print(str(train_count) + " Current loss: " + str(cross_entropy))
                    except tf.errors.OutOfRangeError: # Thrown when end of training dataset is reached
                        break

                    # After every batch of training images:
                    # - Run validation
                    # - Save a checkpoint if the model has improved
                    if train_count % cfg['train_steps_between_validation'] == 0:
                        ##### Validation #####
                        print("Performing validation:")
                        sess.run(val_iterator.initializer) # Re-initialize the validation val_iterator
                        sess.run(iou_metric_initializers[stage]) # Reset the iou metric
                        validation_metric_total = 0
                        validation_count = 0
                        while True: # Iterate over entire validation set
                            try:
                                sess.run(iou_metric_updaters[stage], feed_dict={handle: validation_handle})
                                iou = sess.run(iou_metric_calculators[stage])
                                print('Cumulative mean IoU: ' + str(iou)) # This is not the iou for a single image, but rather the iou calculated up to this point
                            except tf.errors.OutOfRangeError: # Thrown when end of validation dataset is reached:
                                break
                        final_iou = iou

                        ##### Save Checkpoint #####
                        # If this is the first checkpoint, or the model performed better than the best so far on the validation set
                        if  (best_validation_metric is None) or (final_iou > best_validation_metric) :
                            save_path = saver.save(sess, cfg['trained_checkpoint_filename'])
                            print("Model saved in file: %s" % save_path)
                            best_validation_metric = final_iou
                            validation_did_not_improve_count = 0 # reset count
                        else :
                            print("Not saving checkpoint, because model had worse performance on validation set.")
                            validation_did_not_improve_count += 1

                        if validation_did_not_improve_count >= 4:
                            done_stage = True
                            break

if __name__ == "__main__":
    tf.app.run()
