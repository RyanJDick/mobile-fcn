import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import scipy
import os

def _load_data_files(data_file_path):
    '''
    Parses the data file and generates a tensor in input image filenames along
    with a tensor of annotation image filenames. (Annotation image refers to an
    image with pixel-wise class labels)

    The data file is expected to have the following format:

    path/from/data_file_path/to/input/image1 path/from/data_file_path/to/annotation/image1
    path/from/data_file_path/to/input/image2 path/from/data_file_path/to/annotation/image2
    ...

    Params:
    data_file_path      - Path to data file (describing image and annotation file locations)

    Returns:
    input_paths         - 1D list of full paths to input images
    annotation_paths    - 1D listof full paths to corresponding annotation images
    '''

    '''
    # This is the code that I threw together to populate the
    # '../data/road_seg_file_list.txt' file:
    data_file_path = '../data/road_seg_file_list.txt'
    with open(data_file_path, 'w') as f:
        for i in range(95):
            f.write('data_road/training/image_2/um_' + str(i).zfill(6) + '.png data_road/training/gt_image_2/um_road_' + str(i).zfill(6) + '.png\n')
        for i in range(96):
            f.write('data_road/training/image_2/umm_' + str(i).zfill(6) + '.png data_road/training/gt_image_2/umm_road_' + str(i).zfill(6) + '.png\n')
        for i in range(98):
            f.write('data_road/training/image_2/uu_' + str(i).zfill(6) + '.png data_road/training/gt_image_2/uu_road_' + str(i).zfill(6) + '.png\n')
    '''

    input_path_list = []
    annotation_path_list = []
    data_folder_path = os.path.dirname(os.path.abspath(data_file_path))
    with open(data_file_path, 'rt') as f:
        for line in f:
            input_path, annotation_path = line.split(" ") # Split paths on space
            input_path_list.append(os.path.join(data_folder_path, input_path.strip()))
            annotation_path_list.append(os.path.join(data_folder_path, annotation_path.strip()))
    return np.array(input_path_list), np.array(annotation_path_list)

def _load_test_data_files(test_data_file_path):
    input_test_path_list = []
    test_data_folder_path = os.path.dirname(os.path.abspath(test_data_file_path))
    with open(test_data_file_path, 'rt') as f:
        for line in f:
            input_test_path_list.append(os.path.join(test_data_folder_path, line.strip()))
    return np.array(input_test_path_list)

def _parse_images_function(img_file, annotation_file, class_mapping):
    '''
    Function to be applied to dataset.
    Reads and decodes input image and annotation image from files.

    The annotation image is converted to a pixel-wise class mapping.
    This pixel_labels map has dimensions (img_width, img_height, 1).
    A class index of 0 indicates that the pixel has been masked and should not
    be considered for training purposes.
    '''
    img_decoded = scipy.misc.imread(img_file, mode = 'RGB')
    cached_annotation_map_file = os.path.splitext(annotation_file.decode("utf-8"))[0] + '_map_cache'

    if os.path.isfile(cached_annotation_map_file + '.npy') : # Check if the cached file exists
        annotation_map = np.load(cached_annotation_map_file + '.npy')
    else:
        annotation_decoded = scipy.misc.imread(annotation_file, mode = 'RGB')
        annotation_image_shape = annotation_decoded.shape

        # New approach:
        class_map = {}
        for i in range(len(class_mapping)):
            klass = class_mapping[i]
            class_hash = klass[0] + 256 * klass[1] + 256 * 256 * klass[2]
            class_map[class_hash] = i

        def class_mapper(a):
            a_hash = a[0] + 256 * a[1] + 256 * 256 * a[2]
            #print('a_hash: ' + str(a_hash))
            #print('class_mapping: ' + str(class_map))
            if a_hash in class_map:
                return class_map[a_hash]
            else:
                return 255
        annotation_map = np.apply_along_axis(class_mapper, 2, annotation_decoded)
        annotation_map = annotation_map.astype(np.uint8)
        annotation_map = np.expand_dims(annotation_map, axis = -1)
        '''
        # Value of 255 means that the pixel is invalid due to ambiguity (or some other
        # reason) and should not be used for classification
        annotation_map = np.full((annotation_image_shape[0], annotation_image_shape[1], 1), 255, dtype=np.uint8) # Added 3rd dimension so that it can be treated as an image with a single channel

        # Compare each pixel against colors in class_mapping (This process is very slow, so save the result to a cache file)
        for row in range(annotation_image_shape[0]):
            for col in range(annotation_image_shape[1]):
                for i in range(len(class_mapping)):
                    if np.array_equal(annotation_decoded[row][col], class_mapping[i]):
                        annotation_map[row][col][0] = i
        '''

        print("Saving annotation map cache: " + os.path.basename(cached_annotation_map_file))
        np.save(cached_annotation_map_file, annotation_map) # Cache the result

    return img_decoded, annotation_map

def _parse_test_images_function(img_file, annotation_file, class_mapping):
    img_decoded, annotation_map = _parse_images_function(img_file, annotation_file, class_mapping)
    return img_decoded, annotation_map, img_file

def _parse_pred_input_image_function(img_file):
    img_decoded = scipy.misc.imread(img_file, mode = 'RGB')
    img_size = np.array([img_decoded.shape[0], img_decoded.shape[1]])
    return img_decoded, img_file, img_size

def _preprocess_images_function(img, annotation, img_height, img_width):
    '''
    Resize the input image and annotation image for compatibility with the
    network architecture.

    When resizing the annotation, the NEAREST_NEIGHBOR method (no averaging) is
    used to guarantee that all pixels have values of 0 or 1.
    '''

    img.set_shape([None, None, 3])
    img_resized = tf.image.resize_images(img, [img_height, img_width], method=tf.image.ResizeMethod.BICUBIC)
    img_standardized = tf.to_float(img_resized)

    annotation.set_shape([None, None, 1])
    annotation_resized = tf.image.resize_images(annotation, [img_height, img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return img_standardized, annotation_resized

def _preprocess_test_images_function(img, annotation, img_file_name, img_height, img_width):
    '''
    Resize the input image, but LEAVE the annotation image at its ORIGINAL SIZE.

    The predictions must be resized back to the original image size before
    comparing against the annotation image.
    '''
    img.set_shape([None, None, 3])
    img_resized = tf.image.resize_images(img, [img_height, img_width], method=tf.image.ResizeMethod.BICUBIC)
    img_standardized = tf.to_float(img_resized)

    annotation.set_shape([None, None, 1])

    return img_standardized, annotation, img_file_name

def _preprocess_pred_input_image_function(img, img_file, img_size, img_height, img_width):
    '''
    Preprocess a single input image when there is no annotation file and we are
    just interested in making predictions on the input image.

    Resize the input image for compatibility with the network architecture.

    Note: Predictions from network should be resized back to original image size.
    '''
    img.set_shape([None, None, 3])
    img_resized = tf.image.resize_images(img, [img_height, img_width], method=tf.image.ResizeMethod.BICUBIC)
    img_standardized = tf.to_float(img_resized)

    img_size.set_shape([2])

    return img_standardized, img_file, img_size

def build_train_val_datasets(cfg):
    '''
    Builds a train and validation dataset from the provided data_file_path.
    Each data element contains an image of size (h, w, 3) and an annotation file
    of size (h, w, 1).
    '''
    input_paths, annotation_paths = _load_data_files(cfg['data_index_file'])
    # Split file paths into train and validation sets:
    input_train, input_val, annotation_train, annotation_val = train_test_split(input_paths, annotation_paths, test_size=cfg['test_size'])

    # Train and validation datasets consisting of tuples of paths to input image
    # and paths to class annotation images:
    train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(input_train), tf.constant(annotation_train)))
    val_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(input_val), tf.constant(annotation_val)))

    # Load images, apply preprocessing, and image augmentation for training images
    train_dataset = train_dataset.map(lambda img_file, annotation_file: tuple(tf.py_func(_parse_images_function, [img_file, annotation_file, cfg['classes']], [tf.uint8, tf.uint8])))
    train_dataset = train_dataset.map(lambda img_file, annotation_file: _preprocess_images_function(img_file, annotation_file, cfg['scaled_img_height'], cfg['scaled_img_width']))
    #train_dataset = train_dataset.map(_augment_images_function)

    # Load images and apply preprocessing for validation images (not necessary to perform data augmentation)
    val_dataset = val_dataset.map(lambda img_file, annotation_file: tuple(tf.py_func(_parse_images_function, [img_file, annotation_file, cfg['classes']], [tf.uint8, tf.uint8])))
    val_dataset = val_dataset.map(lambda img_file, annotation_file: _preprocess_images_function(img_file, annotation_file, cfg['scaled_img_height'], cfg['scaled_img_width']))

    return train_dataset, val_dataset

def build_test_dataset(cfg):
    '''
    Builds a single test dataset from the provided data_file_path. Each data
    element contains an image of size (h, w, 3) and an annotation file of size
    (h, w, 1).
    '''
    input_paths, annotation_paths = _load_data_files(cfg['test_data_index_file'])

    # Test dataset consisting of tuples of paths to input image and paths to
    # class annotation images:
    test_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(input_paths), tf.constant(annotation_paths)))

    test_dataset = test_dataset.map(lambda img_file, annotation_file: tuple(tf.py_func(_parse_test_images_function, [img_file, annotation_file, cfg['classes']], [tf.uint8, tf.uint8, tf.string])))
    test_dataset = test_dataset.map(lambda img_file, annotation_file, img_file_name: _preprocess_test_images_function(img_file, annotation_file, img_file_name, cfg['scaled_img_height'], cfg['scaled_img_width']))

    return test_dataset


def build_prediction_dataset(cfg):
    '''
    Builds a single dataset consisting only of input files on which we want to
    pixelwise predictions (there is no annotation file to compare against).
    '''
    test_img_paths, annotation_paths = _load_data_files(cfg['prediction_input_file']) # When making predictions only, we discard the annotations

    test_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(test_img_paths))

    test_dataset = test_dataset.map(lambda img_file: tuple(tf.py_func(_parse_pred_input_image_function, [img_file], [tf.uint8, tf.string, tf.int64])))
    test_dataset = test_dataset.map(lambda img, img_file, img_size: _preprocess_pred_input_image_function(img, img_file, img_size, cfg['scaled_img_height'], cfg['scaled_img_width']))

    return test_dataset
