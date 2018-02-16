import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

sys.path.insert(0, '../slim_lib/models/research/slim')
import nets.mobilenet_v1 as mobilenet

from preprocessing.vgg_preprocessing import _R_MEAN, _G_MEAN, _B_MEAN

def MOBILENET( image_batch_tensor, is_training):
    '''
    Returns the MobileNet model definition for use within the MobileSeg model.

    Parameters
    ----------
    image_batch_tensor : [batch_size, height, width, channels] Tensor
        Tensor containing a batch of input images.

    is_training : bool
        True if network is being trained, False otherwise. This controls whether
        dropout layers should be enabled, and the behaviour of the batchnorm
        layers.

    Returns
    -------


    conv13_features:
        Features with a stride length of 32. The layer is referred to as
        'MobilenetV1/Conv2d_13_pointwise/Conv2D' in the MobileNet Tensorflow
        implementation. These features feed into the average pooling layer in
        the original network; however the pooling layer and subsequent fc and
        softmax layers have been removed in this implementation.

    conv11_features:
        Features with a stride length of 16. (Output of the
        'MobilenetV1/Conv2d_11_pointwise/Conv2D' layer.)

    conv5_features:
        Features with a stride length of 8. (Output of the
        'MobilenetV1/Conv2d_5_pointwise/Conv2D' layer.)
    '''
    # Convert image to float32 before subtracting the mean pixel values
    image_batch_float = tf.to_float(image_batch_tensor)

    # Subtract the mean pixel value from each pixel
    mean_centered_image_batch = image_batch_float - [_R_MEAN, _G_MEAN, _B_MEAN]

    with slim.arg_scope(mobilenet.mobilenet_v1_arg_scope(is_training = is_training)):
        conv13_features, end_points = mobilenet.mobilenet_v1_base(image_batch_tensor,
                                          final_endpoint='Conv2d_13_pointwise',
                                          min_depth=8,
                                          depth_multiplier=1.0,
                                          conv_defs=None,
                                          output_stride=None,
                                          scope=None)

    return conv13_features, end_points['Conv2d_11_pointwise'], end_points['Conv2d_5_pointwise']
