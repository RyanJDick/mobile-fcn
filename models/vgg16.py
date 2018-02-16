import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

sys.path.insert(0, '../slim_lib/models/research/slim')
import nets.vgg as vgg

# Mean values for VGG-16
from preprocessing.vgg_preprocessing import _R_MEAN, _G_MEAN, _B_MEAN

def VGG_16( image_batch_tensor,
            is_training):
    '''
    Returns the VGG16 model definition for use within the FCN model.

    Parameters
    ----------
    image_batch_tensor : [batch_size, height, width, channels] Tensor
        Tensor containing a batch of input images.

    is_training : bool
        True if network is being trained, False otherwise. This controls whether
        dropout layers should be enabled. (Dropout is only enabled during training.)

    Returns
    -------
    conv7_features:
        Features with a stride length of 32 (The coarsest layer in the VGG16
        network). The layer is referred to as 'fc7' in the original VGG16 network.
        These features feed into the fc8 logits layer in the original network;
        however the 'fc8' layer has been removed in this implementation.

    pool4_features:
        Features with a stride length of 16. (Output of the 'pool4' layer.)

    pool3_features:
        Features with a stride length of 8. (Output of the 'pool3' layer.)
    '''
    # Convert image to float32 before subtracting the mean pixel values
    image_batch_float = tf.to_float(image_batch_tensor)

    # Subtract the mean pixel value from each pixel
    mean_centered_image_batch = image_batch_float - [_R_MEAN, _G_MEAN, _B_MEAN]

    with slim.arg_scope(vgg.vgg_arg_scope()):
        # By setting num_classes to 0 the logits layer is omitted and the input
        # features to the logits layer are returned instead. This logits layer
        # will be added as part of the FCN_32s model. (Note: Some FCN
        # implementations choose to use the 'fc8' logits layer that is already
        # present in the VGG16 network instead.)

        # fc_conv_padding = 'SAME' is necessary to ensure that downsampling/
        # upsampling work as expected. So, if an image with dimensions that are
        # multiples of 32 is fed into the network, the resultant FCN pixel
        # classification will have the same dimensions as the original image.
        conv7_features, end_points = vgg.vgg_16(mean_centered_image_batch,
                                        num_classes=0,
                                        is_training=is_training,
                                        spatial_squeeze=False,
                                        fc_conv_padding='SAME')

    return conv7_features, end_points['vgg_16/pool4'], end_points['vgg_16/pool3']
