import tensorflow as tf
from upsampling_utils import bilinear_upsample_weights

slim = tf.contrib.slim

def FCN_EXTENSION(downsampled_by_32_features,
        downsampled_by_16_features,
        downsampled_by_8_features,
        num_classes):
    '''
    Adds the FCN8s, FCN16s, and FCN32s on top of a base model.

    Parameters
    ----------
    downsampled_by_32_features : [batch_size, height / 32, width / 32, ?] Tensor
        Features that have been downsamled by factor 32 from original image size.

    downsampled_by_16_features : [batch_size, height / 16, width / 16, ?] Tensor
        Features that have been downsamled by factor 32 from original image size.

    downsampled_by_8_features : [batch_size, height / 8, width / 8, ?] Tensor
        Features that have been downsamled by factor 32 from original image size.

    num_classes : Int
        Number of segmentation classes.

    Returns
    -------
    output_logits_dict :
        A dict containing the pixel-wise logit output for FCN-32s, FCN-16s,
        FCN-8s. The logits tensors can be accessed using the following keys:
            - 'fcn32'
            - 'fcn16'
            - 'fcn8'
        Each tensor has the following shape: [batch_size, height, width, num_classes]
    '''


    output_logits_dict = {}
    train_op_dict = {}


    ################
    # FCN-32s Model
    ################
    with tf.variable_scope("fcn32"):
        downsampled_by_32_logits = slim.conv2d( downsampled_by_32_features,
                                    num_classes,
                                    [1, 1],
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    scope='downsampled_by_32_fc')

        # Upsample to original image size
        fcn32_output_logits = add_upsampling_layer(downsampled_by_32_logits, 32, num_classes, False)
        output_logits_dict['fcn32'] = fcn32_output_logits

    ################
    # FCN-16s Model
    ################
    with tf.variable_scope("fcn16"):
        # Upsample downsampled_by_32_logits by factor of 2, so that they have the same
        # stride as downsampled_by_16_features
        downsampled_by_32_logits_upsampled_by_2 = add_upsampling_layer(downsampled_by_32_logits, 2, num_classes, True)

        # Zero-initialize the weights to start training with the same
        # accuracy that we ended training FCN-32s
        downsampled_by_16_logits = slim.conv2d(downsampled_by_16_features,
                                   num_classes,
                                   [1, 1],
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   weights_initializer=tf.zeros_initializer,
                                   scope='downsampled_by_16_fc')

        fused_down16_and_down32_logits = downsampled_by_16_logits + downsampled_by_32_logits_upsampled_by_2

        # Upsample to original image size
        fcn16_output_logits = add_upsampling_layer(fused_down16_and_down32_logits, 16, num_classes, False)
        output_logits_dict['fcn16'] = fcn16_output_logits

    ################
    # FCN-8s Model
    ################
    with tf.variable_scope("fcn8"):
        # Upsample fused_down16_and_down32_logits by factor of 2,
        # so that it has the same stride as pool3_features
        fused_down16_and_down32_logits_upsampled_by_2 = add_upsampling_layer(fused_down16_and_down32_logits, 2, num_classes, True)

        # Zero-initialize the weights to start training with the same
        # accuracy that we ended training FCN-16s
        downsampled_by_8_logits = slim.conv2d(downsampled_by_8_features,
                                   num_classes,
                                   [1, 1],
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   weights_initializer=tf.zeros_initializer,
                                   scope='downsampled_by_8_fc')

        fused_down32_down16_down8_logits = downsampled_by_8_logits + fused_down16_and_down32_logits_upsampled_by_2

        # Upsample to original image size
        fcn8_output_logits = add_upsampling_layer(fused_down32_down16_down8_logits, 8, num_classes, False)
        output_logits_dict['fcn8'] = fcn8_output_logits

        return output_logits_dict


def add_upsampling_layer(   input_logits,
                            upsample_factor,
                            num_classes,
                            trainable):
    '''
    Adds a bilinear interpolation upsampling layer to the model.

    Parameters
    ----------
    input_logits : [batch_size, in_height, in_width, num_classes] Tensor
        Tensor to be upsampled.

    upsample_factor : Int
        Factor by which to upsample the input_logits.

    num_classes : Int
        Number of classes. Equal to number of channels of input_logits and
        upsampled_logits.

    trainable : Bool
        Controls whether the weights of the upsampling filter are constant
        (trainable = false) or variable (trainable = true). All filters are
        initialized to perform bilinear interpolation regardless of whether they
        are trainable.

    Returns
    -------
    upsampled_logits : [batch_size, in_height * upsample_factor, in_width * upsample_factor, num_classes]
        The output tensor from the upsampling layer.

    '''

    # Calculate the ouput size of the upsampled tensor
    input_logits_shape = tf.shape(input_logits)
    upsampled_logits_shape = tf.stack(   [
                                          input_logits_shape[0],
                                          input_logits_shape[1] * upsample_factor,
                                          input_logits_shape[2] * upsample_factor,
                                          input_logits_shape[3]
                                         ])

    # Get bilinear upsample kernel
    upsample_filter_np = bilinear_upsample_weights(factor=upsample_factor, number_of_classes=num_classes)
    upsample_filter_tensor = tf.get_variable('upsample_by_' + str(upsample_factor),
                                            initializer=tf.constant(upsample_filter_np),
                                            trainable=trainable)
    #if trainable :
    #    upsample_filter_tensor = tf.Variable(upsample_filter_np)
    #else :
    #    upsample_filter_tensor = tf.constant(upsample_filter_np)

    # Perform the upsampling
    upsampled_logits = tf.nn.conv2d_transpose(input_logits,
                                              upsample_filter_tensor,
                                              output_shape=upsampled_logits_shape,
                                              strides=[1, upsample_factor, upsample_factor, 1])
    return upsampled_logits
