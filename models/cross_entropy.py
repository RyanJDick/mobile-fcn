import tensorflow as tf

def get_one_hot_labels_from_annotation(annotation_tensor, num_classes):
    """
    Returns tensor of size (height, width, num_classes) derived from annotation
    tensor of size (height, width). The returned tensor is a one-hot encoding of
    the classes in the annotation_tensor.

    Parameters
    ----------
    annotation_tensor : Tensor of size (height, width, num_classes)
        Tensor with class labels for each element
    num_classes : Int
        Number of classes. Classes are assumed to be numbered from 0 to num_classes-1
        in the annotation_tensor.

    Returns
    -------
    labels_2d_stacked : Tensor of size (height, width, num_classes).
        Tensor with labels for each pixel.
    """
    
    # Last value in the classes list should show
    # which number was used in the annotation to mask out
    # the ambigious regions or regions that should not be
    # used for training.
    valid_entries_class_labels = range(num_classes)

    # Stack the binary masks for each class
    labels_2d = list(map(lambda x: tf.equal(annotation_tensor, x),
                    valid_entries_class_labels))

    # Perform the merging of all of the binary masks into one matrix
    labels_2d_stacked = tf.stack(labels_2d, axis=2)

    # Convert tf.bool to tf.float
    # Later on in the labels and logits will be used
    # in tf.softmax_cross_entropy_with_logits() function
    # where they have to be of the float type.
    labels_2d_stacked_float = tf.to_float(labels_2d_stacked)

    return labels_2d_stacked_float

def get_one_hot_labels_from_annotation_batch(annotation_batch_tensor, num_classes):
    """Returns tensor of size (batch_size, height, width, num_classes) derived
    from annotation batch tensor. The returned tensor is a one-hot encoding of
    the classes in the annotation_batch_tensor.

    Parameters
    ----------
    annotation_batch_tensor : Tensor of size (batch_size, height, width)
        Tensor with class labels for each element
    num_classes : Int
        Number of classes. Classes are assumed to be numbered from 0 to num_classes-1
        in the annotation_tensor.

    Returns
    -------
    batch_labels : Tensor of size (batch_size, height, width, num_classes).
        Tensor with labels for each batch.
    """

    batch_labels = tf.map_fn(fn=lambda x: get_one_hot_labels_from_annotation(annotation_tensor=x, num_classes=num_classes),
                             elems=annotation_batch_tensor,
                             dtype=tf.float32)

    return batch_labels

def get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor):
    """Returns tensor of size (num_valid_entries, 3).
    Returns tensor that contains the indices of valid entries according
    to the annotation tensor. This can be used to later on extract only
    valid entries from logits tensor and labels tensor. This function is
    supposed to work with a batch input like [b, h, w] -- where b is a
    batch size, w, h -- are width and height sizes. So the output is
    a tensor which contains indexes of valid entries. This function can
    also work with a single annotation like [h, w] -- the output will
    be (num_valid_entries, 2).

    Parameters
    ----------
    annotation_batch_tensor : Tensor of size (batch_size, height, width)
        Tensor with class labels for each batch
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.

    Returns
    -------
    valid_labels_indices : Tensor of size (num_valid_entries, 3).
        Tensor with indices of valid entries
    """

    # Masked areas are assumed to have class label 255.
    # These regions could be masked due to difficulty or ambiguity.
    mask_out_class_label = 255

    # Get binary mask for the pixels that we want to use for training. We do
    # this because some pixels are marked as ambigious and we don't want to use
    # them for trainig to avoid confusing the model
    valid_labels_mask = tf.not_equal(annotation_batch_tensor,
                                        mask_out_class_label)

    valid_labels_indices = tf.where(valid_labels_mask)

    return tf.to_int32(valid_labels_indices)
