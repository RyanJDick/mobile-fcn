import numpy as np

def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2

    # Center point of pixels numbered from 0 to size - 1
    # For example, a kernel of size 4 (factor of 2) would have a center of 1.5
    # falling between pixel 1 and 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size] # Produces a tuple of two arrays the first is a
    # column vector with numbers from 0 to size-1, the second is a row vector with
    # numbers from 0 to size-1

    # Center of filter has highest weighting, dropping off linearly towards the
    # edges of the filter
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights
