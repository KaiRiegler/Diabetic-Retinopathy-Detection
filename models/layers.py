import gin
import tensorflow as tf


@gin.configurable
def se_block(input_tensor, reduction):
    '''
    Squeeze and Excitation (SE) Block

        Parameters:
                input_tensor (Tensor): Input tensor to be transformed
                reduction (int): Reduction factor for channel compression

        Returns:
                Tensor after applying SE transformation
    '''
    channels = input_tensor.shape[-1]
    reduced_channels = max(1, channels * reduction)

    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    se = tf.keras.layers.Conv2D(reduced_channels, 1, padding='same', activation='relu')(se)
    se = tf.keras.layers.Conv2D(channels, 1, padding='same', activation='sigmoid')(se)

    return tf.keras.layers.Multiply()([input_tensor, se])


@gin.configurable
def mbconv_block(input_tensor, filters, kernel_size, strides, expansion_factor, use_se, reduction_mbconv):
    '''
    Mobile Inverted Bottleneck Convolution (MBConv) Block

        Parameters:
                input_tensor (Tensor): Input tensor to the MBConv Block
                filters (int): Number of output filters
                kernel_size (tuple: (int x int)): Size of the convolution kernel
                strides (int): Stride for depthwise convolution kernel
                expansion_factor (int): Factor to expand the input channels
                use_se (bool): Whether to include the SE block or not
                reduction_mbconv (float): Reduction factor for se-block

        Returns:
                out (Tensor): Transformed tensor after MBConv Block
    '''
    input_channels = input_tensor.shape[-1]
    expanded_channels = int(input_channels * expansion_factor)

    # Expansion
    if expansion_factor != 1:
        out = tf.keras.layers.Conv2D(expanded_channels, 1, padding='same', use_bias=False)(input_tensor)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out) 
    else:
        out = input_tensor

    # Depthwise Convolution
    out = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)

    # Squezze and Excitation (optional)
    if use_se:
        out = se_block(out, reduction_mbconv)

    # Projection
    out = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)

    # Residual Connection (only if stride is 1 and input/output shapes match)
    if strides == 1 and input_channels == filters:
        out = tf.keras.layers.add([input_tensor, out])

    return out


@gin.configurable
def fused_mbconv_block(input_tensor, filters, strides, expansion_factor, use_se, reduction_fmbconv):
    '''
    Fused Mobile Inverted Bottleneck Convolution (Fused-MBConv) Block

        Parameters:
                input_tensor (Tensor): Input tensor to the MBConv Block
                filters (int): Number of output filters
                strides (int): Stride for depthwise convolution kernel
                expansion_factor (int): Factor to expand the input channels
                use_se (bool): Whether to include the SE block or not
                reduction_fmbconv (float): Reduction factor for se-block

        Returns:
                out (Tensor): Transformed tensor after MBConv Block
    '''
    input_channels = input_tensor.shape[-1]
    expanded_channels = int(input_channels * expansion_factor)

    # Expansion
    if expansion_factor != 1:
        out = tf.keras.layers.Conv2D(expanded_channels, 3, strides, padding='same', use_bias=False)(input_tensor)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)
        
    else:
        out = input_tensor

    # Squezze and Excitation (optional)
    if use_se:
        out = se_block(out, reduction_fmbconv)

    # Projection
    out = tf.keras.layers.Conv2D(filters, 1 if expansion_factor != 1 else 3, strides=1 if expansion_factor != 1 else strides, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    if expansion_factor == 1:
        out = tf.keras.layers.ReLU()(out)

    # Residual Connection (only if stride is 1 and input/output shapes match)
    if strides == 1 and input_channels == filters:
        out = tf.keras.layers.add([input_tensor, out])

    return out

