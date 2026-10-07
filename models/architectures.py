import gin
import gin.config
import tensorflow as tf
import math

from models.layers import mbconv_block, fused_mbconv_block


@gin.configurable
class EfficientNetV2:
    '''
    EfficientNetV2 Architecture Class

        Attributes:
                input_shape (Tensor): Shape of the input data (height, width, channels)
                num_classes (int): Number of output classes for classification
                width_factor (float): Scaling factor for the network's width
                depth_factor (float): Scaling factor for the network's depth
                use_se_mbconv (bool): Whether to include Squeeze-and-Excitation blocks in MBConv blocks
                use_se_fmbconv (bool): Whether to include Squeeze-and-Excitation blocks in fused MBConv blocks
    '''
    def __init__(self, input_shape, num_classes, width_factor, depth_factor, use_se_mbconv, use_se_fmbconv):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.width_factor = width_factor
        self.depth_factor = depth_factor
        self.use_se_mbconv = use_se_mbconv
        self.use_se_fmbconv = use_se_fmbconv

    def repeat_mbconv_block(self, x, filters, kernel_size, strides, expansion_factor, repeats):
        '''
        Repeat an MBConv Block a specified number of times

            Parameters:
                    x (Tensor): Input tensor
                    filters (int): Number of filters for the block
                    kernel_size (Tuple: (int x int)): Kernel size for depthwise convolution
                    strides (int): Stride for the block
                    expansion_factor (int): Factor to expand input channels
                    repeats (int): Number of repetitions for the block

            Returns:
                    x (Tensor): Output tensor after repeated blocks
        '''
        for y in range(repeats):
            if y > 0:
                strides = 1
                # input_filters = output_filters
            x = mbconv_block(x, filters, kernel_size, strides, expansion_factor, use_se=self.use_se_mbconv)
        return x
    
    def repeat_fused_mbconv_block(self, x, filters, strides, expansion_factor, repeats):
        '''
        Repeat an Fused-MBConv Block a specified number of times

            Parameters:
                    x (Tensor): Input tensor
                    filters (int): Number of filters for the block
                    strides (int): Stride for the block
                    expansion_factor (int): Factor to expand input channels
                    repeats (int): Number of repetitions for the block

            Returns:
                    x (Tensor): Output tensor after repeated blocks
        '''
        for y in range(repeats):
            if y > 0:
                strides = 1
                # input_filters = output_filters
            x = fused_mbconv_block(x, filters, strides, expansion_factor, use_se=self.use_se_fmbconv)
        return x
    
    @gin.configurable
    def build_eff_net_v2(self, dropout_rate, dense_units):
        '''
        Build the EfficientNetV2 model

            Parameters:
                    dropout_rate (float): dropout rate
                    dense_units (int): Number of dense units

            Returns:
                    model (tf.keras.Model): Compiled EfficientNetV2 model
        '''
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # Stem
        out = tf.keras.layers.Conv2D(int(32 * self.width_factor), 3, strides=2, padding='same', use_bias=False)(inputs)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)

        # MBConv Blocks
        out = self.repeat_fused_mbconv_block(out, filters=int(16 * self.width_factor), strides=1, expansion_factor=1, repeats=max(1, math.ceil(1 * self.depth_factor)))
        out = self.repeat_fused_mbconv_block(out, filters=int(32 * self.width_factor), strides=2, expansion_factor=4, repeats=max(1, math.ceil(2 * self.depth_factor)))
        out = self.repeat_fused_mbconv_block(out, filters=int(48 * self.width_factor), strides=2, expansion_factor=4, repeats=max(1, math.ceil(2 * self.depth_factor)))

        out = self.repeat_mbconv_block(out, filters=int(96 * self.width_factor), kernel_size=3, strides=2, expansion_factor=4, repeats=max(1, math.ceil(3 * self.depth_factor)))
        out = self.repeat_mbconv_block(out, filters=int(112 * self.width_factor), kernel_size=3, strides=1, expansion_factor=6, repeats=max(1, math.ceil(5 * self.depth_factor)))
        out = self.repeat_mbconv_block(out, filters=int(192 * self.width_factor), kernel_size=3, strides=2, expansion_factor=6, repeats=max(1, math.ceil(8 * self.depth_factor)))

        # Head
        out = tf.keras.layers.Conv2D(int(1280 * self.width_factor), 1, padding='same', use_bias=False)(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)

        out = tf.keras.layers.GlobalAveragePooling2D()(out)
        if dense_units:
            out = tf.keras.layers.Dense(dense_units, activation='relu')(out)
        out = tf.keras.layers.Dropout(dropout_rate)(out)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(out)

        model = tf.keras.models.Model(inputs, outputs, name='efficient_net_v2')
        return model
    

@gin.configurable
class TransferModel():
    '''
    A class for building transfer learning models using EfficientNet variants

        Attributes:
                input_shape (tuple): The shape of the input images
                num_classes (int): The number of output classes
                dropout_rate (float): Dropout rate applied to the classifier head
    '''
    def __init__(self, input_shape, num_classes, dropout_rate):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    @gin.configurable
    def build_pretrained_model(self, name, unfreezed_layers):
        '''
        Builds a transfer learning model using EfficientNet variants as the base

        Parameters:
            name (str): The name of the EfficientNet variant ('v2s', 'v2b0', 'v2b1')
            unfreezed_layers (int): The number of layers to unfreeze for fine-tuning

        Returns:
            tf.keras.Model: A compiled transfer learning model

        Raises:
            ValueError: If the specified model name is not recognized
        '''
        if name == "v2s":
            model = tf.keras.applications.EfficientNetV2S(include_top=False, weights="imagenet", input_shape=self.input_shape, include_preprocessing=False)
        elif name == "v2b0":
            model = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=self.input_shape, include_preprocessing=False)
        elif name == "v2b1":
            model = tf.keras.applications.EfficientNetV2B1(include_top=False, weights="imagenet", input_shape=self.input_shape, include_preprocessing=False)
        else:
            raise ValueError("Could not find Model.") 

        model.trainable = False
        if unfreezed_layers != 0:
            model.trainable = True
            for layer in model.layers[:len(model.layers)-unfreezed_layers]:
                layer.trainable = False     

        inputs = tf.keras.layers.Input(shape=self.input_shape)
        out = model(inputs, training=False)
        out = tf.keras.layers.GlobalAveragePooling2D()(out)
        out = tf.keras.layers.Dropout(self.dropout_rate)(out)
        out = tf.keras.layers.Dense(self.num_classes, activation='softmax')(out)
        model = tf.keras.Model(inputs, out, name='transfer')

        return model
    