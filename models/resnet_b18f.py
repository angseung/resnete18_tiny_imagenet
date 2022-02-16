import tensorflow as tf
import larq as lq

# set Network common constraint
input_quantizer = lq.quantizers.SteSign(clip_value=1.25)
kernel_quantizer = lq.quantizers.SteSign(clip_value=1.25)
kernel_constraint = lq.constraints.WeightClip(clip_value=1.25)


def residual_block_b18_v0(x: tf.Tensor, filters: int, strides: int = 1) -> tf.Tensor:
    downsample = x.get_shape().as_list()[-1] != filters

    if downsample:

        residual = lq.layers.QuantConv2D(
            filters,
            kernel_size=1,
            strides=strides,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_constraint=kernel_constraint,
            use_bias=False,
        )(x)
        residual = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(
            residual
        )
    else:
        residual = x

    x = lq.layers.QuantConv2D(
        filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        input_quantizer=input_quantizer,
        kernel_quantizer=kernel_quantizer,
        kernel_constraint=kernel_constraint,
        kernel_initializer="glorot_normal",
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    x = lq.layers.QuantConv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        input_quantizer=input_quantizer,
        kernel_quantizer=kernel_quantizer,
        kernel_constraint=kernel_constraint,
        kernel_initializer="glorot_normal",
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    return tf.keras.layers.add([x, residual])


def residual_block_b18_v1(x: tf.Tensor, filters: int, strides: int = 1) -> tf.Tensor:
    downsample = x.get_shape().as_list()[-1] != filters

    if downsample:

        residual = lq.layers.QuantConv2D(
            filters,
            kernel_size=1,
            strides=strides,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_constraint=kernel_constraint,
            use_bias=False,
        )(x)
        residual = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(
            residual
        )
    else:
        residual = x

    x = lq.layers.QuantConv2D(
        filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        input_quantizer=input_quantizer,
        kernel_quantizer=kernel_quantizer,
        kernel_constraint=kernel_constraint,
        kernel_initializer="glorot_normal",
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    return tf.keras.layers.add([x, residual])


spec = ([2, 2, 2, 2], [64, 128, 256, 512])


def resnet_b18_v0(input_shape, num_classes, weight_path=None):
    x = tf.keras.Input(shape=input_shape)
    #
    y = tf.keras.layers.QuantConv2D(64,
                                    kernel_size=3,
                                    padding="same",
                                    kernel_initializer="he_normal",
                                    use_bias=False,
                                    )(x)

    y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(y)
    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(y)
    y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(y)

    for block, (layers, filters) in enumerate(zip(*spec)):
        for layer in range(layers * 2):
            strides = 1 if block == 0 or layer != 0 else 2
            y = residual_block_b18_v1(y, filters, strides=strides)

    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.GlobalAvgPool2D()(y)
    y = lq.layers.QuantDense(num_classes,
                             kernel_initializer="glorot_normal",
                             use_bias=False
                             )(y)
    y = tf.keras.layers.Activation("softmax", dtype="float32")(y)

    model = tf.keras.Model(
        inputs=x,
        outputs=y,
        name=f"Resnet_B18_v0",
    )
    if weight_path is not None:
        model.load_weights(weight_path)

    return model


def resnet_b18_v1(input_shape, num_classes, weight_path=None):
    ''' remove maxpool and batchnorm and bn after 1st qconv epsilon 1e-3'''
    x = tf.keras.Input(shape=input_shape)
    #
    y = tf.keras.layers.QuantConv2D(64,
                                    kernel_size=3,
                                    padding="same",
                                    kernel_initializer="he_normal",
                                    use_bias=False,
                                    )(x)

    y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(y)
    y = tf.keras.layers.Activation("relu")(y)

    for block, (layers, filters) in enumerate(zip(*spec)):
        for layer in range(layers * 2):
            strides = 1 if block == 0 or layer != 0 else 2
            y = residual_block_b18_v1(y, filters, strides=strides)

    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.GlobalAvgPool2D()(y)
    y = lq.layers.QuantDense(num_classes,
                             kernel_initializer="glorot_normal",
                             use_bias=True
                             )(y)
    y = tf.keras.layers.Activation("softmax", dtype="float32")(y)

    model = tf.keras.Model(
        inputs=x,
        outputs=y,
        name=f"Resnet_B18_v1",
    )
    if weight_path is not None:
        model.load_weights(weight_path)

    return model

def resnet_b18_v2(input_shape, num_classes, weight_path=None):
    ''' 2 shortcut'''
    x = tf.keras.Input(shape=input_shape)
    #
    y = lq.layers.QuantConv2D(64,
                              kernel_size=3,
                              padding="same",
                              kernel_initializer="he_normal",
                              use_bias=False,
                              )(x)

    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(y)

    for block, (layers, filters) in enumerate(zip(*spec)):
        for layer in range(layers):
            strides = 1 if block == 0 or layer != 0 else 2
            y = residual_block_b18_v0(y, filters, strides=strides)

    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.GlobalAvgPool2D()(y)
    y = lq.layers.QuantDense(
        num_classes,
        kernel_initializer="glorot_normal",
        use_bias=False
    )(y)
    y = tf.keras.layers.Activation("softmax", dtype="float32")(y)

    model = tf.keras.Model(
        inputs=x,
        outputs=y,
        name=f"Resnet_B18_v2",
    )
    if weight_path is not None:
        model.load_weights(weight_path)

    return model
