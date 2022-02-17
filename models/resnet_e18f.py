import tensorflow as tf
import larq as lq

# set Network common constraint
input_quantizer = lq.quantizers.SteSign(clip_value=1.25)
kernel_quantizer = lq.quantizers.SteSign(clip_value=1.25)
kernel_constraint = lq.constraints.WeightClip(clip_value=1.25)


def vgg_block_e18(x: tf.Tensor, filters: int, strides: int = 1) -> tf.Tensor:
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

    return x


def residual_block_e18(x: tf.Tensor, filters: int, strides: int = 1) -> tf.Tensor:
    downsample = x.get_shape().as_list()[-1] != filters

    if downsample:
        residual = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)(x)
        residual = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            use_bias=False,
            kernel_initializer="glorot_normal",
        )(residual)
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


def vgg_e18(input_shape, num_classes, weight_path=None):
    x = tf.keras.Input(shape=input_shape)

    y = tf.keras.layers.Conv2D(
        64,
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
            y = vgg_block_e18(y, filters, strides=strides)

    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.GlobalAvgPool2D()(y)
    y = tf.keras.layers.Dense(num_classes, kernel_initializer="glorot_normal")(y)
    y = tf.keras.layers.Activation("softmax", dtype="float32")(y)

    model = tf.keras.Model(
        inputs=x,
        outputs=y,
        name=f"VGG_E18",
    )
    if weight_path is not None:
        model.load_weights(weight_path)

    return model


def resnet_e18(input_shape, num_classes, weight_path=None):
    x = tf.keras.Input(shape=input_shape)
    #
    y = tf.keras.layers.Conv2D(
        64,
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
            y = residual_block_e18(y, filters, strides=strides)

    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.GlobalAvgPool2D()(y)
    y = tf.keras.layers.Dense(num_classes, kernel_initializer="glorot_normal")(y)
    y = tf.keras.layers.Activation("softmax", dtype="float32")(y)

    model = tf.keras.Model(
        inputs=x,
        outputs=y,
        name=f"Resnet_E18",
    )
    if weight_path is not None:
        model.load_weights(weight_path)

    return model
