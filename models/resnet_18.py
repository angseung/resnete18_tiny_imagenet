import tensorflow as tf


def basicblock18(x: tf.Tensor, filters: int, strides: int = 1) -> tf.Tensor:
    downsample = x.get_shape().as_list()[-1] != filters

    if downsample:
        residual = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            use_bias=False,
            strides=strides,
            kernel_initializer="glorot_normal",
        )(x)
        residual = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(
            residual
        )
    else:
        residual = x

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        kernel_initializer="glorot_normal",
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="glorot_normal",
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)

    return tf.keras.layers.add([x, residual])


spec = ([2, 2, 2, 2], [64, 128, 256, 512])


def resnet_18(input_shape, num_classes, weight_path=None):
    x = tf.keras.Input(shape=input_shape)

    y = tf.keras.layers.Conv2D(
        64,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(x)

    y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(y)
    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(y)
    # y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(y)

    for block, (layers, filters) in enumerate(zip(*spec)):
        for layer in range(layers * 1):
            strides = 1 if block == 0 or layer != 0 else 2
            y = basicblock18(y, filters, strides=strides)

    y = tf.keras.layers.GlobalAvgPool2D()(y)
    y = tf.keras.layers.Dense(num_classes, kernel_initializer="glorot_normal")(y)
    y = tf.keras.layers.Activation("softmax", dtype="float32")(y)

    model = tf.keras.Model(
        inputs=x,
        outputs=y,
        name=f"Resnet_18",
    )
    if weight_path is not None:
        model.load_weights(weight_path)

    return model


if __name__ == "__main__":

    model = resnet_18((224, 224, 3), 3)
    import numpy as np

    a = tf.convert_to_tensor(np.random.randn(1, 224, 224, 3))
    k = model(a)
    model.summary()
