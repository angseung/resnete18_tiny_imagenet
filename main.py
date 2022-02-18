import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import datetime
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import larq as lq
import larq_zoo
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.image import resize
from models.resnet_e18f import resnet_e18, vgg_e18
import tensorflow_datasets as tfds
from utils import TinyImageNet, replace_intermediate_layer_resnet18
from classification_models.keras import Classifiers

parser = argparse.ArgumentParser(description='resnet model')
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--input_size", type=int, default=64)
parser.add_argument("--method", type=str, default="bilinear")
parser.add_argument("--tune", type=bool, default=True)
args = parser.parse_args()
assert args.method in ["bilinear", "nearest", "bicubic"]
assert args.input_size in [16, 32, 64]

input_size = args.input_size
base_dir = "C:/tiny-imagenet-200"
target_dir = "./data"
test_name = "resnet_18_%s" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
num_classes = 200

ResNet18, preprocess_input = Classifiers.get('resnet18')
model = ResNet18(input_shape=(input_size, input_size, 3), weights='imagenet', include_top=False)

# input = tf.keras.Input(shape=(input_size, input_size, 3))
# y = model(input)
# y = tf.keras.layers.GlobalAvgPool2D()(y)
# y = tf.keras.layers.Dense(num_classes, kernel_initializer="glorot_normal")(y)
# y = tf.keras.layers.Activation("softmax", dtype="float32")(y)
# model = tf.keras.Model(input, y)


def get_output_of_layer(layer):
    if layer.name in layer_outputs:
        return layer_outputs[layer.name]

    if layer.name == starting_layer_name:
        out = layer(input)
        layer_outputs[layer.name] = out
        return out

    prev_layers = []
    for node in layer._inbound_nodes:
        try:
            prev_layers.extend(node.inbound_layers)
        except:
            prev_layers.append(node.inbound_layers)

    pl_outs = []
    for pl in prev_layers:
        pl_outs.extend([get_output_of_layer(pl)])

    out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
    layer_outputs[layer.name] = out
    return out


layer_outputs = {}
starting_layer_name = 'bn_data'
input = tf.keras.layers.Input(batch_shape=model.get_layer(starting_layer_name).get_input_shape_at(0))
output = get_output_of_layer(model.layers[-1])
output = tf.keras.layers.GlobalAvgPool2D()(output)
output = tf.keras.layers.Dense(num_classes, kernel_initializer="glorot_normal")(output)
output = tf.keras.layers.Activation("softmax", dtype="float32")(output)
model = tf.keras.Model(input, output)

if args.tune:
    h5_dir = "resnet_18_20220218-201443"
    model.load_weights("results/{}/tinyimagenet.h5".format(h5_dir))

# Data Load TinyImageNet
(train_images, train_labels), (test_images, test_labels) = TinyImageNet(
    base_dir, target_dir
).load_data()

if args.input_size != 64:
    train_images = tf.image.resize(tf.convert_to_tensor(train_images), size=[args.input_size, args.input_size], method=args.method)
    test_images = tf.image.resize(tf.convert_to_tensor(test_images), size=[args.input_size, args.input_size], method=args.method)

# train_images = preprocess_input(train_images)
# test_images = preprocess_input(test_images)
# train_images = train_images.reshape((50000, 32, 32, 3)).astype("float32")
# test_images = test_images.reshape((10000, 32, 32, 3)).astype("float32")

# Normalize pixel values to be between -1 and 1
# train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

# train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
# test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

# Data augmentation
datagen = ImageDataGenerator(
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # randomly flip images
    horizontal_flip=True,
)

datagen.fit(train_images)
log_dir = "log_%s"

lq.models.summary(model)
tb = tf.keras.callbacks.TensorBoard(
    log_dir="results/{}/log".format(test_name), histogram_freq=1
)

learning_rate = args.lr
learning_factor = 0.3
learning_steps = [40, 80, 100]


def learning_rate_schedule(epoch):
    lr = learning_rate
    for step in learning_steps:
        if epoch < step:
            return lr
        lr *= learning_factor
    return lr


lrcb = tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)
top5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")
mcp_save = ModelCheckpoint(
    "results/{}/tinyimagenet.h5".format(test_name),
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
)

model.compile(
    # tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=0.0001),
    tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy", top5_acc],
)

trained_model = model.fit(
    datagen.flow(train_images, train_labels, batch_size=64),
    epochs=100,
    validation_data=(test_images, test_labels),
    shuffle=True,
    callbacks=[tb, mcp_save, lrcb],
)

model.load_weights("results/{}/tinyimagenet.h5".format(test_name))

# Model Evaluate!
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy {test_acc * 100:.2f} %")
