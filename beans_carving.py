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
import tensorflow_datasets as tfds
from utils import TinyImageNet, replace_intermediate_layer_resnet18
from classification_models.keras import Classifiers
from models.resnet_e18f import resnet_e18
from utils import BeansImageNet

test_name = "beans_resnet_18_%s" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CARVING = True

if CARVING:
    input_size = 256
else:
    input_size = 224

(
    (train_data, train_label),
    (test_data, test_label),
    (val_data, val_label),
) = BeansImageNet(input_size=input_size, norm=True, carving=CARVING).load_data_as_numpy()
model = resnet_e18((224, 224, 3), 3, None)
model.summary()

tb = tf.keras.callbacks.TensorBoard(
    log_dir="results/{}/log".format(test_name), histogram_freq=1
)

learning_rate = 0.001
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
    "results/{}/beansimagenet.h5".format(test_name),
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
)

model.compile(
    # tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=0.0001),
    tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

trained_model = model.fit(
    x=train_data,
    y=train_label,
    epochs=50,
    validation_data=(val_data, val_label),
    shuffle=True,
    batch_size=64,
    callbacks=[tb, mcp_save, lrcb],
)

model.load_weights("results/{}/beansimagenet.h5".format(test_name))

# Model Evaluate!
test_loss, test_acc = model.evaluate(x=test_data, y=test_label)
print(f"Test accuracy {test_acc * 100:.2f} %")
