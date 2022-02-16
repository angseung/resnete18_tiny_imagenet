import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import larq as lq
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.image import resize
from models.resnet_e18f import resnet_e18, vgg_e18
import tensorflow_datasets as tfds
from utils import TinyImagenet

base_dir = "C:/tiny-imagenet-200"
target_dir = "./data"
test_name = 'resnet_e18'
num_classes = 200
model = resnet_e18(input_shape=(64, 64, 3), num_classes=num_classes)

# Data Load Cifar10
(train_images, train_labels), (test_images, test_labels) = TinyImagenet(base_dir, target_dir).load_data()

# train_images = train_images.reshape((50000, 32, 32, 3)).astype("float32")
# test_images = test_images.reshape((10000, 32, 32, 3)).astype("float32")

# Normalize pixel values to be between -1 and 1
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

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

lq.models.summary(model)
tb = TensorBoard(log_dir='results/{}/log'.format(test_name))

learning_rate = 1e-3
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
mcp_save = ModelCheckpoint('results/{}/cifar10.h5'.format(test_name), save_best_only=True, monitor='val_accuracy',
                           mode='max')

model.compile(
    tf.keras.optimizers.Adam(lr=learning_rate, decay=0.0001),
    # tf.keras.optimizers.RMSprop(lr=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

trained_model = model.fit(
    datagen.flow(train_images, train_labels, batch_size=64),
    epochs=120,
    validation_data=(test_images, test_labels),
    shuffle=True,
    callbacks=[tb, mcp_save, lrcb]
)

model.load_weights('results/{}/cifar10.h5'.format(test_name))

# Model Evaluate!
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy {test_acc * 100:.2f} %")
