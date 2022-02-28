import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import datetime
import argparse
import numpy as np
import tensorflow as tf
import larq as lq
from tensorflow.python.keras.callbacks import ModelCheckpoint
from models.resnet_18 import resnet_18
from models.resnet_b18f import resnet_b18_v2
from utils import BeansImageNet

test_name = "beans_resnet_18_%s" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser(description="resnet model")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument("--carving", type=bool, default=False)
parser.add_argument("--bnn", type=bool, default=False)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--interpolation", type=str, default="bilinear") # [bilinear, nearest, bicubic]
parser.add_argument("--seed", type=bool, default=True)

args = parser.parse_args()
print(args)

if args.seed:
    seed = 1
    tf.random.set_seed(seed)
    np.random.seed(seed)

input_size = args.input_size
resize_size = int(input_size * (256 / 224))

(
    (train_data, train_label),
    (test_data, test_label),
    (val_data, val_label),
) = BeansImageNet(
    input_size=input_size, resize_size=resize_size, norm=True, carving=args.carving, interpolation=args.interpolation,
).load_data_as_numpy()


if args.bnn:
    model = resnet_b18_v2((input_size, input_size, 3), 3, None)
else:
    model = resnet_18((input_size, input_size, 3), 3, None)

lq.models.summary(model)
os.mkdir("results/{}".format(test_name))
with open("results/{}/config.txt".format(test_name), "w") as f:
    lq.models.summary(model, print_fn=f.write)
    f.write("\n%s\n" % str(args))

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
mcp_save = ModelCheckpoint(
    "results/{}/beansimagenet.h5".format(test_name),
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
)

model.compile(
    tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

trained_model = model.fit(
    x=train_data,
    y=train_label,
    epochs=args.epoch,
    validation_data=(val_data, val_label),
    shuffle=True,
    batch_size=args.batch_size,
    callbacks=[tb, mcp_save, lrcb],
)

model.load_weights("results/{}/beansimagenet.h5".format(test_name))

# Model Evaluate!
test_loss, test_acc = model.evaluate(x=test_data, y=test_label)
print(f"Test accuracy {test_acc * 100:.2f} %")
with open("results/{}/config.txt".format(test_name), "a") as f:
    f.write(f"Test accuracy {test_acc * 100:.2f} %")
