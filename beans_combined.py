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
from keras.callbacks import CSVLogger

test_name = "beans_resnet_18_%s" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser(description="resnet model")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument("--carving", type=bool, default=False)
parser.add_argument("--bnn", type=bool, default=False)
parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--interpolation", type=str, default="combined") # [bilinear, nearest, bicubic]
parser.add_argument("--seed", type=bool, default=True)

args = parser.parse_args()
print(args)

if args.seed:
    seed = 1
    tf.random.set_seed(seed)
    np.random.seed(seed)

input_size_list = [224, 160, 128, 64]
# input_size_list = [160]
interpolation_list = [
    "combined",
    # "bilinear,"
    # "nearest",
    # "bicubic",
    # "seam-carving"
]
bin_opt = [
    True,
    False,
           ]

for is_bnn in bin_opt:
    for input_size in input_size_list:
        for interpolation in interpolation_list:
            test_name = "auto_beans_resnet_18_%s_%d_%s" % (is_bnn, input_size, interpolation)

            if interpolation == "seam-carving":
                args.carving = True
                args.interpolation = "bilinear"

            elif interpolation in [
                "bilinear",
                "nearest",
                "bicubic",
                "combined",
            ]:
                args.carving = False
                args.interpolation = interpolation

            args.input_size = input_size
            args.bnn = is_bnn

            input_size = args.input_size
            resize_size = int(input_size * (256 / 224))

            (
                (train_data_nearest, train_label),
                (test_data_nearest, test_label),
                (val_data_nearest, val_label),
            ) = BeansImageNet(
                input_size=input_size, resize_size=resize_size, norm=False, carving=args.carving, interpolation="nearest",
            ).load_data_as_numpy()

            (
                (train_data_bilinear, train_label_bilinear),
                (test_data_bilinear, test_label_bilinear),
                (val_data_bilinear, val_label_bilinear),
            ) = BeansImageNet(
                input_size=input_size, resize_size=resize_size, norm=False, carving=args.carving, interpolation="bilinear",
            ).load_data_as_numpy()

            assert np.array_equal(train_label, train_label_bilinear)
            assert np.array_equal(test_label, test_label_bilinear)
            assert np.array_equal(val_label, val_label_bilinear)

            train_data = (train_data_nearest.astype(np.uint16) + train_data_bilinear.astype(np.uint16)) / 2.0
            test_data = (test_data_nearest.astype(np.uint16) + test_data_bilinear.astype(np.uint16)) / 2.0
            val_data = (val_data_nearest.astype(np.uint16) + val_data_bilinear.astype(np.uint16)) / 2.0

            train_data, test_data, val_data = (
                train_data / 127.5 - 1,
                test_data / 127.5 - 1,
                val_data / 127.5 - 1,
            )

            if args.bnn:
                model = resnet_b18_v2((input_size, input_size, 3), 3, None)
            else:
                model = resnet_18((input_size, input_size, 3), 3, None)

            lq.models.summary(model)
            os.mkdir("./results_comb/{}".format(test_name))
            with open("results_comb/{}/config.txt".format(test_name), "w") as f:
                lq.models.summary(model, print_fn=f.write)
                f.write("\n%s\n" % str(args))

            tb = tf.keras.callbacks.TensorBoard(
                log_dir="results_comb/{}/log".format(test_name), histogram_freq=1
            )
            csv_logger = CSVLogger('results_comb/{}/log.csv'.format(test_name), append=True, separator=',')

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
                "results_comb/{}/beansimagenet.h5".format(test_name),
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
                callbacks=[tb, mcp_save, lrcb, csv_logger],
            )

            model.load_weights("results_comb/{}/beansimagenet.h5".format(test_name))

            # Model Evaluate!
            test_loss, test_acc = model.evaluate(x=test_data, y=test_label)
            print(f"Test accuracy {test_acc * 100:.2f} %")
            with open("results_comb/{}/config.txt".format(test_name), "a") as f:
                f.write(f"Test accuracy {test_acc * 100:.2f} %")
