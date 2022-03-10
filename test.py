import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
from models.resnet_18 import resnet_18
from models.resnet_b18f import resnet_b18_v2
from utils import BeansImageNet

input_size_list = [64]
bnn_opt = True
interpolation_list = [
    "bilinear",
    "nearest",
    "bicubic",
    "seam-carving"
]
test_set = {
    64 : "auto_beans_resnet_18_20220222-195006"
}
dat_comb = [
    "BC",
    "CB",
    "BN",
    "NB",
    "CN",
    "NC"
]

for input_size in input_size_list:
    resize_size = int(input_size * (256 / 224))
    test_name = test_set[input_size]


    if bnn_opt:
        model = resnet_b18_v2((input_size, input_size, 3), 3, None)
        model.load_weights("./results/%s/beansimagenet.h5" % test_name)
    else:
        model = resnet_18((input_size, input_size, 3), 3, None)
        dirlist = os.listdir("./result")

    model.compile(
        tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    (
        (train_data_bilinear, train_label),
        (test_data_bilinear, test_label),
        (val_data_bilinear, val_label),
    ) = BeansImageNet(
        input_size=input_size, resize_size=resize_size, norm=False, carving=False, interpolation="bilinear",
    ).load_data_as_numpy()

    (
        (train_data_bicubic, train_label),
        (test_data_bicubic, test_label),
        (val_data_bicubic, val_label),
    ) = BeansImageNet(
        input_size=input_size, resize_size=resize_size, norm=False, carving=False, interpolation="bicubic",
    ).load_data_as_numpy()
    (
        (train_data_nearest, train_label),
        (test_data_nearest, test_label),
        (val_data_nearest, val_label),
    ) = BeansImageNet(
        input_size=input_size, resize_size=resize_size, norm=False, carving=False, interpolation="nearest",
    ).load_data_as_numpy()

    datasets = {
        "B" : test_data_bilinear,
        "N" : test_data_nearest,
        "C" : test_data_bicubic
    }

    fig = plt.figure()
    max_acc = 0.7422
    max_comb = "Bilinear"

    for i, comb in enumerate(dat_comb):
        first = comb[0]
        last = comb[1]

        test_result = []

        for k in np.linspace(0.0, 3.0, num=100):
            k = k.item()
            test_data = (k+1) * datasets[first].astype(np.float32) - (k) * datasets[last].astype(np.float32)
            test_data_re = (test_data - test_data.min()) / (test_data.max() - test_data.min()) * 2 - 1
            test_loss, test_acc = model.evaluate(x=test_data_re, y=test_label)
            test_result.append(test_acc * 100)

        curr_max_acc = np.array(test_result).max().item()

        if curr_max_acc > max_acc:
            max_index = np.linspace(0.0, 3.0, num=100)[np.argmax(np.array(test_result)).item()]
            max_acc = curr_max_acc
            max_comb = comb

        plt.plot(np.linspace(0.0, 3.0, num=100), test_result, label=comb)

    plt.grid(True)
    plt.hlines(y=74.22, xmin=-10, xmax=10, colors="k", linestyle="dotted", label="Nearest")
    plt.legend(loc="best", ncol=2)
    plt.xlabel("k")
    plt.plot(max_index, max_acc, "k*")
    plt.xlim([-0.2, 3.2])
    plt.ylabel("Test Acc(%)")
    plt.title("Test with weighted-decoupled images")
    plt.show()
