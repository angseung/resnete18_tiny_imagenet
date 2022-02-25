import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import datetime
import time
import argparse
import numpy as np
import tensorflow as tf
import larq as lq
from tensorflow.python.keras.callbacks import ModelCheckpoint
from models.resnet_e18f import resnet_e18
from models.resnet_b18f import resnet_b18_v2


input_size_list = [224, 160, 128, 64]
# input_size_list = [160]
bin_opt = [
    True,
    False,
           ]
results = []

for is_bnn in bin_opt:
    for input_size in input_size_list:

        if is_bnn:
            model = resnet_b18_v2((input_size, input_size, 3), 3, None)
        else:
            model = resnet_e18((input_size, input_size, 3), 3, None)

        times = np.zeros((1000, ), dtype=np.float32)

        for i in range(1000):
            input_data = np.random.randn(1, input_size, input_size, 3).astype(np.float32)

            start = time.time()
            model(input_data)
            end = time.time()

            elapsed_time = end - start
            times[i] = elapsed_time

        elapsed_time = times.sum() / 1000
        results.append((is_bnn, input_size, elapsed_time))

