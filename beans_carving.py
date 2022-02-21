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

traindata, testdata = BeansImageNet().load_data()
model = resnet_e18((224, 224, 3), 3, None)

