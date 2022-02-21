import os
import time
from PIL import Image
from typing import Tuple
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm
from carve import resize


class TinyImageNet:
    def __init__(
        self,
        base_dir: str = "C:/tiny-imagenet-200",
        target_dir: str = "./data",
    ) -> None:
        self.base_dir = base_dir
        self.target_dir = target_dir
        self.is_npz_made = (
            (os.path.isfile(target_dir + "/train_images.npy"))
            and (os.path.isfile(target_dir + "/train_labels.npy"))
            and (os.path.isfile(target_dir + "/test_images.npy"))
            and (os.path.isfile(target_dir + "/test_labels.npy"))
        )
        self.num_classes = 200

    def load_data(
        self,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        if not self.is_npz_made:
            print("File not exists, generating TinyImage dataset...")
            (train_images, train_labels), (test_images, test_labels) = get_data(
                self.base_dir, get_id_dictionary(self.base_dir)
            )

            np.save(self.target_dir + "/train_images.npy", train_images)
            np.save(self.target_dir + "/train_labels.npy", train_labels)
            np.save(self.target_dir + "/test_images.npy", test_images)
            np.save(self.target_dir + "/test_labels.npy", test_labels)

        else:
            print("Load from saved npy data...")
            train_images = np.load(self.target_dir + "/train_images.npy")
            train_labels = np.load(self.target_dir + "/train_labels.npy")
            test_images = np.load(self.target_dir + "/test_images.npy")
            test_labels = np.load(self.target_dir + "/test_labels.npy")

        return (train_images, train_labels), (test_images, test_labels)


class BeansImageNet:
    def __init__(
        self,
        base_dir: str = "C:/beans",
        target_dir: str = "./data_beans",
        input_size: int = 224,
        resize_size: int = 256,
        norm: bool = True,
        carving: bool = False,
    ) -> None:
        self.base_dir = base_dir
        self.target_dir = target_dir
        self.num_classes = 3
        self.input_size = input_size
        self.resize_size = resize_size
        self.norm = norm
        self.carving = carving
        self.is_npz_made = (
            (os.path.isfile(target_dir + "/train_images_sc.npy"))
            and (os.path.isfile(target_dir + "/train_labels_sc.npy"))
            and (os.path.isfile(target_dir + "/test_images_sc.npy"))
            and (os.path.isfile(target_dir + "/test_labels_sc.npy"))
            and (os.path.isfile(target_dir + "/val_images_sc.npy"))
            and (os.path.isfile(target_dir + "/val_labels_sc.npy"))
        )

    # return dataset as tf.data.Dataset class
    def load_data(
        self,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        if self.carving:
            print("apply carving...")
            traindata = tf.keras.utils.image_dataset_from_directory(
                self.base_dir + "/train",
                image_size=(self.resize_size, self.resize_size),
                interpolation="bilinear",
                shuffle=True,
                label_mode="categorical",
            )
            testdata = tf.keras.utils.image_dataset_from_directory(
                self.base_dir + "/test",
                image_size=(self.resize_size, self.resize_size),
                interpolation="bilinear",
                shuffle=False,
                label_mode="categorical",
            )
            valdata = tf.keras.utils.image_dataset_from_directory(
                self.base_dir + "/validation",
                image_size=(self.resize_size, self.resize_size),
                interpolation="bilinear",
                shuffle=False,
                label_mode="categorical",
            )
        else:
            traindata = tf.keras.utils.image_dataset_from_directory(
                self.base_dir + "/train",
                image_size=(self.input_size, self.input_size),
                interpolation="bilinear",
                shuffle=True,
                label_mode="categorical",
            )
            testdata = tf.keras.utils.image_dataset_from_directory(
                self.base_dir + "/test",
                image_size=(self.input_size, self.input_size),
                interpolation="bilinear",
                shuffle=False,
                label_mode="categorical",
            )
            valdata = tf.keras.utils.image_dataset_from_directory(
                self.base_dir + "/validation",
                image_size=(self.input_size, self.input_size),
                interpolation="bilinear",
                shuffle=False,
                label_mode="categorical",
            )

        return (traindata, testdata, valdata)

    def load_data_as_numpy(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        if self.carving and self.is_npz_made:
            # if carving applied images are saved already...
            train_data = np.load(self.target_dir + "/train_images_sc.npy")
            train_label = np.load(self.target_dir + "/train_labels_sc.npy")
            test_data = np.load(self.target_dir + "/test_images_sc.npy")
            test_label = np.load(self.target_dir + "/test_labels_sc.npy")
            val_data = np.load(self.target_dir + "/val_images_sc.npy")
            val_label = np.load(self.target_dir + "/val_labels_sc.npy")

        else:
            traindata, testdata, valdata = self.load_data()

            for i, (images, targets) in enumerate(tqdm(traindata.as_numpy_iterator())):
                if self.carving:
                    for j, image in enumerate(images):
                        if j == 0:
                            sc_image = resize(image, (self.input_size, self.input_size))[np.newaxis, :]
                        else:
                            sc_image = np.concatenate(
                                (
                                    sc_image,
                                    resize(image, (self.input_size, self.input_size))[np.newaxis, :],
                                )
                            )
                    images = sc_image.copy()

                if i == 0:
                    train_data = images
                    train_label = targets
                else:
                    train_data = np.concatenate((train_data, images), axis=0)
                    train_label = np.concatenate((train_label, targets), axis=0)

            for i, (images, targets) in enumerate(tqdm(testdata.as_numpy_iterator())):
                if self.carving:
                    for j, image in enumerate(images):
                        if j == 0:
                            sc_image = resize(image, (self.input_size, self.input_size))[np.newaxis, :]
                        else:
                            sc_image = np.concatenate(
                                (
                                    sc_image,
                                    resize(image, (self.input_size, self.input_size))[np.newaxis, :],
                                )
                            )
                    images = sc_image.copy()
                if i == 0:
                    test_data = images
                    test_label = targets
                else:
                    test_data = np.concatenate((test_data, images), axis=0)
                    test_label = np.concatenate((test_label, targets), axis=0)

            for i, (images, targets) in enumerate(tqdm(valdata.as_numpy_iterator())):
                if self.carving:
                    for j, image in enumerate(images):
                        if j == 0:
                            sc_image = resize(image, (self.input_size, self.input_size))[np.newaxis, :]
                        else:
                            sc_image = np.concatenate(
                                (
                                    sc_image,
                                    resize(image, (self.input_size, self.input_size))[np.newaxis, :],
                                )
                            )
                    images = sc_image.copy()
                if i == 0:
                    val_data = images
                    val_label = targets
                else:
                    val_data = np.concatenate((val_data, images), axis=0)
                    val_label = np.concatenate((val_label, targets), axis=0)

            if self.norm:
                train_data, test_data, val_data = (
                    train_data / 127.5 - 1,
                    test_data / 127.5 - 1,
                    val_data / 127.5 - 1,
                )

            if not os.path.isdir(self.target_dir):
                os.mkdir(self.target_dir)

            if self.carving:
                np.save(self.target_dir + "/train_images_sc.npy", train_data)
                np.save(self.target_dir + "/train_labels_sc.npy", train_label)
                np.save(self.target_dir + "/test_images_sc.npy", test_data)
                np.save(self.target_dir + "/test_labels_sc.npy", test_label)
                np.save(self.target_dir + "/val_images_sc.npy", val_data)
                np.save(self.target_dir + "/val_labels_sc.npy", val_label)

        return (train_data, train_label), (test_data, test_label), (val_data, val_label)


def get_id_dictionary(path):
    id_dict = {}
    for i, line in enumerate(open(path + "/wnids.txt", "r")):
        id_dict[line.replace("\n", "")] = i
    return path, id_dict


def get_class_to_id_dict():
    path, id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open(path + "/words.txt", "r")):
        n_id, word = line.split("\t")[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])
    return result


def get_data(path, id_dict):
    id_dict = id_dict[1]
    print("starting loading data")
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    for key, value in id_dict.items():
        train_data += [
            np.array(
                Image.open(
                    path + "/train/{}/images/{}_{}.JPEG".format(key, key, str(i))
                ).convert("RGB")
            )
            for i in range(500)
        ]
        train_labels_ = np.array([[0] * 200] * 500)
        train_labels_[:, value] = 1
        train_labels += train_labels_.tolist()

    for line in open(path + "/val/val_annotations.txt"):
        img_name, class_id = line.split("\t")[:2]
        test_data.append(
            np.array(
                Image.open(path + "/val/images/{}".format(img_name)).convert("RGB")
            )
        )
        test_labels_ = np.array([[0] * 200])
        test_labels_[0, id_dict[class_id]] = 1
        test_labels += test_labels_.tolist()

    print("finished loading data, in {} seconds".format(time.time() - t))

    # return np.concatenate(train_data, axis=0), np.concatenate(train_labels, axis=0), np.concatenate(test_data, axis=0), np.concatenate(test_labels, axis=0)
    return (np.array(train_data), np.array(train_labels)), (
        np.array(test_data),
        np.array(test_labels),
    )


def replace_intermediate_layer(model, layer_id, new_layer):
    from keras.models import Model

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model


def replace_intermediate_layer_resnet18(model, layer_id, new_layer):
    from keras.models import Model

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(len(layers)):
        if i == 0:
            continue
        elif i == layer_id:
            x = new_layer(x)
        elif i in [9, 28, 47, 66]:
            x_conv = layers[i](x)
            x = layers[i](x)
        elif i in [15]:
            x_0 = layers[i](x)
            x = layers[i](x)
        elif i in [16]:
            x_1 = layers[i](x_conv)
            x = layers[i](x_conv)
        elif i in [17]:
            x = layers[i]([x_0, x_1])
            x_0 = layers[i]([x_0, x_1])
        elif i == 25:
            x_1 = layers[i](x)
            x = layers[i](x)
        elif i == 26:
            x = layers[i]([x_0, x_1])
            x_0 = layers[i]([x_0, x_1])
        elif i == 34:
            x_0 = layers[i](x)
            x = layers[i](x)
        elif i == 35:
            x_1 = layers[i](x_conv)
            x = layers[i](x_conv)
        elif i == 36:
            x = layers[i]([x_0, x_1])
            x_0 = layers[i]([x_0, x_1])
        elif i == 44:
            x_1 = layers[i](x)
            x = layers[i](x)
        elif i == 45:
            x = layers[i]([x_0, x_1])
            x_0 = layers[i]([x_0, x_1])
        elif i == 53:
            x_0 = layers[i](x)
            x = layers[i](x)
        elif i == 54:
            x_1 = layers[i](x_conv)
            x = layers[i](x_conv)
        elif i == 55:
            x = layers[i]([x_0, x_1])
            x_0 = layers[i]([x_0, x_1])
        elif i == 63:
            x_1 = layers[i](x)
            x = layers[i](x)
        elif i == 64:
            x = layers[i]([x_0, x_1])
            x_0 = layers[i]([x_0, x_1])
        elif i == 72:
            x_0 = layers[i](x)
            x = layers[i](x)
        elif i == 73:
            x_1 = layers[i](x_conv)
            x = layers[i](x_conv)
        elif i == 74:
            x = layers[i]([x_0, x_1])
            x_0 = layers[i]([x_0, x_1])
        elif i == 82:
            x_1 = layers[i](x)
            x = layers[i](x)
        elif i == 83:
            x = layers[i]([x_0, x_1])
            x_0 = layers[i]([x_0, x_1])
        else:
            x = layers[i](x)

    new_model = Model(inputs=layers[0].input, outputs=x)
    return new_model


if __name__ == "__main__":

    train_data, train_labels, test_data, test_labels = get_data(get_id_dictionary())

    print("train data shape: ", train_data.shape)
    print("train label shape: ", train_labels.shape)
    print("test data shape: ", test_data.shape)
    print("test_labels.shape: ", test_labels.shape)
