from tensorflow.keras.applications.resnet import ResNet50
import larq as lq
import datetime
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.layers as L
from classification_models.keras import Classifiers

# input_size = args.input_size
base_dir = "C:/tiny-imagenet-200"
target_dir = "./data"
test_name = "resnet_18_%s" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
num_classes = 200

ResNet18, preprocess_input = Classifiers.get("resnet18")
resnet = ResNet18(input_shape=(64, 64, 3), weights="imagenet", include_top=False)

# resnet = ResNet50()
lq.models.summary(resnet)

starting_layer_name = "data"
new_input = tf.keras.layers.Input(
    batch_shape=resnet.get_layer(starting_layer_name).get_input_shape_at(0)
)

layer_outputs = {}


def get_output_of_layer(layer):
    if layer.name in layer_outputs:
        return layer_outputs[layer.name]

    if layer.name == starting_layer_name:
        out = layer(new_input)
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


new_output = get_output_of_layer(resnet.layers[-1])
model = Model(new_input, new_output)
