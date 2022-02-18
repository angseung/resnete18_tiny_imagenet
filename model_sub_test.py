import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import models
from tensorflow.keras import layers
from classification_models.keras import Classifiers

# resnet = ResNet50()
ResNet18, preprocess_input = Classifiers.get('resnet18')
resnet = ResNet18(input_shape=(64, 64, 3), weights='imagenet', include_top=False)

# this is the split point, i.e. the starting layer in our sub-model
starting_layer_name = 'bn0'

# create a new input layer for our sub-model we want to construct
new_input = layers.Input(batch_shape=resnet.get_layer(starting_layer_name).get_input_shape_at(0))

layer_outputs = {}
def get_output_of_layer(layer):
    # if we have already applied this layer on its input(s) tensors,
    # just return its already computed output
    if layer.name in layer_outputs:
        return layer_outputs[layer.name]

    # if this is the starting layer, then apply it on the input tensor
    if layer.name == starting_layer_name:
        out = layer(new_input)
        layer_outputs[layer.name] = out
        return out

    # find all the connected layers which this layer
    # consumes their output
    prev_layers = []
    for node in layer._inbound_nodes:
        prev_layers.extend(node.inbound_layers)

    # get the output of connected layers
    pl_outs = []
    for pl in prev_layers:
        pl_outs.extend([get_output_of_layer(pl)])

    # apply this layer on the collected outputs
    out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
    layer_outputs[layer.name] = out
    return out

# note that we start from the last layer of our desired sub-model.
# this layer could be any layer of the original model as long as it is
# reachable from the starting layer
new_output = get_output_of_layer(resnet.layers[-1])

# create the sub-model
model = models.Model(new_input, new_output)