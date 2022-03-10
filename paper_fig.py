import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from carve import resize

img_file = "C:/beans/train/bean_rust/bean_rust_train.10.jpg"
img = Image.open(img_file).convert("RGB")
B = img.resize((64, 64), Image.BILINEAR)
N = img.resize((64, 64), Image.NEAREST)
k = 0.272727

WD = (1 + k) * np.array(B).astype(np.float32) - k * np.array(N)
WD = (WD - WD.min()) / (WD.max() - WD.min())

fig = plt.figure()
plt.subplot(131)
plt.axis("off")
plt.title("BILINEAR")
plt.imshow(B)

plt.subplot(132)
plt.title("NEAREST")
plt.axis("off")
plt.imshow(N)

plt.subplot(133)
plt.title("Proposed")
plt.axis("off")
plt.imshow(WD)

plt.show()
plt.tight_layout()

import tensorflow as tf
sc_64 = tf.keras.layers.CenterCrop(height=64, width=64)(resize(np.array(img), (73, 73)))
sc_128 = tf.keras.layers.CenterCrop(height=128, width=128)(resize(np.array(img), (146, 146)))
sc_160 = tf.keras.layers.CenterCrop(height=160, width=160)(resize(np.array(img), (182, 182)))
sc_224 = tf.keras.layers.CenterCrop(height=224, width=224)(resize(np.array(img), (256, 256)))

fig2 = plt.figure()
plt.subplot(141)
plt.axis("off")
plt.title("SC, size=64")
plt.imshow(sc_64.numpy().astype(np.uint8))

plt.subplot(142)
plt.axis("off")
plt.title("SC, size=128")
plt.imshow(sc_128.numpy().astype(np.uint8))

plt.subplot(143)
plt.axis("off")
plt.title("SC, size=160")
plt.imshow(sc_160.numpy().astype(np.uint8))

plt.subplot(144)
plt.axis("off")
plt.title("SC, size=224")
plt.imshow(sc_224.numpy().astype(np.uint8))

plt.tight_layout()
plt.show()
fig2.savefig("sc.png", dpi=200, transparent=True)

