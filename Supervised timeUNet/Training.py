import faulthandler

faulthandler.enable()

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
import os

from models import build_model
from csbdeep.utils.tf import limit_gpu_memory

from datawrapper import make_sequence

limit_gpu_memory(fraction=0.75, allow_growth=False)
# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

##### architecture ########
architecture = ["time CNN"][0]

image_size = [256, 256]
batch_size = 8
timepoints = 11
timeinterval = [
    1
]  # can be used as an data augmentation to generate training dataset with various movements
signallevel = [s for s in range(10, 100, 10)]
gaussnoise = [s for s in np.linspace(0.5, 5, 10)]
offset = 100

# possibly simulate and then load paths to simulation data
path_train = (
    os.path.dirname(os.path.realpath(__file__)) + "/Experimental_data/training/"
)
validation_ratio = 0.1
train_sequence, val_sequence = make_sequence(
    path_train,
    validation_ratio,
    batch_size,
    timepoints,
    timeinterval,
    signallevel,
    gaussnoise,
    offset,
)

# model configuration
use_bias = True
optimizer = ks.optimizers.Adam(lr=1e-3)

# build model
model = build_model(image_size, timepoints, architecture, use_bias=use_bias)
# for layer in model.layers:
#    print(layer.name, layer.output_shape)

# compile model
model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])

# train
model.fit_generator(
    generator=train_sequence,
    validation_data=val_sequence,
    epochs=50,
    max_queue_size=3,
    verbose=1,
    callbacks=[
        ks.callbacks.EarlyStopping(mode="min", patience=10, verbose=1),
        ks.callbacks.ModelCheckpoint(
            filepath="./models/best_model.h5", save_best_only=True
        ),
        ks.callbacks.TensorBoard(log_dir="./logs"),
        ks.callbacks.TerminateOnNaN(),
    ],
)
