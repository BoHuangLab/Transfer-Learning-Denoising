"""
Clean experimental data and supply generator functions to neural networks
"""

from tensorflow import keras as ks
from random import shuffle
import tifffile as tiff
import os
import numpy as np


def extract_ROIs(
    movie,
    train_size,
    intensity_threshold,
    timesteps,
    timeinterval,
    overlapratio=0.5,
    normalization=True,
    pad=None,
):

    """ "

    Args:
        movie:    input .tif images used to generate movie
        train_size:     experimental images are all acquired by sCMOS camera (2x2 bin); cut into small ROIs
        intensity_threshold: a minimal average intensity requirement for accepting the ROI in the training dataset, make sure ROI is not mostly background
        timesteps:      length of timelapse movie
        timeinterval:   time interval in frame number, augmentation to include the effect of movement with different scale
        overlapratio:   the overlap ratio between ROIs, default 50%
        normalization:  normalize each ROI image to the range of [0,1]
        pad:            pad mode for the starting and ending frames ['same','zero', None]

    Returns:
        ROIS:
        signals:

    Raises:
    """

    movie_shape = movie.shape  # tuple of movie shape (frame number, x, y)
    if train_size[0] > movie_shape[1] or train_size[1] > movie_shape[2]:
        raise ValueError("Desired training data size larger than input movie x/y size!")
    if timesteps > movie_shape[0]:
        raise ValueError("Input movie shorter than required timelapse length!")

    ROIs = []
    signals = []
    step_size = [int(s * (1 - overlapratio)) for s in train_size]
    x_bl = train_size[0]  # track relative bottom left coordinates of ROI
    while x_bl < movie_shape[1]:
        y_bl = train_size[1]
        while y_bl < movie_shape[2]:
            ROI = movie[:, (x_bl - train_size[0]) : x_bl, (y_bl - train_size[1]) : y_bl]
            if np.mean(ROI[0, :, :]) > intensity_threshold:
                signal = np.max(ROI) - np.min(ROI)
                ROI = normalize(ROI, clip=True)
                # for interval in timeinterval:
                #     f0 = 0
                #     while (f0 + interval * (timesteps - 1) + 1) < movie_shape[0]:
                #         ROIs.append(ROI[f0:(f0 + interval * (timesteps - 1) + 1):interval, :, :])
                #         f0 = f0 + 1
                signals.append(signal)
                ROIs.append(
                    ROI
                )  # storing temporally augmented timelapse took a lot of space, do that in generating keras sequence
            y_bl = y_bl + step_size[1]
        x_bl = x_bl + step_size[0]
    return ROIs, signals


def experimental_images(
    file_path,
    train_size=[256, 256],
    intensity_threshold=1000,
    timesteps=11,
    timeinterval=[1, 3],
    overlapratio=0.5,
    normalization=True,
    offset=100,
    pad=None,
):
    """Imports images and exports cleaned and normalized images

    Retrieves tif files and passes them through extract_ROIS() function to crop, normalize, threshold, pad, and export as movie.

    Args:
        file_path:      the directory where experimental images stored
        train_size:     experimental images are all acquired by sCMOS camera (2x2 bin); cut into small ROIs
        intensity_threshold: a minimal average intensity requirement for accepting the ROI in the training dataset, make sure ROI is not mostly background
        timesteps:      length of timelapse movie
        timeinterval:   time interval in frame number, augmentation to include the effect of movement with different scale
        overlapratio:   the overlap ratio between ROIs, default 50%
        normalization:  normalize each ROI image to the range of [0,1]
        pad:            pad mode for the starting and ending frames ['same','zero', None]

    Returns:
        None

    Raises:
        FileNotFoundError: Report error if file path not found
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError

    allROIs = []
    allsignals = []
    index = 0
    for filename in os.listdir(file_path):
        if os.path.splitext(filename)[1] == ".tif":  # read tif files
            movie = tiff.imread(file_path + filename)
            ROIs, signals = extract_ROIs(
                movie - offset,
                train_size,
                intensity_threshold,
                timesteps,
                timeinterval,
                overlapratio,
                normalization,
            )
            allROIs = allROIs + ROIs
            allsignals = allsignals + signals
            if False:
                for i in range(len(ROIs)):
                    tiff.imwrite(
                        file_path + filename + "ROI_{}.tif".format(index),
                        ROIs[i],
                        dtype="float32",
                    )
                    index = index + 1
    np.savez(file_path + "highSNR", highSNR=allROIs, signal=allsignals)


class Imagesequence_generator(ks.utils.Sequence):
    """
    subclass of keras.utils.Sequence for generating training data sequence
    """

    def __init__(
        self, data, batch_size, timestep, timeinterval, signallevel, gaussnoise, offset
    ):
        self.data = data  # loaded npz file
        self.batch_size = batch_size
        self.timestep = timestep
        self.timeinterval = timeinterval
        self.signallevel = signallevel  # list
        self.gaussnoise = gaussnoise
        self.offset = offset

        # generate the indexes of sequence
        self.dataset_size = 0
        for interval in timeinterval:
            self.dataset_size = self.dataset_size + (
                (self.data.shape[1]) - (self.timestep - 1) * interval
            )
        self.dataset_size = self.data.shape[0] * self.dataset_size
        self.dataset_index = [i for i in range(0, self.dataset_size)]
        shuffle(self.dataset_index)
        print(
            "{} timelapse short sequence will be generated!".format(self.dataset_size)
        )

    def __len__(self):
        return int(np.floor(self.dataset_size / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Return a tuple with a batch of training inputs and targets
        inputs have a leading axis corresponding to time and targets dont
        :param idx: batch index
        :return:
        """

        def addnoise(data, signal, offset):
            gn = self.gaussnoise[random.randint(0, len(self.gaussnoise) - 1)]
            data_noise = np.random.poisson(data * signal) + np.random.normal(
                0, gn, data.shape
            )
            return normalize(data_noise, clip=True)

        def addgaussiannoise(data, signal, offset):
            gn = self.gaussnoise[random.randint(0, len(self.gaussnoise) - 1)]
            data_noise = data * signal + np.random.normal(0, gn, data.shape)
            return normalize(data_noise, clip=True)

        import random

        input = []
        target = []
        n_movie = self.data.shape[0]
        middle_frame = self.timestep // 2
        batch_index = [
            self.dataset_index[i]
            for i in range(
                idx * self.batch_size,
                min(self.dataset_size, (idx + 1) * self.batch_size),
            )
        ]
        for index in batch_index:
            movie_index = index % n_movie
            frame_index = index // (n_movie * len(self.timeinterval))
            interval_index = (index // n_movie) % len(self.timeinterval)
            frame = range(
                frame_index,
                (frame_index + self.timestep * self.timeinterval[interval_index]),
                self.timeinterval[interval_index],
            )
            image = self.data[movie_index, frame, :, :]
            signal = self.signallevel[random.randint(0, len(self.signallevel) - 1)]

            input_image = addnoise(image, signal, self.offset)  # noisy input
            target_image = image[middle_frame, :, :]  # clean target

            # add trailing dimension for channel
            input_image = np.expand_dims(input_image, axis=len(input_image.shape))
            if self.mode != "n2s":
                target_image = np.expand_dims(
                    target_image, axis=len(target_image.shape)
                )

            input.append(input_image)
            target.append(target_image)

        # return all in batch stacked as one big numpy array
        input = np.stack(input)
        target = np.stack(target)
        # np.savez('/home/yina/DeepDenoising/TimelapseDenoise/batch', input = input, target = target)
        return input, target


###


def normalize(
    x, pmin=0.2, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32
):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr

        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


###


def make_sequence(
    file_path,
    validation_ratio,
    batch_size,
    timestep,
    timeinterval,
    signallevel,
    gaussnoise,
    offset,
):
    try:
        data = np.load(file_path + "highSNR.npz")
    except:
        experimental_images(file_path)
        data = np.load(file_path + "highSNR.npz")
    data = data["highSNR"][:, 0:30, :]
    print("{} movies loaded!".format(data.shape))

    sample_size = data.shape[0]
    print(sample_size)
    validation_size = int(np.ceil(sample_size * validation_ratio))
    print(validation_size)
    return Imagesequence_generator(
        data[0 : sample_size - validation_size],
        batch_size,
        timestep,
        timeinterval,
        signallevel,
        gaussnoise,
        offset,
        training_mode,
        masker,
    ), Imagesequence_generator(
        data[-validation_size:],
        batch_size,
        timestep,
        timeinterval,
        signallevel,
        gaussnoise,
        offset,
        training_mode,
        masker,
    )


if __name__ == "__main__":
    file_path = (
        os.path.dirname(os.path.realpath(__file__)) + "/Experimental_data/training/"
    )
    from skimage.measure import compare_psnr

    try:
        data = np.load(file_path + "highSNR.npz")
    except:
        experimental_images(file_path)
        data = np.load(file_path + "highSNR.npz")

    batch_size = 32
    timestep = 11
    timeinterval = [3]
    signallevel = [100]
    gaussnoise = [5]
    offset = 100
    seq = Imagesequence_generator(
        data["highSNR"],
        batch_size,
        timestep,
        timeinterval,
        signallevel,
        gaussnoise,
        offset,
    )
    input, target = seq.__getitem__(3)
    index = 0
    if True:
        for i in range(len(input)):
            tiff.imwrite(
                file_path + "ROI_{}_input.tif".format(index),
                np.squeeze(input[i]),
                dtype="float32",
            )
            tiff.imwrite(
                file_path + "ROI_{}_target.tif".format(index),
                np.squeeze(target[i]),
                dtype="float32",
            )
            print(compare_psnr(target[i], input[i]))
            index = index + 1
