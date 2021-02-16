import os
import numpy as np
import torch
import scipy.io as sio
import scipy.ndimage as ip


from util import normalize, structure_axes
from torch.utils.data import DataLoader
from data_loader import randompatch

import structures
from structures import microtubule


gainmap_dir = os.path.dirname(os.path.dirname(os.getcwd()))
gainmap_dir = gainmap_dir + r'/Data/Confocal/gaincalibration.mat'


def addnoise(data, signals, gain_map=None, readnoise_map=None):
    """function for adding sythetic noise to clean data 
    input:
        data: 	    clean data source, with the last two dimensions X and Y
        signals:    e.g. [10, 20, 100], for rescale image intensity to add poisson noise
                    FIXME: currently only works for single signal level; 
			        output tensor dimension mismatch if input multiple signal level
        gain_map:   	 optional, sCMOS gain map: if None, load it from disk (512x512)
        readnoise_map:   optional, sCMOS readnoise map: if None, load it from disk (512x512)

    Output:
        noisy dataset with the same shape with input clean data
    """
    nx, ny = data.shape[-2:]
    if gain_map is None or readnoise_map is None:
        fmat_gain = sio.loadmat(gainmap_dir)
        gain_map = fmat_gain['gain']
        readnoise_map = fmat_gain['ccdvar']
        
    assert nx <= gain_map.shape[0] and ny <= gain_map.shape[1]

    rx, ry = randompatch(gain_map.shape, ny)
    gain_map = np.fabs(gain_map[rx:rx+nx, ry:ry+ny])
    readnoise_map = readnoise_map[rx:rx+nx, ry:ry+ny]

    for dim in range(len(data.shape[0:-2])):
        gain_map = np.repeat(gain_map, data.shape[dim], axis=dim)
        readnoise_map = np.repeat(readnoise_map, data.shape[dim], axis=dim)
    gain_map = np.reshape(gain_map, data.shape)
    readnoise_map = np.reshape(readnoise_map, data.shape)

    noisy = []
    for signal in signals:
        readnoise = [np.random.normal(0, np.sqrt(r), (1)) for r in np.nditer(readnoise_map)]
        readnoise = np.reshape(readnoise, data.shape)
        data_noise = np.random.poisson(data * signal * gain_map) + readnoise
        noisy.append(data_noise)
    if len(signals) > 1:
        noisy = np.stack(noisy, axis=0)
    else:
        noisy = noisy[0]
    noisy = normalize(noisy, clip=True)

    return torch.from_numpy(noisy)


class ConfocalDataset(torch.utils.data.Dataset):
    def __init__(self, datasize, axes, sim_structures, psignal_levels):
        super().__init__()

        self.datasize = datasize
        self.axes = structure_axes(axes)
        self.structures = sim_structures
        self.signal = psignal_levels

        # simulation parameters
        self.maxmove = 3  # pixel
        self.labeldensity = 40  # probes per pixel
        # Guassian PSF
        self.sigma = 1  # pixel
        self.psf = structures.gaussian_psf(self.sigma)  # generate 2D psf

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        # on the fly simulation
        clean = np.zeros(self.datasize[1::])
        noisy = np.zeros(self.datasize[1::])
        index_signal = [self.signal[np.random.randint(0, len(self.signal))]]

        im = np.zeros(self.datasize[2:])
        for s in self.structures:
            structure = globals()[s]
            for i in range(self.structures[s]):
                mt = structure(self.axes, pointstep=1.0 / self.labeldensity, verbose=False)
                im += mt.image()

            for t in range(self.axes['T']):
                clean[0, t, :] = normalize(ip.convolve(im[t, :], self.psf), clip=True)
                noisy[0, t, :] = addnoise(clean[0, t, :], index_signal)

        if self.axes['T'] == 1:
            clean = clean[:, 0, 0, :, :]
            noisy = noisy[:, 0, 0, :, :]

        return noisy, clean

    def __len__(self):
        return self.datasize[0]


# generate on the fly of training
def load_simulation(batch_size, datasize, axes, structures, psignal_levels, savetiff=False, gain_map=None, readnoise_map=None):
    """function for generating simulations to train an initial point for transfer learning
    Args:
        batch_size:     	        loading batch size
        datasize:                   simulated data size, shape should match axes
        axes:                       for example: "SCTYX" sample, channel, timepoints, Y, X
        structures:                 dictionary of structures to simulation: {'microtubule': 10}
        psignal_levels (seq): 		e.g. [10, 20, 100], for rescale image intensity to add poisson noise
    """
    assert len(datasize) == len(axes)

    dataset = ConfocalDataset(datasize, axes, structures, psignal_levels)
    kwargs = {'num_workers': 4, 'pin_memory': True} \
                if torch.cuda.is_available() else {}
    data_loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=False, **kwargs)

    return data_loader
