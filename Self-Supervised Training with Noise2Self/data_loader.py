import os
import math
import tifffile as tiff
import numpy as np
import torch
import json
import scipy.io as sio


from util import normalize
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage import img_as_float


gainmap_dir = r'/home/yina/DeepDenoising/Data/Confocal/gaincalibration.mat'


def tiff_loader(path):
    img = tiff.imread(path)
    return img


def randompatch(data_shape, patch_size):
    '''
    return the upper left corner coordinate of the patch
    '''
    np.random.seed(0)
    nx, ny = data_shape[-2], data_shape[-1]
    rx = int(np.random.randint(0, high=nx-patch_size, size=1, dtype=int))
    ry = int(np.random.randint(0, high=nx-patch_size, size=1, dtype=int))
    return rx, ry


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
    nsample = data.shape[0]
    if gain_map == None or readnoise_map == None:
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
        data_noise = np.random.poisson(data.numpy() * signal * gain_map) + readnoise
        noisy.append(data_noise) 
    if len(signals) > 1:
        noisy = np.stack(noisy, axis=0)
    else:
        noisy = noisy[0]
    noisy = normalize(noisy, clip = True)

    return torch.from_numpy(noisy)


def randomcrop(data, patch_size):
    data = normalize(data, clip = True)

    print(data.shape)
    if data.shape[-2] == patch_size and data.shape[-1] == patch_size:
        #print("Data has same shape with patch size")
        img = data
        img = img.squeeze(axis=1)
        img = np.expand_dims(img, axis=1)
        img = normalize(img, clip = True)
        return torch.from_numpy(img)
    elif data.shape[-2] < patch_size or data.shape[-1] < patch_size:
        return None
    else:
        avg_intensity = 0.0
        threshold = 0.05
        rx, ry = 0, 0
        while avg_intensity < threshold:
            rx, ry = randompatch(data.shape, patch_size)
            cropped = data[..., rx:rx+patch_size, ry:ry+patch_size]
            avg_intensity = np.mean(cropped)

        img = data[..., rx:rx+patch_size, ry:ry+patch_size]
        img = img.squeeze(axis=1)
        img = np.expand_dims(img, axis=1)
        img = normalize(img, clip = True)

        return torch.from_numpy(img)


class ConfocalDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()

        self.noisy = data[0]
        self.clean = data[1]        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        return self.noisy[index,:], self.clean[index,:]

    def __len__(self):
        return self.noisy.shape[0]


def gather_files(root, sample_size, split_ratio, types, train=True):
    samples = []

    # types: cell structures
    subdirs = [os.path.join(root, name+'/') for name in os.listdir(root)
               if (os.path.isdir(os.path.join(root, name+'/')) and name in types)]

    for subdir in subdirs:
        file_counts = len([name for name in os.listdir(subdir) if name[-4:]=='.tif'])
        file_index = math.ceil(file_counts*split_ratio)

        startfile = 0
        file_to_use = 0
        if train:
            start_file = file_index + 1
            file_to_use = file_counts - file_index
        else:
            start_file = 1
            file_to_use = file_index
        fov_per_file = math.ceil(sample_size/file_to_use)

        for f in range(file_to_use):
            fname = str(start_file + f) + '.tif'
            for fov in range(fov_per_file):
                image_file = os.path.join(subdir, fname)
                samples.append(image_file)

    return samples[0:sample_size]


def load_confocal(root, train, batch_size, psignal_levels, sample_size, split_ratio=0.2, types='all', captures=2,
                  patch_size=256, loader=tiff_loader, gain_map=None, readnoise_map=None):
    """function for loading the denoising confocal dataset for testing 
    file structure:
        data_root/type/xxx.tif
        type:       ['DNA', 'lysosome', 'microtubule', 'mitochonria']
        xxx.tif:    timelapse movie in each fov --> can set to use fewer shots

    Args:
        root (str): 			root directory to the dataset
        train (bool): 			Training set if True, else Test set
        batch_size:     	        loading batch size
        psignal_levels (seq): 		e.g. [10, 20, 100], for rescale image intensity to add poisson noise
        sample_size: 			number of desired training or testing data
        split_ratio (optional): 	ratio for spliting tiff files for trainig and testing
        types (seq, optional): 		e.g. ['DNA', 'lysosome', 'microtubule', 'mitochonria']
        captures (int): 		select # images within one folder
        patch_size (optional): 	        cropping patch size
    """
    transform = randomcrop
    all_types = ['DNA', 'lysosome', 'microtubule', 'mitochondria']
    if types is 'all':
        types = all_types
    else:
        assert all([img_type in all_types for img_type in types])

    # generating noisy and clean data pairs
    clean_file = gather_files(root, sample_size, split_ratio, types, train=train)
    clean = loader(clean_file)
    #print("shape of loaded data:", clean.shape)
    if sample_size == 1:
       clean = np.expand_dims(clean,axis=0)
    if transform is not None:
       clean = transform(clean[:,0:captures,:], patch_size)

    noisy = addnoise(clean, psignal_levels, gain_map=gain_map, readnoise_map=readnoise_map) 
    if len(psignal_levels) > 1:
       clean = np.repeat(clean, len(psignal_levels), axis=0)
    assert noisy.shape == clean.shape

    dataset_info = {'Dataset': 'train' if train else 'test',
                        'Peak signal levels': psignal_levels,
                        f'{len(types)} Types': types,
                        '# samples': len(clean_file)
                        }
    #print(json.dumps(dataset_info, indent=4))
        
    dataset = ConfocalDataset([noisy, clean])
    kwargs = {'num_workers': 4, 'pin_memory': True} \
              if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                  shuffle=True, drop_last=False, **kwargs)

    return data_loader, [noisy, clean]
