import numpy as np
import torch
import torch.nn.functional as F
from numpy import clip, exp
from scipy.signal import convolve2d
from util import normalize
import matplotlib
import matplotlib.pyplot as plt
from skimage.measure import compare_mse, compare_ssim
from statistics import mean, stdev

myfloat   = np.float32

def match_intensity(img1, img2):
    '''
    scale the intensity of img2 to img1 using least square
    img1 = A * img2 + b (element-wise)
    '''

    if img1.shape != img2.shape:
        raise('Input Image 1 & 2 have different sizes!')

    img2 = normalize(img2, clip = True)
    img1_flat, img2_flat = img1.flatten(), img2.flatten()
    A = np.stack([img2_flat, np.ones((img2_flat.size))], axis=0)
    b = img1_flat
    At = A.transpose()
    x = np.linalg.lstsq(At,b, rcond=None)
    img2_scale = img2 * x[0][0] + x[0][1]

    return img2_scale


def find_points_interval( array, inf, sup ):
    arr_inf = array >= inf
    arr_sup = array <= sup
    ind = np.argwhere( arr_inf * arr_sup)
    return ind


def frc( img1 , img2 , width_ring = 1):   
    '''
    Calculate frc curve between two images
    '''
  
    if img1.shape != img2.shape:
        raise('Input Image 1 & 2 have different sizes!')

    #  Get the Nyquist frequency
    nx,ny = img1.shape
    if nx == ny:
        nmax = nx
    elif nx < ny:
        nmax = ny
    else:
        nmax = nx 
    freq_nyq = int( np.floor( nmax/2.0 ) )

    #  Create Fourier grid
    x = np.arange( -np.floor( nx/2.0 ) , np.ceil( nx/2.0 ) )
    y = np.arange( -np.floor( ny/2.0 ) , np.ceil( ny/2.0 ) )
    assert x.size == nx and y.size == ny

    x,y = np.meshgrid( x , y )
    map_dist = np.sqrt( x*x + y*y )


    #  FFT transforms of the input images
    fft_img1 = np.fft.fftshift( np.fft.fft2( img1 ) )
    fft_img2 = np.fft.fftshift( np.fft.fft2( img2 ) )
    
    #  Calculate FRC 
    C1 = [];  C2 = [];  C3 = [];  n = []
    r = 0.0
    l = 0
    radii = []
    while r + width_ring < freq_nyq :
        ind_ring = find_points_interval( map_dist , r , r + width_ring )
        
        aux1 = fft_img1[ind_ring[:,0],ind_ring[:,1]]
        aux2 = fft_img2[ind_ring[:,0],ind_ring[:,1]]

        C1.append( np.sum( aux1 * np.conjugate(aux2) ) )
        C2.append( np.sum( np.abs( aux1 )**2 ) )
        C3.append( np.sum( np.abs( aux2 )**2 ) )

        n.append( len( aux1 ) )
        
        radii.append( r )
        r += width_ring
        l += 1

    radii = np.array( radii );  n = np.array( n )
    FRC = np.abs( np.array( C1 ) )/ np.sqrt( np.array( C2 ) * np.array( C3 ) )

    #  Plot FRC curve (alone)
    spatial_freq = radii / myfloat( freq_nyq )       
    
    return FRC , spatial_freq


def compare_avgFRC(img1, img2):
    FRC, _ = frc(img1, img2)
    return FRC.mean()

def quantify(results, metrics, target, net_output):
    all_metrics = ['mse', 'ssmi', 'frc']
    switcher = {
        'mse':compare_mse,
        'ssmi':compare_ssim,
        'frc':compare_avgFRC
    }
    assert target.shape == net_output.shape and len(target.shape) == 2

    if metrics is 'all':
        metrics = all_metrics
    else:
        assert all([metric in all_metrics for metric in metrics])

    for metric in metrics:
        func = switcher.get(metric)
        if metric not in results:
           results[metric]=[]
        results[metric].append(func(target, net_output))

    return results


def plot_quantifications(model_results, model_labels,metrics, ylabel=None, xlabel=None,title=None):
    assert len(model_results) == len(model_labels)

    plt.style.use('ggplot')
    font = {'family' : 'sans-serif',
            'sans-serif': 'Arial',
            'weight' : 'normal',
            'size'   : 20}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(nrows=1, ncols=len(metrics), sharex=True, figsize=(20,8), constrained_layout=True)

    for i in range(len(metrics)):
        ax = axs[i]

        for j, result in enumerate(model_results):
            x = list(map(int, result.keys())) 
            #x = result.keys()
            yavg = []
            yerr = []
            for x0 in result.keys():
                y = result[x0][metrics[i]]
                y.sort()
                if len(y) > 2:
                    yavg.append(mean(y[1:-1]))
                    yerr.append(stdev(y[1:-1]))
                else:
                    yavg.append(mean(y))
                    yerr.append(stdev(y))
            ax.errorbar(x, yavg, yerr=yerr, label = model_labels[j], linewidth=2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel[i])
	
        if i == 0:
    	    fig.legend(bbox_to_anchor=(.08,1.02,1.,.102), ncol=2, loc='lower left', borderaxespad=0.)

    fig.suptitle(title,y=1.15, fontsize=28)
