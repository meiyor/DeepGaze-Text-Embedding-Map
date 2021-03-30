import math
#from torch.autograd import Variable

import scipy.sparse
#from mine.models.mine import Mine
from scipy import signal
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

from layers import GaussianFilterNd, Conv2dMultiInput


def encode_scanpath_features(x_hist, y_hist, size, device=None, include_x=True, include_y=True, include_duration=False):
    assert include_x
    assert include_y
    assert not include_duration

    height = size[0]
    width = size[1]

    xs = torch.arange(width, dtype=torch.float32).to(device)
    ys = torch.arange(height, dtype=torch.float32).to(device)
    YS, XS = torch.meshgrid(ys, xs)

    print(xs.shape,ys.shape,'shapekk')
     
    XS = torch.repeat_interleave(
        torch.repeat_interleave(
            XS[np.newaxis, np.newaxis, :, :],
            repeats=x_hist.shape[0],
            dim=0,
        ),
        repeats=x_hist.shape[1],
        dim=1,
    )

    YS = torch.repeat_interleave(
        torch.repeat_interleave(
            YS[np.newaxis, np.newaxis, :, :],
            repeats=y_hist.shape[0],
            dim=0,
        ),
        repeats=y_hist.shape[1],
        dim=1,
    )
    print(XS.shape,x_hist.shape,x_hist,y_hist,'shape_vals')

    #XS -= x_hist.type('torch.cuda.FloatTensor')
    #YS -= y_hist.type('torch.cuda.FloatTensor')
    XS -= x_hist.unsqueeze(2).unsqueeze(3).type('torch.cuda.FloatTensor')
    YS -= y_hist.unsqueeze(2).unsqueeze(3).type('torch.cuda.FloatTensor')

    distances = torch.sqrt(XS**2 + YS**2)

    return torch.cat((XS, YS, distances), dim=1)


class FeatureExtractor(torch.nn.Module):
    def __init__(self, features, targets):
        super().__init__()
        self.features = features
        self.targets = targets

        self.outputs = {}

        for target in targets:
            def hook(module, input, output, target=target):
                self.outputs[target] = output.clone()
            getattr(self.features, target).register_forward_hook(hook)

    def forward(self, x):
        self.outputs.clear()
        x=x.type('torch.cuda.FloatTensor')
        self.features(x).type('torch.cuda.FloatTensor')

        return [self.outputs[target] for target in self.targets]


def upscale(tensor, size):
    tensor_size = torch.tensor(tensor.shape[2:]).type(torch.float32)
    target_size = torch.tensor(size).type(torch.float32)
    factors = torch.ceil(target_size / tensor_size)
    factor = torch.max(factors).type(torch.int64).to(tensor.device)
    assert factor >= 1

    tensor = torch.repeat_interleave(tensor, factor, dim=2)
    tensor = torch.repeat_interleave(tensor, factor, dim=3)

    tensor = tensor[:, :, :size[0], :size[1]]

    return tensor

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

class Finalizer(nn.Module):
    """Transforms a readout into a gaze prediction
    A readout network returns a single, spatial map of probable gaze locations.
    This module bundles the common processing steps necessary to transform this into
    the predicted gaze distribution:
     - resizing to the stimulus size
     - smoothing of the prediction using a gaussian filter
     - removing of channel and time dimension
     - weighted addition of the center bias
     - normalization
    """

    def __init__(
        self,
        sigma,
        kernel_size=None,
        learn_sigma=False,
        center_bias_weight=1.0,
        learn_center_bias_weight=True,
        saliency_map_factor=4,
    ):
        """Creates a new finalizer
        Args:
            size (tuple): target size for the predictions
            sigma (float): standard deviation of the gaussian kernel used for smoothing
            kernel_size (int, optional): size of the gaussian kernel
            learn_sigma (bool, optional): If True, the standard deviation of the gaussian kernel will
                be learned (default: False)
            center_bias (string or tensor): the center bias
            center_bias_weight (float, optional): initial weight of the center bias
            learn_center_bias_weight (bool, optional): If True, the center bias weight will be
                learned (default: True)
        """
        super(Finalizer, self).__init__()

        self.saliency_map_factor = saliency_map_factor

        self.gauss = GaussianFilterNd([2, 3], sigma, truncate=3, trainable=learn_sigma)
        self.center_bias_weight = nn.Parameter(torch.Tensor([center_bias_weight]), requires_grad=learn_center_bias_weight)

    def forward(self, readout, centerbias):
        """Applies the finalization steps to the given readout"""

        downscaled_centerbias = F.interpolate(
            centerbias.view(centerbias.shape[0], 1, centerbias.shape[1], centerbias.shape[2]),
            scale_factor=1 / self.saliency_map_factor)[:, 0, :, :]

        out = F.interpolate(readout, size=[downscaled_centerbias.shape[1], downscaled_centerbias.shape[2]])

        # apply gaussian filter
        out = self.gauss(out)

        # remove channel dimension
        out = out[:, 0, :, :]

        # add to center bias
        #print(self.center_bias_weight.type('torch.cuda.FloatTensor').shape,downscaled_centerbias.type('torch.cuda.FloatTensor').shape,out.shape,'shape_jjjjj')
        out = out + self.center_bias_weight.type('torch.cuda.FloatTensor') * downscaled_centerbias.type('torch.cuda.FloatTensor')

        out = F.interpolate(out[:, np.newaxis, :, :], size=[centerbias.shape[1], centerbias.shape[2]])[:, 0, :, :]

        # normalize
        out = out - out.logsumexp(dim=(1, 2), keepdim=True)
         
        return out

## only use this if you want to  use TEM-derive centerbias model in the prediction
class Finalizer_TEM(nn.Module):
    """Transforms a readout into a gaze prediction
    A readout network returns a single, spatial map of probable gaze locations.
    This module bundles the common processing steps necessary to transform this into
    the predicted gaze distribution:
     - resizing to the stimulus size
     - smoothing of the prediction using a gaussian filter
     - removing of channel and time dimension
     - weighted addition of the center bias
     - normalization
    """

    def __init__(self,sigma,kernel_size=None,learn_sigma=False,center_bias_weight=1.0,center_bias_weight_TEM=1.0,learn_center_bias_weight=True,saliency_map_factor=4,):
        super(Finalizer_TEM, self).__init__()
        self.saliency_map_factor = saliency_map_factor
        self.gauss = GaussianFilterNd([2, 3], sigma, truncate=3, trainable=learn_sigma)
        self.center_bias_weight = nn.Parameter(torch.Tensor([center_bias_weight]), requires_grad=learn_center_bias_weight)
        self.center_bias_weight_TEM = nn.Parameter(torch.Tensor([center_bias_weight_TEM]), requires_grad=learn_center_bias_weight)

    def forward(self, readout1, readout2, centerbias, centerbias_TEM):
        """Applies the finalization steps to the given readout"""

        #print(centerbias.shape,'centerbias_shape',centerbias.view(centerbias.shape[0], 1, centerbias.shape[1], centerbias.shape[2]).shape,centerbias.view(centerbias.shape[0], 1, centerbias.shape[1], centerbias.shape[2]),'loud_centerbias')
        downscaled_centerbias = F.interpolate(
            centerbias.view(centerbias.shape[0], 1, centerbias.shape[1], centerbias.shape[2]),
            scale_factor=1 / self.saliency_map_factor)[:, 0, :, :]
       
        #downscaled_centerbias_TEM = F.interpolate(
        #    readout2,
        #    scale_factor=1 / self.saliency_map_factor)[:, 0, :, :]

        downscaled_centerbias_TEM = F.interpolate(
            centerbias_TEM.view(centerbias_TEM.shape[0], 1, centerbias_TEM.shape[1], centerbias_TEM.shape[2]),
            scale_factor=1 / self.saliency_map_factor)[:, 0, :, :]

        #print(readout1.shape,'shape1')
        out1 = F.interpolate(readout1, size=[downscaled_centerbias.shape[1], downscaled_centerbias.shape[2]])

        # apply gaussian filter
        #print(out1.shape,'shape2')
        out1 = self.gauss(out1)
       
        #print(out1.shape,'shape3')      
        out1 = out1[:, 0, :, :]
         
        print(out1.shape,readout2.shape,'shape4')
        # add to center bias
        #print(self.center_bias_weight.type('torch.cuda.FloatTensor').shape,downscaled_centerbias.type('torch.cuda.FloatTensor').shape,out.sh$
        out1 = out1 + self.center_bias_weight.type('torch.cuda.FloatTensor') * (downscaled_centerbias.type('torch.cuda.FloatTensor'))

        out1 = F.interpolate(out1[:, np.newaxis, :, :], size=[centerbias.shape[1], centerbias.shape[2]])[:, 0, :, :]

        # normalize
        #out1 = out1 - out1.logsumexp(dim=(1, 2), keepdim=True)

        out2 = F.interpolate(readout2, size=[downscaled_centerbias.shape[1], downscaled_centerbias.shape[2]])

        # apply gaussian filter
        out2 = self.gauss(out2)

        # remove channel dimension
        out2 = out2[:, 0, :, :]

        # add to center bias
        #print(self.center_bias_weight.type('torch.cuda.FloatTensor').shape,downscaled_centerbias.type('torch.cuda.FloatTensor').shape,out.sh$
        out2 = out2 + self.center_bias_weight_TEM.type('torch.cuda.FloatTensor') * (downscaled_centerbias.type('torch.cuda.FloatTensor'))
        out2 = F.interpolate(out2[:, np.newaxis, :, :], size=[centerbias.shape[1], centerbias.shape[2]])[:, 0, :, :]

        # normalize
        #out2 = out2 - out2.logsumexp(dim=(1, 2), keepdim=True)
        
        #out = torch.zeros((out1.shape))
 
        #for k in range(0,out2.shape[0]):
        #     out[k,:,:]=out1[k,:,:]*torch.round(out2[k,:,:]/torch.max(out2[k,:,:]))   
          
        #out = out1
        #out = (out1 + out2)/2
        
        #out1 = out1 - out1.logsumexp(dim=(1, 2), keepdim=True)

        #out2 = out2 - out2.logsumexp(dim=(1, 2), keepdim=True)

        out = (out1+out2)/2
        #out = (out1 + out2)/2

        out = out - out.logsumexp(dim=(1, 2), keepdim=True)
        
        return out



class DeepGazeII(torch.nn.Module):
    def __init__(self, features, readout_network, downsample=2, readout_factor=16, saliency_map_factor=2, initial_sigma=8.0):
        super().__init__()

        self.readout_factor = readout_factor
        self.saliency_map_factor = saliency_map_factor

        self.features = features

        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.readout_network = readout_network
        self.finalizer = Finalizer(
            sigma=initial_sigma,
            learn_sigma=True,
            saliency_map_factor=self.saliency_map_factor,
        )
        self.downsample = downsample

    def forward(self, x, centerbias):
        orig_shape = x.shape
        x = F.interpolate(x, scale_factor=1 / self.downsample)
        x = self.features(x)

        readout_shape = [math.ceil(orig_shape[2] / self.downsample / self.readout_factor), math.ceil(orig_shape[3] / self.downsample / self.readout_factor)]
        x = [F.interpolate(item, readout_shape) for item in x]

        x = torch.cat(x, dim=1)
        x = self.readout_network(x)
        x = self.finalizer(x, centerbias)

        return x

    def train(self, mode=True):
        self.features.eval()
        self.readout_network.train(mode=mode)
        self.finalizer.train(mode=mode)


class DeepGazeIII(torch.nn.Module):
    def __init__(self, features, saliency_network, scanpath_network, fixation_selection_network, downsample=2, readout_factor=2, saliency_map_factor=2, included_fixations=-2, initial_sigma=8.0):
        super().__init__()

        self.downsample = downsample
        self.readout_factor = readout_factor
        self.saliency_map_factor = saliency_map_factor
        self.included_fixations = included_fixations

        self.features = features

        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network

        self.finalizer = Finalizer(
            sigma=initial_sigma,
            learn_sigma=True,
            saliency_map_factor=self.saliency_map_factor,
        )

    def forward(self, x, centerbias, x_hist=None, y_hist=None, durations=None):
        orig_shape = x.shape
        x = F.interpolate(x, scale_factor=1 / self.downsample)
        
        #sel_val=random.sample(range(x.shape[1]-1),3)
        #print(x.shape,x.type(),'wiii')
        #x[:,3:7,:,:]=x[:,3:7,:,:]*255
        x=x.type('torch.FloatTensor')
        #print(x.shape,x.type(),'wiii')
        TEM = 255*x[:,3:x.shape[1]+1,:,:]
        #x=self.features(x.cuda())
        x = self.features(x[:,[0,1,2],:,:])
        #TEMx = self.features(TEM[:,[0,1,2],:,:])

        readout_shape = [math.ceil(orig_shape[2] / self.downsample / self.readout_factor), math.ceil(orig_shape[3] / self.downsample / self.readout_factor)]
        x = [F.interpolate(item, readout_shape) for item in x]
     
        x = torch.cat(x,dim=1)
        x = self.saliency_network(x)

        if self.scanpath_network is not None:
            scanpath_features = encode_scanpath_features(x_hist, y_hist, size=(orig_shape[2], orig_shape[3]), device=x.device)
            scanpath_features = F.interpolate(scanpath_features, scale_factor=1 / self.downsample / self.readout_factor)
            y = self.scanpath_network(scanpath_features)
        else:
            y = None

        x = self.fixation_selection_network((x, y))

        x = self.finalizer(x.type('torch.cuda.FloatTensor'), centerbias.type('torch.cuda.FloatTensor'))
        
        return x

    def train(self, mode=True):
        self.features.eval()
        self.saliency_network.train(mode=mode)
        if self.scanpath_network is not None:
            self.scanpath_network.train(mode=mode)
        self.fixation_selection_network.train(mode=mode)
        self.finalizer.train(mode=mode)


class DeepGaze_TEM(torch.nn.Module):
    def __init__(self, features, saliency_network, saliency_network_TEM, conv_all_parameters, conv_all_parameters_trans,scanpath_network, fixation_selection_network, fixation_selection_network_TEM, downsample=2, readout_factor=2, saliency_map_factor=2, included_fixations=-2, initial_sigma=8.0):
        super().__init__()

        self.downsample = downsample
        self.readout_factor = readout_factor
        self.saliency_map_factor = saliency_map_factor
        self.included_fixations = included_fixations

        self.features = features

        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()
        
        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network
        torch.nn.init.xavier_normal_(self.fixation_selection_network.conv1.weight)
        torch.nn.init.xavier_normal_(self.fixation_selection_network.conv2.weight)
        self.conv_all_parameters_trans =  conv_all_parameters_trans
        torch.nn.init.xavier_normal_(self.conv_all_parameters_trans.conv0.weight)
        self.conv_all_parameters =  conv_all_parameters
        torch.nn.init.xavier_normal_(self.conv_all_parameters.conv0.weight)
        self.batch_norm3=nn.BatchNorm2d(1)
        self.linear = nn.Linear(600,300)
        torch.nn.init.xavier_normal_(self.linear.weight)
        self.dropout5 = nn.Dropout2d(0.18)

        self.finalizer = Finalizer(
            sigma=initial_sigma,
            learn_sigma=True,
            saliency_map_factor=self.saliency_map_factor,
        )

    def forward(self, x, TEM,  centerbias, centerbias_TEM,  x_hist=None, y_hist=None, durations=None):

        orig_shape = x.shape
        x=x.type('torch.cuda.FloatTensor')
        x = F.interpolate(x, scale_factor=1 / self.downsample)
        salv = []
        
        for k in range(0,x.shape[0]):
              TEM[k,0:21,:,:] = 255*((TEM[k,0:21,:,:]-torch.min(TEM[k,0:21,:,:]))/(torch.max(TEM[k,0:21,:,:])-torch.min(TEM[k,0:21,:,:])))
        
        x = self.features(x)
        readout_shape = [math.ceil(orig_shape[2] / self.downsample / self.readout_factor), math.ceil(orig_shape[3] / self.downsample / self.readout_factor)]
        
        x = [F.interpolate(item, readout_shape) for item in x]
        
        TEM=F.interpolate(TEM,size=readout_shape,mode="bilinear")
        
        salv=x
        
        salv.append(sal1)
        salv.append(sal2)
        salv.append(saln)
        salv.append(TEM[:,:,:,:])
        
        salv = torch.cat(salv, dim=1)
        
        x = self.saliency_network(salv)
        x = self.batch_norm3(x)
        
        if self.scanpath_network is not None:
            scanpath_features = encode_scanpath_features(x_hist, y_hist, size=(orig_shape[2], orig_shape[3]), device=x.device)
            scanpath_features = F.interpolate(scanpath_features, scale_factor=1 / self.downsample / self.readout_factor)
            y = self.scanpath_network(scanpath_features)
        else:
            y = None
           
        x = self.fixation_selection_network((x,y))
        x = self.dropout5(x)
        x = self.finalizer(x.type('torch.cuda.FloatTensor'), centerbias.type('torch.cuda.FloatTensor'))
        return x

    def train(self, mode=True):
        self.features.eval()
        self.conv_all_parameters.train(mode=mode)
        self.conv_all_parameters_trans.train(mode=mode)
        self.saliency_network.train(mode=mode)
        self.batch_norm3.train(mode=mode)
        if self.scanpath_network is not None:
            self.scanpath_network.train(mode=mode)
        self.fixation_selection_network.train(mode=mode)
        self.dropout5.train(mode=mode)
        self.finalizer.train(mode=mode)
