# This file contains a bunch of functions from Benjamin's lab2im package
# (it's just much lighter to import...)
import nibabel as nib
import numpy as np
import os
from torch import nn
import torch
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage import gaussian_filter as gauss_filt
import csv
import matplotlib.pyplot as plt

from torch.utils import data
class MRIdataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, device='cpu', time_choice='half', slices=None, transform=None, scalefactor=2):
        'Initialization'
        """
        list_IDs=lista de imagenes con su path
        labels=lista de labels de las imagenes
        """
        # self.labels = labels
        self.list_IDs = list_IDs
#         self.n_volumes = n_volumes
        self.transform = transform
        self.device = device
        self.time_choice=time_choice
        self.slices = slices
        self.scalefactor=scalefactor
        self.ratio = scalefactor.as_integer_ratio()[0] # works with scale 1.25, 1.5, 1.75 and 2 (multiples of 4)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        # randomly pick an image and read it
#         index = np.random.randint(n_training)
        vol, aff = load_volume(self.list_IDs[idx]) 
            
        if len(vol.shape)>3:
            #pick random time frame or choose one
            if self.time_choice=='random':
                vol=vol[:,:,:,np.random.randint(vol.shape[3])]
            else:
                vol=vol[:,:,:,int(vol.shape[3]/2)]
#         hr=vol
      
        #check divisible dimensions
        sizes=np.zeros(len(vol.shape),dtype=int)
        sizes[:]=vol.shape[:]
        for d,dim in enumerate(sizes[:3]):
            if dim%self.scalefactor!=0:
                rest=dim%self.ratio
                sizes[d]= dim+(self.ratio-rest)

        hr = np.zeros(sizes)
        hr[:vol.shape[0], :vol.shape[1],:vol.shape[2]]=vol[:vol.shape[0], :vol.shape[1],:vol.shape[2]]
                
        if self.slices==None:
            init=0
            n_slices=hr.shape[2]
           
        else:
            n_slices=self.slices
            rest=n_slices%(self.ratio)
            if rest !=0:
                aux = (self.ratio)-rest if rest > self.ratio/2 else -(rest)
                n_slices=np.clip(n_slices+aux,self.ratio,total_slices)
            init=np.random.randint(hr.shape[2]+1-n_slices)
            
        hr = hr[:,:,init:init+n_slices]     
        
        hr = np.squeeze(hr)
#         orig_shape = hr.shape
#         orig_center = (np.array(orig_shape) - 1) / 2
        hr = torch.tensor(hr, device=self.device)

        ratios = self.scalefactor*np.ones(3)
#         
        lr = myzoom_torch(hr, 1 / ratios, device=self.device)


        # We also renormalize here (as we do at test time!)
        # And also keep the target at the same scale
        maxi = torch.max(lr)
        lr = lr / maxi
        target = hr / maxi

        # Finally, we go back to the original resolution
        upscaled = myzoom_torch(lr, ratios, device=self.device)

        return lr, upscaled, target
    
class MRIdataset_NoDownsize(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, device='cpu', time_choice='half', slices=None,transform=None, scalefactor=2):
        'Initialization'
        """
        list_IDs=lista de imagenes con su path
        labels=lista de labels de las imagenes
        """
        # self.labels = labels
        self.list_IDs = list_IDs
#         self.n_volumes = n_volumes
        self.transform = transform
        self.device = device
        self.time_choice=time_choice
        self.slices=slices #Might be necessary to split excesively BIG images
        self.scalefactor=scalefactor
        down_ratio,up_ratio=scalefactor.as_integer_ratio()
        
        if slices!=None and slices%scalefactor!=0:
            rest=slices%up_ratio
            self.slices= slices-rest
            print('n_slices has changed:',slices)
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

         # pick an image and read it
        hr, aff = load_volume(self.list_IDs[idx])
        
        if len(hr.shape)>3:
        #pick random time frame or choose one
            if self.time_choice=='random':
                hr=hr[:,:,:,np.random.randint(hr.shape[3])]
            else:
                hr=hr[:,:,:,int(hr.shape[3]/2)]
            #hr is now a single 3D volume [x,y,z]    
        
        if self.slices!=None:
            #Choose a cube of side = slices or prisma if any dimension is lower
            h = np.clip(self.slices, 0, hr.shape[0])
            w = np.clip(self.slices, 0, hr.shape[1])
            d = np.clip(self.slices, 0, hr.shape[2])
            if self.time_choice=='random':
                init_h=np.random.randint(hr.shape[0]+1-h)
                init_w=np.random.randint(hr.shape[1]+1-w)
                init_d=np.random.randint(hr.shape[2]+1-d)
            else:
                init_h=np.clip(int((hr.shape[0]/2)-h/2),0,None)
                init_w=np.clip(int((hr.shape[1]/2)-w/2),0,None)
                init_d=np.clip(int((hr.shape[2]/2)-d/2),0,None)
                
            hr = hr[init_h:init_h+h, init_w:init_w+w, init_d:init_d+d]
            #if n_slices hr is now a 3d cube [w,h,d]
        
        hr = np.squeeze(hr)
#         orig_shape = hr.shape
#         orig_center = (np.array(orig_shape) - 1) / 2
        hr = torch.tensor(hr, device=self.device)

        ratios = self.scalefactor*np.ones(3)
#         
#         lr = myzoom_torch(hr, 1 / ratios, device=self.device)


        # We also renormalize here (as we do at test time!)
        # And also keep the target at the same scale
        maxi = torch.max(hr)
#         lr = lr / maxi
        hr = hr / maxi

        # Finally, we UPSAMPLE
        upscaled = myzoom_torch(hr, ratios, device=self.device)

        return hr.float(), upscaled.float()
    
class MRIdataset_randomfactor(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, device='cpu', time_choice='half', slices=None, transform=None):
        'Initialization'
        """
        list_IDs=lista de imagenes con su path
        labels=lista de labels de las imagenes
        """
        # self.labels = labels
        self.list_IDs = list_IDs
#         self.n_volumes = n_volumes
        self.transform = transform
        self.device = device
        self.time_choice=time_choice
        self.slices = slices
        self.factors = [1.25, 1.5, 1.75, 2]
#         self.scalefactor=scalefactor
#         self.ratio = scalefactor.as_integer_ratio()[0] # works with scale 1.25, 1.5, 1.75 and 2 (multiples of 4)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #random scalefactor
        scalefactor=self.factors[np.random.randint(4)]
        ratio = scalefactor.as_integer_ratio()[0]
        
        # randomly pick an image and read it
#         index = np.random.randint(n_training)
        vol, aff = load_volume(self.list_IDs[idx]) 
            
        if len(vol.shape)>3:
            #pick random time frame or choose one
            if self.time_choice=='random':
                vol=vol[:,:,:,np.random.randint(vol.shape[3])]
            else:
                vol=vol[:,:,:,int(vol.shape[3]/2)]
#         hr=vol
      
        #check divisible dimensions
        sizes=np.zeros(len(vol.shape),dtype=int)
        sizes[:]=vol.shape[:]
        for d,dim in enumerate(sizes[:3]):
            if dim%scalefactor!=0:
                rest=dim%ratio
                sizes[d]= dim+(ratio-rest)

        hr = np.zeros(sizes)
        hr[:vol.shape[0], :vol.shape[1],:vol.shape[2]]=vol[:vol.shape[0], :vol.shape[1],:vol.shape[2]]
                
        if self.slices==None:
            init=0
            n_slices=hr.shape[2]
           
        else:
            n_slices=self.slices
            rest=n_slices%(ratio)
            if rest !=0:
                aux = (ratio)-rest if rest > ratio/2 else -(rest)
                n_slices=np.clip(n_slices+aux,ratio,total_slices)
            init=np.random.randint(hr.shape[2]+1-n_slices)
            
        hr = hr[:,:,init:init+n_slices]     
        
        hr = np.squeeze(hr)
#         orig_shape = hr.shape
#         orig_center = (np.array(orig_shape) - 1) / 2
        hr = torch.tensor(hr, device=self.device)

        ratios = scalefactor*np.ones(3)
#         
        lr = myzoom_torch(hr, 1 / ratios, device=self.device)


        # We also renormalize here (as we do at test time!)
        # And also keep the target at the same scale
        maxi = torch.max(lr)
        lr = lr / maxi
        target = hr / maxi

        # Finally, we go back to the original resolution
        upscaled = myzoom_torch(lr, ratios, device=self.device)

        return lr, upscaled, target
    
def OpNeuroReader(opNeuroPath):
    print('Not implemented')
    
def ensure_directory(filename):
    ''' Checks if directory of file exists and if not creates the whole tree from where it is needed'''
    to_check=filename.rsplit('/',1)[0]
    flag=os.path.isdir(to_check)
#     print(to_check,flag)
    if not flag:
        ensure_directory(to_check)
        os.mkdir(to_check)
        print(to_check, 'has been created')

# Load nifti or mgz file
def load_volume(path_volume):

    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz')), 'Unknown data file: %s' % path_volume

    x = nib.load(path_volume)
    volume = x.get_fdata()
    aff = x.affine

    return volume, aff

# Save nifti or mgz file
def save_volume(volume, aff, path):

    header = nib.Nifti1Header()

    if aff is None:
        aff = np.eye(4)

    nifti = nib.Nifti1Image(volume, aff, header)

    nib.save(nifti, path)
    
def evaluate_memory(step):
    t = torch.cuda.get_device_properties(0).total_memory/1024/1024/1024
    r = torch.cuda.memory_reserved(0)/1024/1024
    a = torch.cuda.memory_allocated(0)/1024/1024
    f = r-a  # free inside reserved
    print(step,'- Total:', t, 'GB Reserv:', r, 'GB Alloc:', a, 'GB Free:',f)

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices), figsize=(30,10))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
       axes[i].axis('equal')

def show_overview(img, i=None, j=None, k=None, t=None):
    x,y,z,tt = img.shape
    if i==None:
        i = int(x/2)
    if j==None:
        j = int(y/2)
    if k==None:
        k = int(z/2)
    if t==None:
        t = int(tt/2)

    slice_0 = img[i, :, :,t]
    slice_1 = img[:, j, :,t]
    slice_2 = img[:, :, k,t]

    show_slices([slice_0, slice_1, slice_2])

def TVloss_3d(volume):
  # volume [bs, c, sagital, coronal, axial]
    bs_img, c_img, sag_sz, cor_sz, axi_sz = volume.shape
    tv_sag = torch.pow(volume[:,:,:,1:,:]-volume[:,:,:,:-1,:],2).sum()
    tv_cor = torch.pow(volume[:,:,1:,:,:]-volume[:,:,:-1,:,:],2).sum()
    tv_axi = torch.pow(volume[:,:,:,:,1:]-volume[:,:,:,:,:-1],2).sum()
    return (tv_sag+tv_cor+tv_axi)/(bs_img*c_img*sag_sz*cor_sz*axi_sz)


def save_log(filename, epoch, loss, mse_loss, TV_loss, val_mse_loss=0, val_psnr_loss=0, val_ssim_loss=0, name_list=None):
    if name_list==None:
        name_list=['epoch', 'loss', 'mse_loss', 'TV_loss', 'val_mse_loss', 'val_psnr_loss', 'val_ssim_loss']
    
    if not os.path.isfile(filename):
        with open(filename, mode='a') as result_file:
            result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(name_list)
            result_writer.writerow([epoch, loss, mse_loss, TV_loss, val_mse_loss, val_psnr_loss, val_ssim_loss])
    else:
         with open(filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss, mse_loss, TV_loss, val_mse_loss, val_psnr_loss, val_ssim_loss])

def empty_folder(path):
    files= glob.glob(path+'/*')
    for f in files:
        os.remove(f)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr_new = lr * (lr_gamma ** (epoch // lr_steps))
    # print('lr:',lr_new)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

            
            
# Creates a 3x3 rotation matrix from a vector with 3 rotations about x, y, and z
def make_rotation_matrix(rot):

    Rx = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
    Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
    Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])
    R = np.matmul(np.matmul(Rx, Ry), Rz)

    return R


def myzoom_torch(X, factor, device='cpu'):

    if len(X.shape)==3:
        X = X[..., None]

    delta = (1.0 - factor) / (2.0 * factor)
    newsize = np.round(X.shape[:-1] * factor).astype(int)

    vx = torch.arange(delta[0], delta[0] + newsize[0] / factor[0], 1 / factor[0], device=device)
    vy = torch.arange(delta[1], delta[1] + newsize[1] / factor[1], 1 / factor[1], device=device)
    vz = torch.arange(delta[2], delta[2] + newsize[2] / factor[2], 1 / factor[2], device=device)

    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0]-1)] = (X.shape[0]-1)
    vy[vy > (X.shape[1] - 1)] = (X.shape[1] - 1)
    vz[vz > (X.shape[2] - 1)] = (X.shape[2] - 1)

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0]-1)] = (X.shape[0]-1)
    wcx = vx - fx
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1]-1)] = (X.shape[1]-1)
    wcy = vy - fy
    wfy = 1 - wcy

    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2]-1)] = (X.shape[2]-1)
    wcz = vz - fz
    wfz = 1 - wcz

    Y = torch.zeros([newsize[0], newsize[1], newsize[2], X.shape[3]], device=device)

    for channel in range(X.shape[3]):
        Xc = X[:,:,:,channel]

        tmp1 = torch.zeros([newsize[0], Xc.shape[1], Xc.shape[2]], device=device)
        for i in range(newsize[0]):
            tmp1[i, :, :] = wfx[i] * Xc[fx[i], :, :] +  wcx[i] * Xc[cx[i], :, :]
        tmp2 = torch.zeros([newsize[0], newsize[1], Xc.shape[2]], device=device)
        for j in range(newsize[1]):
            tmp2[:, j, :] = wfy[j] * tmp1[:, fy[j], :] +  wcy[j] * tmp1[:, cy[j], :]
        for k in range(newsize[2]):
            Y[:, :, k, channel] = wfz[k] * tmp2[:, :, fz[k]] +  wcz[k] * tmp2[:, :, cz[k]]

    if Y.shape[3] == 1:
        Y = Y[:,:,:, 0]

    return Y

# Nearst negithbor or trilinear 3D interpolation with pytorch
def fast_3D_interp_torch(X, II, JJ, KK, mode, device='cpu'):
    if mode=='nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        if len(X.shape)==3:
            X = X[..., None]
        Y = torch.zeros([*II.shape, X.shape[3]], device=device)
        for channel in range(X.shape[3]):
            aux = X[:, :, :, channel]
            Y[:,:,:,channel] = aux[IIr, JJr, KKr]
        if Y.shape[3] == 1:
            Y = Y[:, :, :, 0]

    elif mode=='linear':
        ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz

        if len(X.shape)==3:
            X = X[..., None]

        Y = torch.zeros([*II.shape, X.shape[3]], device=device)
        for channel in range(X.shape[3]):
            Xc = X[:, :, :, channel]

            c000 = Xc[fx, fy, fz]
            c100 = Xc[cx, fy, fz]
            c010 = Xc[fx, cy, fz]
            c110 = Xc[cx, cy, fz]
            c001 = Xc[fx, fy, cz]
            c101 = Xc[cx, fy, cz]
            c011 = Xc[fx, cy, cz]
            c111 = Xc[cx, cy, cz]

            c00 = c000 * wfx + c100 * wcx
            c01 = c001 * wfx + c101 * wcx
            c10 = c010 * wfx + c110 * wcx
            c11 = c011 * wfx + c111 * wcx

            c0 = c00 * wfy + c10 * wcy
            c1 = c01 * wfy + c11 * wcy

            c = c0 * wfz + c1 * wcz

            Yc = torch.zeros(II.shape, device=device)
            Yc[ok] = c.float()
            Y[:,:,:,channel] = Yc

        if Y.shape[3]==1:
            Y = Y[:,:,:,0]

    else:
        raise Exception('mode must be linear or nearest')

    return Y


# Make a discrete gaussian kernel
def make_gaussian_kernel(sigma):
    sl = np.ceil(sigma * 2.5).astype(int)
    v = np.arange(-sl, sl+1)
    gauss = np.exp((-(v / sigma)**2 / 2))
    kernel = gauss / np.sum(gauss)

    return kernel


#
#

#
#

#
#
#
#
def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels

def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this funtion returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformated list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var

def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var

def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            volume = np.swapaxes(volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            volume = np.flip(volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (volume.shape[i] - 1)

    if return_aff:
        return volume, aff_flo
    else:
        return volume

def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    return img_ras_axes




def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels

def resample_like(vol_ref, aff_ref, vol_flo, aff_flo, method='linear'):
    """This function reslices a floating image to the space of a reference image
    :param vol_res: a numpy array with the reference volume
    :param aff_ref: affine matrix of the reference volume
    :param vol_flo: a numpy array with the floating volume
    :param aff_flo: affine matrix of the floating volume
    :param method: linear or nearest
    :return: resliced volume
    """

    T = np.matmul(np.linalg.inv(aff_flo), aff_ref)

    xf = np.arange(0, vol_flo.shape[0])
    yf = np.arange(0, vol_flo.shape[1])
    zf = np.arange(0, vol_flo.shape[2])

    my_interpolating_function = rgi((xf, yf, zf), vol_flo, method=method, bounds_error=False, fill_value=0.0)

    xr = np.arange(0, vol_ref.shape[0])
    yr = np.arange(0, vol_ref.shape[1])
    zr = np.arange(0, vol_ref.shape[2])

    xrg, yrg, zrg = np.meshgrid(xr, yr, zr, indexing='ij', sparse=False)
    n = xrg.size
    xrg = xrg.reshape([n])
    yrg = yrg.reshape([n])
    zrg = zrg.reshape([n])
    bottom = np.ones_like(xrg)
    coords = np.stack([xrg, yrg, zrg, bottom])
    coords_new = np.matmul(T, coords)[:-1, :]
    result = my_interpolating_function((coords_new[0, :], coords_new[1, :], coords_new[2, :]))

    if vol_ref.size == result.size:
        return result.reshape(vol_ref.shape)
    else:
        return result.reshape([*vol_ref.shape, vol_flo.shape[-1]])





# Computer linear (flattened) indices for linear interpolation of a volume of size nx x ny x nz at locations xx, yy, zz
# (as well as a boolean vector 'ok' telling which indices are inbounds)
# Note that it doesn't support sparse xx, yy, zz
def nn_interpolator_indices(xx, yy, zz, nx, ny, nz):
    xx2r = np.round(xx).astype(int)
    yy2r = np.round(yy).astype(int)
    zz2r = np.round(zz).astype(int)
    ok = (xx2r >= 0) & (yy2r >= 0) & (zz2r >= 0) & (xx2r <= nx - 1) & (yy2r <= ny - 1) & (zz2r <= nz - 1)
    idx = xx2r[ok] + nx * yy2r[ok] + nx * ny * zz2r[ok]
    return idx, ok

# Similar to nn_interpolator_indices but does not check out of bounds.
# Note that it *does* support sparse xx, yy, zz
def nn_interpolator_indices_nocheck(xx, yy, zz, nx, ny, nz):
    xx2r = np.round(xx).astype(int)
    yy2r = np.round(yy).astype(int)
    zz2r = np.round(zz).astype(int)
    idx = xx2r + nx * yy2r + nx * ny * zz2r
    return idx

# Subsamples a volume by a given ration in each dimension.
# It carefully accounts for origin shifts
def subsample(X, ratio, size, method='linear', return_locations=False):
    xi = np.arange(0.5 * (ratio[0] - 1.0), size[0] - 1 + 1e-6, ratio[0])
    yi = np.arange(0.5 * (ratio[1] - 1.0), size[1] - 1 + 1e-6, ratio[1])
    zi = np.arange(0.5 * (ratio[2] - 1.0), size[2] - 1 + 1e-6, ratio[2])
    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    interpolator = rgi((range(size[0]), range(size[1]), range(size[2])), X, method=method)
    Y = interpolator((xig, yig, zig))
    if return_locations:
        return Y, xig, yig, zig
    else:
        return Y


def upsample(X, ratio, size, method='linear', return_locations=False):
    start = (1.0 - ratio[0]) / (2.0 * ratio[0])
    inc = 1.0 / ratio[0]
    end = start + inc * size[0] - 1e-6
    xi = np.arange(start, end, inc)
    xi[xi < 0] = 0
    xi[xi > X.shape[0] - 1] = X.shape[0] - 1

    start = (1.0 - ratio[1]) / (2.0 * ratio[1])
    inc = 1.0 / ratio[1]
    end = start + inc * size[1] - 1e-6
    yi = np.arange(start, end, inc)
    yi[yi < 0] = 0
    yi[yi > X.shape[1] - 1] = X.shape[1] - 1

    start = (1.0 - ratio[2]) / (2.0 * ratio[2])
    inc = 1.0 / ratio[2]
    end = start + inc * size[2] - 1e-6
    zi = np.arange(start, end, inc)
    zi[zi < 0] = 0
    zi[zi > X.shape[2] - 1] = X.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    interpolator = rgi((range(X.shape[0]), range(X.shape[1]), range(X.shape[2])), X, method=method)
    Y = interpolator((xig, yig, zig))

    if return_locations:
        return Y, xig, yig, zig
    else:
        return Y

# Augmentation of FA with gaussian noise and gamma transform
def augment_fa(X, gamma_std, max_noise_std_fa):
    gamma_fa = np.exp(gamma_std * np.random.randn(1)[0])
    noise_std = max_noise_std_fa * np.random.rand(1)[0]
    Y = X + noise_std * np.random.randn(*X.shape)
    Y[Y < 0] = 0
    Y[Y > 1] = 1
    Y = Y ** gamma_fa
    return Y

# Augmentation of T1 intensities with random contrast, brightness, gamma, and gaussian noise
def augment_t1(X, gamma_std, contrast_std, brightness_std, max_noise_std):
    # TODO: maybe add bias field? If we're working with FreeSurfer processed images maybe it's not too important
    gamma_t1 = np.exp(gamma_std * np.random.randn(1)[0])  # TODO: maybe make it spatially variable?
    contrast = np.min((1.4, np.max((0.6, 1.0 + contrast_std * np.random.randn(1)[0]))))
    brightness = np.min((0.4, np.max((-0.4, brightness_std * np.random.randn(1)[0]))))
    noise_std = max_noise_std * np.random.rand(1)[0]
    Y = ((X - 0.5) * contrast + (0.5 + brightness)) + noise_std * np.random.randn(*X.shape)
    Y[Y < 0] = 0
    Y[Y > 1] = 1
    Y = Y ** gamma_t1
    return Y


def rescale_voxel_size(volume, aff, new_vox_size):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    volume_filt = gauss_filt(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = rgi((x, y, z), volume_filt)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2



class SobelFilter3d(nn.Module):
    def __init__(self):
        super(SobelFilter3d, self).__init__()
        self.sobel_filter = self.setup_filter()

    def setup_filter(self):
        sobel_filter = nn.Conv3d(1, 3, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor(
                [
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                    [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
                ]))
        sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            ]))
        sobel_filter.weight.data[2, 0].copy_(
            torch.FloatTensor([
                [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            ])
        )
        sobel_filter.bias.data.zero_()
        for p in sobel_filter.parameters():
            p.requires_grad = False
        return sobel_filter

    def forward(self, x):
        bs, ch, l, h, w = x.shape
        combined_edge_map = 0
        for idx in range(ch):
            g_x = self.sobel_filter(x[:, idx:idx+1])[:, 0]
            g_y = self.sobel_filter(x[:, idx:idx+1])[:, 1]
            g_z = self.sobel_filter(x[:, idx:idx+1])[:, 2]
            combined_edge_map += torch.sqrt((g_x **2 + g_y ** 2 + g_z ** 2))
        return combined_edge_map

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim3D(img1, img2, window, self.window_size, channel, self.size_average)

    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)