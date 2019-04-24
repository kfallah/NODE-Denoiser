import numpy as np
import torch
from torch.utils import data
from PIL import Image
import h5py

class RandomPatchDataset(data.Dataset):
    'Characterizes a dataset for loading patches for denoising'
    
    def __init__(self, filepath, patch_size=0, train_data=True, transform=None):
        'Initialization'
        super(RandomPatchDataset, self).__init__()
        self.patch_size = patch_size
        self.transform = transform
        self.files = glob.glob(os.path.join(filepath, '*.png'))
        self.train_data = train_data
                        
        if self.train_data:
            self.length = len(self.files) * len(scales) * patch_per_image
        else:
            self.length = len(self.files)
            
    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, index):
        'Generates one single image patch'
        file_idx = (index // (len(scales) * patch_per_image)) % len(self.files)
        scale_idx = (index // patch_per_image) % len(scales)
        patch_idx = index % patch_per_image
        
        if not self.train_data:
            image = Image.open(self.files[index])
        else:
            image = Image.open(self.files[file_idx])
            if self.transform is not None:        
                image = self.transform(image)
        if self.train_data:
            height, width = image.size
            image = image.resize((int(height*scales[scale_idx]), int(width*scales[scale_idx])), Image.BICUBIC)
        patch = image = np.float32(np.array(image) / 255., axis=0)
        if self.train_data:
            patch = np.expand_dims(extract_patches_2d(image, (self.patch_size, self.patch_size), 
                                                  max_patches=patch_per_image), axis=1)[patch_idx,:]  
        return torch.Tensor(patch)
    
class h5pyDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_name, augment=False):
        'Initialization'
        super(h5pyDataset, self).__init__()
        self.file_name = file_name
        self.augment = augment
        with h5py.File(self.file_name, 'r') as db:
            self.length = len(db.keys())

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        with h5py.File(self.file_name, 'r') as db:
            image = np.array(db[str(index)])
    
        if self.augment:
            data_augmentation(image)
            
        return torch.Tensor(image)
    
class ImageDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, images, augment=False):
        'Initialization'
        super(ImageDataset, self).__init__()
        self.images = images
        self.augment = augment

    def __len__(self):
        'Denotes the total number of samples'
        if isinstance(self.images,np.ndarray):
            return self.images.shape[0]
        else:
            return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = torch.Tensor(self.images[index])
    
        if self.augment:
            data_augmentation(image)
            
        return image
    
def data_augmentation(image):
    out = np.transpose(image, (1,2,0))
    mode = np.random.randint(0,8)
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))
