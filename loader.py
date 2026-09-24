from torch.utils.data import Dataset
import torch
import numpy as np
import random
from scipy import ndimage



IMAGE_MEAN = np.array([126.79130682, 76.93532417, 55.01129056], dtype=np.float32)
IMAGE_STD = np.array([81.34627638, 54.97843467, 43.56014316], dtype=np.float32)
# IMAGE_MEAN = np.array([92.3947], dtype=np.float32)
# IMAGE_STD = np.array([70.8074], dtype=np.float32)


def dataset_normalized(imgs, mean=None, std=None):
    if mean is None or std is None:
        imgs_std = np.std(imgs)
        imgs_mean = np.mean(imgs)
    else:
        imgs_mean = mean
        imgs_std = std
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    return imgs_normalized

def normalize_depth(depth, eps=1e-8):
    depth = depth.astype(np.float32)

    d_min = np.min(depth)
    d_max = np.max(depth)

    depth = (depth - d_min) / (d_max - d_min + eps)

    return depth


class Polyp_datasets(Dataset):
    def __init__(self, path_Data, train=True, Test=False, test_dataset=None):
        super(Polyp_datasets, self)
        self.train = train
        if train:
            self.data = np.load(path_Data + 'data_train.npy')
            self.mask = np.load(path_Data + 'mask_train.npy')
            self.depth = np.load(path_Data + 'depth_train.npy')
        else:
            if Test:
                suffix = '' if test_dataset is None else f'_{test_dataset}'
                self.data = np.load(path_Data + f'data_test{suffix}.npy')
                self.mask = np.load(path_Data + f'mask_test{suffix}.npy')
                self.depth = np.load(path_Data + f'depth_test{suffix}.npy')
            else:
                self.data = np.load(path_Data + 'data_val.npy')
                self.mask = np.load(path_Data + 'mask_val.npy')
                self.depth = np.load(path_Data + 'depth_val.npy')

        self.data = dataset_normalized(self.data, mean=IMAGE_MEAN, std=IMAGE_STD)
        self.mask = np.expand_dims(self.mask, axis=3) / 255.0
        self.depth = np.expand_dims(self.depth, axis=3) / 255.0

    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        depth = self.depth[indx]
        if self.train:
            if random.random() > 0.5:
                img, seg, depth = self.random_rot_flip(img, seg, depth)
            if random.random() > 0.5:
                img, seg, depth = self.random_rotate(img, seg, depth)

        seg = torch.tensor(seg.copy())
        img = torch.tensor(img.copy())
        depth = torch.tensor(depth.copy())
        img = img.permute(2, 0, 1)
        seg = seg.permute(2, 0, 1)
        depth = depth.permute(2, 0, 1)

        return img, seg, depth
    
    def random_rot_flip(self, image, label, depth):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        depth = np.rot90(depth, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        depth = np.flip(depth, axis=axis).copy()
        return image, label, depth
    
    def random_rotate(self, image, label, depth):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        depth = ndimage.rotate(depth, angle, order=0, reshape=False)
        return image, label, depth

    def __len__(self):
        return len(self.data)