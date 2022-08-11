import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch
from data.utils import get_transform_params, transform_image, take_single_channel, TransformParams

class AlignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        if self.opt.input_nc in [1, 3]:
            file_idx = index // 9
            channel_idx = index % 9
        elif self.opt.input_nc == 9:
            file_idx = index
            channel_idx = -1
        else:
            raise NotImplementedError(f"Unsupported number of input channels: {self.opt.input_nc}")

        A_path = self.A_paths[file_idx % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[file_idx % self.B_size]
        A_img = np.load(A_path)
        B_img = np.load(B_path)

        if channel_idx >= 0:
            A_img = take_single_channel(A_img, channel_idx, self.opt.input_nc)
            B_img = take_single_channel(B_img, channel_idx, self.opt.input_nc)
            A_path += f";{channel_idx}" # encode it here for visualizer to work properly
            B_path += f";{channel_idx}"


        crop_size = self.opt.crop_size # 256
        load_size = self.opt.load_size # 286
        params = get_transform_params((load_size, load_size), crop_size, crop_size, self.opt.no_flip)
        #params = TransformParams(0, crop_size, 0, crop_size, False)
        #load_size = crop_size
        A_img = transform_image(A_img, (load_size, load_size), params)
        B_img = transform_image(B_img, (load_size, load_size), params)

        A = torch.from_numpy(A_img)
        B = torch.from_numpy(B_img)


        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        num_files = max(self.A_size, self.B_size)
        return num_files * 9 if self.opt.input_nc in [1, 3] else num_files