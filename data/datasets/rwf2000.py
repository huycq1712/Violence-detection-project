import os
import glob
import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import cv2


class RWF2000(data.Dataset):
    """
    Dataset for RWF2000
    """
    def __init__(self, root='/home/huycq/Violence/Violence-detection-project/datasets/RWF-2000-npy-seg',
                 n_frames=24,
                 transforms=None,
                 is_train=True) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.is_train = is_train
        self.n_frames = n_frames
        #self.root_npy = self.root + '-npy'

        # get all video from video dir
        #self.video_dir = os.path.join(self.root, 'RWF-2000')
        if is_train:
            self.video_dir = os.path.join(self.root, 'train')
        else:
            self.video_dir = os.path.join(self.root, 'val')

        #print(self.video_dir)
        self.list_video_dir = glob.glob(self.video_dir + '/*/*.npy')

        #print(self.list_video_dir + '/train/Fight/*.avi') 

        print('Number of videos: ', len(self.list_video_dir))

    
    def __len__(self):
        return len(self.list_video_dir)


    def __getitem__(self, index):
        file_path = self.list_video_dir[index].replace('.avi', '.npy')
        #print()
        labels = [1] if file_path.split('/')[-2] == 'Fight' else [0]
        labels = torch.tensor(labels)

        data = np.load(file_path, mmap_mode='r')
        #video = np.float64(data[...,:3])
        #optical_flow = np.float64(data[...,3:])
        frames = data[...,:3]
        opticals = data[...,3:5]
        masks = data[...,5:]
        #print(masks.shape)
        #print(optical_flow.shape)
        padding = np.zeros((opticals.shape[0], opticals.shape[1], opticals.shape[2], 1))
        opticals = np.concatenate((opticals, padding), axis=-1)

        frames, opticals, masks = self.uniform_sample(frames, opticals, masks)

        frames = frames.permute(0, 3, 1, 2)
        opticals = opticals.permute(0, 3, 1, 2)
        masks = masks.permute(0, 3, 1, 2)

        if self.transforms is not None:
           frames, masks = self.transforms[0](frames, masks)
           opticals, _ = self.transforms[1](opticals, None)

        
        
        return frames, opticals, masks, labels


    def uniform_sample(self, frames, opticals, masks):
        #print("len video",len(video))
        """frames = []
        opticals = []
        interval = int(np.ceil(len(video)/self.n_frames))
        background = (np.sum(video, axis=0)/len(video)).astype(np.uint8)

        #print(background.shape)

        for i in range(0, len(video), interval):
            frames.append(np.abs(video[i]-background))
            opticals.append(optical_flow[i])
        
        if self.n_frames > len(frames):
            frames = frames + [video[i] for i in range(len(frames) - self.n_frames,0)]
            opticals = opticals + [optical_flow[i] for i in range(len(opticals) - self.n_frames,0)]"""

        frames = [torch.from_numpy(frame) for frame in frames]
        opticals = [torch.from_numpy(optical) for optical in opticals]
        masks = [torch.from_numpy(mask) for mask in masks]

        return torch.stack(frames), torch.stack(opticals), torch.stack(masks) # num_frames, h,w, c


    @staticmethod
    def keyframes_sample(video, num_frames):
        pass



if __name__ == "__main__":
    a = RWF2000()

    print(a[0][0].shape)
    #print(glob.glob( '...../Violence-detection-project/datasets/RWF-2000/*/Fight/*.avi'))
