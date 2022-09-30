import os
import glob
import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import cv2


class RLVS(data.Dataset):
    """
    Dataset for RLVS
    """
    def __init__(self, root='./datasets',
                 n_frames=10,
                 transforms=None,
                 is_train=True) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.is_train = is_train
        self.n_frames = n_frames

        # get all video from video dir
        self.video_dir = os.path.join(self.root, 'rwf_2000')
        self.list_video_dir = glob.glob(self.video_dir + '/*/*.avi')


    def __getitem__(self, index):
        file_path = self.list_video[index].replace('.avi', '.npy')
        label = [1] if file_path.split['/'][-2] == 'Fight' else [0]
        label = torch.tensor(label)

        data = np.load(file_path, mmap_mode='r')
        video = np.float32(data[...,:3])
        optical_flow = np.float32(data[...,3:])

        frames, opticals = self.uniform_sample(video, optical_flow)

        if self.transforms is not None:
            frames = self.transforms(frames)

        return frames, opticals, label


    def uniform_sample(self, video, optical_flow):
        frames = []
        opticals = []
        interval = int(np.ceil(len(video)/self.n_frames))
        
        for i in range(len(video, interval)):
            frames.append(video[i])
            opticals.append(optical_flow[i])
        
        if self.n_frames > len(video):
            frames = frames + [video[i] for i in range(len(video) - self.n_frames,0)]
            opticals = opticals + [optical_flow[i] for i in range(len(optical_flow) - self.n_frames,0)]

        frames = [torch.from_numpy(frame).float() for frame in frames]
        opticals = [torch.from_numpy(optical).float() for optical in opticals]

        return torch.stack(frames), torch.stack(opticals)


    @staticmethod
    def keyframes_sample(video, num_frames):
        pass



if __name__ == "__main__":
    print("list dir")
    
