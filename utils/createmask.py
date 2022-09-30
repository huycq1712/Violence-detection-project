from people_segmentation.pre_trained_models import create_model

import os
from tqdm import tqdm
import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image

model = create_model("Unet_2020-07-20")
model.eval()

def uniform_sample(video, optical_flow, n_frames):
    frames, opticals = [], []
    interval = int(np.ceil(len(video)/n_frames))
    background = (np.sum(video, axis=0)/len(video)).astype(np.uint8)

    for i in range(0, len(video), interval):
        frames.append(np.abs(video[i]-background))
        opticals.append(optical_flow[i])

    if n_frames > len(frames):
        frames = frames + [video[i] for i in range(len(frames) - n_frames, 0)]
        opticals = opticals + [optical_flow[i] for i in range(len(opticals) - n_frames,0)]

    return frames, opticals # num_frames, h, w, c

def get_items(file_path, n_frames):
    data = np.load(file_path, mmap_mode='r')
    video = data[..., :3]
    optical_flow = data[..., 3:]
    #print(np.shape(optical_flow))
    # padding = np.zeros((optical_flow.shape[0], optical_flow.shape[1], optical_flow.shape[2], 1))
    # optical_flow = np.concatenate((optical_flow, padding), axis=-1)
    frames, opticals = uniform_sample(video, optical_flow, n_frames) 
    
    return frames, opticals    

def predict_human_masks(file_path, n_frames, model):
    """Returns a list of masks of the human in the video.
    Args:
        tensor_frames (tensor): List of frames from the video.
    Returns:
        list: List of masks of the human in the video.
    """
    masks, new_frames = [], []
    frames, opticals = get_items(file_path, n_frames)
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    
    for frame in frames:
        padded_image, pads = pad(frame, factor=32, border=cv2.BORDER_CONSTANT)
        x = transform(image=padded_image)["image"]
        x = torch.squeeze(tensor_from_rgb_image(x), 0)
        new_frames.append(x)

    new_frames = torch.stack(new_frames).to('cuda')
    with torch.no_grad():
        predictions = model(new_frames)

    masks = [(torch.squeeze(prediction) > 0).cpu().numpy().astype(np.uint8) for prediction in predictions]
    masks = np.expand_dims(masks, axis=-1)
    item = np.concatenate((frames, opticals, masks), axis=-1)
    return item # frames, opticals, masks

def save_items2Npy(dir, model, n_frames, output_path):
    """Returns a list of masks of the human in the video.
    Args:
        dir (str): Path to the directory.
    Returns:
        list: List of masks of the human in the video.
    """
    # get video path from dir
    train_test = ['val']
    for tt in train_test:
        # get video path from dir
        tt_path = os.path.join(dir, tt)
        # create output tt file
        output_tt_path = os.path.join(output_path, tt)
        if not os.path.exists(output_tt_path):
            os.mkdir(output_tt_path)

        kinds = os.listdir(tt_path)
        for kind in kinds:
            # get video path from dir
            kind_path = os.path.join(tt_path, kind)
            # create output kinds file
            output_kind_path = os.path.join(output_tt_path, kind)
            if not os.path.exists(output_kind_path):
                os.mkdir(output_kind_path)

            videos = os.listdir(kind_path)
            for video_name in tqdm(videos):
                # get video path
                video_path = os.path.join(kind_path, video_name)
                # get frames, masks, opticals
                item = predict_human_masks(video_path, n_frames, model) # frames, masks, opticals
                #print(np.shape(item))
                
                # save item to .npy
                np.save(os.path.join(output_kind_path, video_name.split('.')[0]+'.npy'), item)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ROOT_DIR = '/home/huycq/Violence/Violence-detection-project/datasets/RWF-2000-npy'
    n_frames = 24
    model = model.to('cuda')
    output_path = '/home/huycq/Violence/Violence-detection-project/datasets/RWF-2000-npy-seg'

    save_items2Npy(ROOT_DIR, model, n_frames, output_path)