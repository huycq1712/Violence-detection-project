import os
import time
import logging
import datetime
from tqdm import tqdm

import torch
from modeling.vionet import build_vionet

def inference(
    model, 
    dataloader,
    device,
    arguments):
    logger = logging.getLogger("Violence-detection-project.engine.inference")
    logger.info("Start inference")
    max_iter = arguments["max_iter"]
    iteration = arguments["iteration"]
    accuracys = []
    model.eval()
    start_inference_time = time.time()
    end = time.time()

    with torch.no_grad():
        for step, (frames, opticals, _, labels) in enumerate(tqdm(dataloader)):
            iteration = iteration + 1
            arguments['iteration'] = iteration
            data_time = time.time() - end
            #inner_iter = step

            frames = frames.to(device)
            opticals = opticals.to(device)
            labels = labels.to(device)
            #masks = masks.to(device)

            output = model(frames, opticals, is_train=False)
            #loss = criterion_classify(output[0], labels) + critertion_seg(output[1], masks)
            accuracy = (output.argmax(dim=1) == labels).float().mean()
            accuracys.append(accuracy)


            batch_time = time.time() - end
            end = time.time()

    print("ACCURACY: ", sum(accuracys)/len(accuracys))
    

    
