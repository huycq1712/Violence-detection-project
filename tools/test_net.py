import argparse
import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from config import cfg
from utils.logger import setup_logger
from modeling.vionet import build_vionet
from utils.checkpoint import Checkpointer
from utils.directory import makedir
from data import make_data_loader
from engine.inference import inference
from utils.comm import synchronize, get_rank
import logging
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="PyTorch Image-Text Matching Inference")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--checkpoint-file",
        default="",
        metavar="FILE",
        help="path to checkpoint file",
        type=str,
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--load-result",
        help="Use saved reslut as prediction",
        action='store_true',
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("PersonSearch", save_dir)
    #logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    model = build_vionet(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = Checkpointer(model, save_dir=output_dir, logger=logger)
    _ = checkpointer.load(args.checkpoint_file)

    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            makedir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False)
    """for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder
        )"""

    evaluate(model, data_loaders_val, cfg.MODEL.DEVICE, {'epoch': 0})

def evaluate(
    model, 
    dataloader,
    device,
    arguments
):
    logger = logging.getLogger("Violence-Detection.engine.trainer")
    model.eval()
    accuracy_test = []
    f1_test = []
    predict_ = []
    labels_ = []
    for step, (frames, opticals, masks, labels) in enumerate(tqdm(dataloader)):
        frames = frames.to(device)
        opticals = opticals.to(device)
        labels = labels.to(device)
        output = model(frames, opticals, is_train=False)

        accuracy = (output[0].argmax(dim=1) == labels).float().mean()
        accuracy_test.append(accuracy)

        labels_.append(labels.cpu().numpy())
        predict_.append(output[0].argmax(dim=1).cpu().numpy())
        

    accuracy_test = sum(accuracy_test)/len(accuracy_test)
    logger.info("Test, Accuracy: {}".format(accuracy_test))

    labels_ = np.concatenate(labels_)
    predict_ = np.concatenate(predict_)

    f1 = f1_score(torch.tensor(predict_), torch.tensor(labels_))
    logger.info("Test, F1: {}".format(f1))


def f1_score(predict, label):
    TP = ((predict == 1) & (label == 1)).sum().item()
    FP = ((predict == 1) & (label == 0)).sum().item()
    FN = ((predict == 0) & (label == 1)).sum().item()
    TN = ((predict == 0) & (label == 0)).sum().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return f1

if __name__ == "__main__":
    main()