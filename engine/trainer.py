import os
import time
import logging
import datetime
from tqdm import tqdm
import  numpy as np

import torch
import torch.nn as nn

from modeling.vionet import build_vionet
from torchvision.transforms import functional as F

from PIL import Image

def f1_score(pred, target):
    #print(pred.shape, target.shape)
    #pred = pred.argmax(dim=1)
    #target = target.argmax(dim=1)
    
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def confusion_matrix(pred, target):
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()
    tn = ((1 - target) * (1 - pred)).sum()
    return (tp, fp, fn, tn)

def evaluate(
    model, 
    dataloader,
    device,
    arguments
):
    logger = logging.getLogger("Violence-Detection.engine.trainer")
    model.eval()
    print(model)
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
    logger.info("Epoch: {}, Accuracy: {}".format(arguments['epoch'], accuracy_test))

    labels_ = np.concatenate(labels_)
    predict_ = np.concatenate(predict_)

    f1 = confusion_matrix(torch.tensor(predict_), torch.tensor(labels_))
    logger.info("Epoch: {}, F1: {}".format(arguments['epoch'], f1))


def do_train(
    model, 
    dataloader, 
    test_dataloader,
    loss,
    #criterion_classify,
    #critertion_seg,
    optimizer,
    scheduler,
    checkpointer,
    meters,
    device,
    checkpoint_period,
    arguments):
    logger = logging.getLogger("Violence-Detection.engine.trainer")
    logger.info("Start training")
    max_epoch = arguments["max_epoch"]
    epoch = arguments["epoch"]
    max_iter = max_epoch * len(dataloader)
    iteration = arguments["iteration"]

    
    start_training_time = time.time()
    end = time.time()

    #for epoch in tqdm(range(0, max_epoch)):
    
    while epoch < max_epoch:
        model.train()

        epoch = epoch + 1
        print('Epoch: {}'.format(epoch))
        arguments['epoch'] = epoch
        scheduler.step()

        loss_epoch = []
        acc_epoch = []

        for step, (frames, opticals, masks, labels) in enumerate(dataloader):
            
            iteration = iteration + 1
            arguments['iteration'] = iteration
            data_time = time.time() - end
            #inner_iter = step

            frames = frames.to(device)
            opticals = opticals.to(device)
            labels = labels.to(device)
            masks = masks.to(device).float()

            output = model(frames, opticals, is_train=True)
            #loss_cls = criterion_classify(output[0], labels)
    
            masks = masks.contiguous().view(-1, *masks.shape[2:])
            masks = F.resize(masks, (output[1].shape[2], output[1].shape[3]), Image.BILINEAR)
            
            #loss_seg = critertion_seg(output[1], masks)

            losses =  loss(output[0], output[1], labels, masks)
            #print(output[1].shape)
            accuracy = (output[0].argmax(dim=1) == labels).float().mean()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            #print(loss)
            meters.update(time=batch_time, data=data_time)
            #meters.update(loss_cls = loss_cls.item())
            #meters.update(loss_seg = loss_seg.item())
            meters.update(loss = losses.item())
            meters.update(accuracy = accuracy.item())

            loss_epoch.append(losses.item())
            acc_epoch.append(accuracy)


            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if step % 1 == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch [{epoch}][{inner_iter}/{num_iter}]",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        inner_iter=step,
                        num_iter=len(dataloader),
                        meters=str(meters),
                        lr=optimizer.param_groups[-1]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if epoch % 1 == 0:
                checkpointer.save("model_{:07d}".format(epoch), **arguments)
            if iteration == max_iter:
                break

        
        loss_epoch = sum(loss_epoch) / len(loss_epoch)
        acc_epoch = sum(acc_epoch)/len(acc_epoch)

        #print("LOSS: ",loss_epoch)
        logger.info("Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, loss_epoch, acc_epoch))
        evaluate(model, test_dataloader, device, arguments)
        

    