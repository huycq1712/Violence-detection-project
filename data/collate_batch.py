import torch


def collate_fn(batch):
    transposed_batch = list(zip(*batch))
    frames = torch.stack(transposed_batch[0])
    opticals = torch.stack(transposed_batch[1])
    
    masks = torch.stack(transposed_batch[2])
    labels = torch.cat(transposed_batch[3])
    return frames, opticals, masks, labels 
