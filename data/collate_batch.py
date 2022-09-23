import torch


def collate_fn(batch):
    transposed_batch = list(zip(*batch))
    frames = torch.stack(transposed_batch[0])
    opticals = torch.stack(transposed_batch[1])
    labels = transposed_batch[2]
    return frames, opticals, labels
