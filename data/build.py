from torch.utils import data


from .datasets.rwf2000 import RWF2000
from .datasets.rlvs import RLVS
from .transforms import build_transforms
from .collate_batch import collate_fn


def build_dataset(dataset_name, transforms=None, is_train=True):
    if dataset_name == 'rwf-2000':
        dataset = RWF2000(transforms=transforms, is_train=is_train)
    
    if dataset_name == 'rlvs':
        dataset = RLVS(transforms=transforms, is_train=is_train)

    return dataset



def make_data_loader(cfg=None, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset('rwf-2000',transforms=transforms, is_train=is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn
    )

    return data_loader


if __name__ == '__main__':
    data = make_data_loader('rwf-2000')

    for i, batch in enumerate(data):
        print(batch[2].shape)
        break