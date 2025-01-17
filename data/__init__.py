"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataloader(opt):
    dataset = find_dataset_using_name(opt['dataset_name'])
    instance = dataset(opt)
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    dataloader = torch.utils.data.DataLoader(instance,
                                             batch_size=opt['batch_size'],
                                             shuffle=True,
                                             num_workers=int(opt['workers']),
                                             drop_last=opt['isTrain'],
                                             pin_memory=True)
    return dataloader

# used for distributed data parallel
def create_dataloader2(opt):
    dataset = find_dataset_using_name(opt['dataset_name'])
    instance = dataset(opt)
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    
    sampler = torch.utils.data.distributed.DistributedSampler(instance,num_replicas=len(opt['gpu_ids']),rank=opt['local_rank'])
    dataloader = torch.utils.data.DataLoader(instance,
                                             batch_size=opt['batch_size'],
                                             shuffle=True,
                                             num_workers=int(opt['workers']),
                                             drop_last=opt['isTrain'],
                                             pin_memory=True,
                                             sampler=sampler)
    return dataloader
