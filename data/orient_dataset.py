from PIL import Image
import os
from data.base_dataset import BaseDataset
import haya_data


class OrientDataset(BaseDataset):
    def __init__(self, opt):
        super(OrientDataset, self).__init__()
        if opt['mode'] == 'train':
            self.inner_ds = haya_data.HairOrientData()
        else:
            self.inner_ds = haya_data.HairOrientData(test=True)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __getitem__(self, index):
        return self.inner_ds[index]

    def __len__(self):
        return len(self.inner_ds)