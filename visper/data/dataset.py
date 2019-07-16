import csv
import os
from os.path import join

import gin
import torch
from torch.utils.data import Dataset

from .utils import read_video, preprocess

@gin.configurable(blacklist=["phase"])
class LipreadingDataset(Dataset):
    def __init__(self, phase, directory=None, labels=None):
        self.LABELS = labels
        num_labels = len(labels)

        self.label_list, self.file_list = self.build_file_list(directory, phase, num_labels)

    def build_file_list(self, directory, phase, num_labels):
        labels = self.LABELS
        completeList = []

        for i, label in enumerate(labels):
            dirpath = directory + "/{}/{}".format(label, phase)

            files = os.listdir(dirpath)

            for file in files:
                if file.endswith("mp4"):
                    filepath = dirpath + "/{}".format(file)
                    entry = (i, filepath)
                    completeList.append(entry)

        return labels, completeList

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # load video into a tensor
        label, filename = self.file_list[idx]
        vidframes = read_video(filename)
        if not vidframes:
            print(f"broken video - {filename}")

        temporalvolume = preprocess(vidframes)

        sample = {'features': temporalvolume, 'targets': torch.tensor(label, dtype=torch.long)}

        return sample


if __name__ == "__main__":
    dtst = LipreadingDataset("/home/dmitry.klimenkov/Documents/datasets/lipreading_datasets/LIPS/LRW_gray_122")

    from torch.utils.data import DataLoader
    from torch.autograd import Variable
    ldr = DataLoader(dtst, batch_size=7, shuffle=True)

    for i_batch, sample_batched in enumerate(ldr):
        inputt = Variable(sample_batched['temporalvolume'])
        labels = Variable(sample_batched['label'])

        #print(inputt.size())
        #print(labels.size())

        #print(labels)

        break