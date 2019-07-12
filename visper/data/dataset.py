import csv
import os
from os.path import join

from torch.utils.data import Dataset

from .preprocess import *

class LipreadingDataset(Dataset):
    """BBC Lip Reading dataset."""

    def get_labels_v2(self, num_labels=5):
        return ["ABOUT", "PEOPLE", "ACTUALLY", "BECAUSE", "FIRST"]
    

    def get_labels_lrs2_w(self, directory):
        labels = os.listdir(directory)

        return labels


    def build_file_list(self, directory, phase, num_labels):
        labels = self.get_labels_v2() #self.get_labels(num_labels=num_labels) # self.get_labels_lrs2_w(directory) # 
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


    def __init__(self, directory, phase, num_labels=20, augment=True):
        self.label_list, self.file_list = self.build_file_list(directory, phase, num_labels)
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # load video into a tensor
        label, filename = self.file_list[idx]
        vidframes = load_video(filename)
        if not vidframes:
            print(f"broken video - {filename}")
        # print(filename)
        temporalvolume = bbc(vidframes, self.augment)

        sample = {'temporalvolume': temporalvolume, 'label': torch.LongTensor([label])}

        return sample


if __name__ == "__main__":
    dtst = LipreadingDataset("/home/dmitry.klimenkov/Documents/datasets/lipreading_datasets/mixed_dataset",
                            "train")

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