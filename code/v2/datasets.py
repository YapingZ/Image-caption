import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np


def id2npz(id):
    feature_folder = '/home/leonard/jinfei/code/self-critical.pytorch/data2/cocobu_att/'
    file_name = str(id) + '.npz'
    return np.load(os.path.join(feature_folder,file_name))

def id2npy(id):
    feature_folder = '/home/leonard/jinfei/code/self-critical.pytorch/data2/cocobu_fc/'
    file_name = str(id) + '.npy'
    return np.load(os.path.join(feature_folder,file_name))
    

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """

        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

  

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' +  '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_cocoid_' + '.json'), 'r') as j:
            self.coco_id = json.load(j)
       


        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)
        self.cpi = 5       

 
    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img_id = self.coco_id[i // self.cpi]
       
        img_att = id2npz(img_id)
        img_att = img_att['feat']
        img_att = torch.FloatTensor(img_att)
        #print(img_att.size())
        img_fc = id2npy(img_id)
        #print(img_fc.shape)
        img_fc = torch.FloatTensor(img_fc.reshape(-1,2048))
        #print(img_fc.size())

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img_att, img_fc,caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img_att, img_fc,caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
