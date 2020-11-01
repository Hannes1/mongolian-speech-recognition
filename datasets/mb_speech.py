"""Mongolian Bible dataset."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import csv
import numpy as np

from torch.utils.data import Dataset

#Going to have to create my own vocab for english
vocab = "B абвгдеёжзийклмноөпрстуүфхцчшъыьэюя"  # B: blank  #vocab gets imported seperatly
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}


#here we return the audios text used ensuring only lowercase characters and valid characters
def convert_text(text):
    text = text.lower()
    # ignore all characters which is not in the vocabulary
    return [char2idx[char] for char in text if char != 'B' and char in char2idx]


def read_metadata(dataset_path, max_duration):
    fnames, texts = [], []

    reader = csv.reader(open(os.path.join(dataset_path, 'metadata.csv'), 'rt'), delimiter='|')
    for line in reader:
        fname, duration, text = line[0], line[1], line[2]
        try:
            duration = float(duration)
            if duration > max_duration:
                continue
            if duration < 1.5:
                continue
        except ValueError:
            continue

        fnames.append(os.path.join(dataset_path, 'wavs', '%s.wav' % fname)) #get path to audio wav
        texts.append(np.array(convert_text(text))) #get text to audio

    return fnames, texts

#gets called and reads to csv file and return the text and file path
class MBSpeech(Dataset):

    def __init__(self, max_duration=16.7, transform=None):
        self.transform = transform

        datasets_path = os.path.dirname(os.path.realpath(__file__))
        dataset_path = os.path.join(datasets_path, 'MBSpeech-1.0')
        self.fnames, self.texts = read_metadata(dataset_path, max_duration)

    def __getitem__(self, index):
        data = {
            'fname': self.fnames[index],
            'text': self.texts[index]
        }

        if self.transform is not None:
            data = self.transform(data) #tranform is passed as param in train.py 

        return data

    def __len__(self):
        return len(self.fnames)
