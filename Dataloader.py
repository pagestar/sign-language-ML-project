import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from Mediapipe import process_video_and_save

class Data(object):
    def __init__(self, video_path, save_path, idx, frame_size=(224, 224)):
        data, frame_count = process_video_and_save(video_path, save_path, frame_size)

        self.logits = data # todo: process frames with mediapipe
        self.frames_path = save_path
        self.label = idx
        self.count = frame_count

class FrameDataset(Dataset):
    def __init__(self, root_dir, frame_size=(224, 224), transform=None):
        """
        Args:
            root_dir: 數據根目錄 (如 pic_frame)。
            frame_size: 圖片的目標大小 (預處理尺寸)。
            transform: 數據增強技術。
        """
        self.root_dir = root_dir
        self.frame_size = frame_size
        self.transform = transform

        self.data = []
        self.labels = []
        self.counts = []
        self.frames_folders = []

        self.word2label = {}
        self.label2word = {}
        self._load_data()

    def get_frame_imgs(self, idx):
        imgs = []
        for pth in os.listdir(self.frames_folders[idx]):
            img_pth = os.path.join(self.frames_folders[idx], pth)
            img = cv2.imread(img_pth)
            img = img / np.max(img)
            img = np.rollaxis(img, 2, 00)
            imgs.append(img)
        return torch.tensor(np.array(imgs)).float()

    # multiple files
    # def get_frame_imgs(self, idxs):
    #     imgs_arr = []
    #     for idx in idxs:
    #         imgs = []
    #         for pth in os.listdir(self.frames_folders[idx]):
    #             img_pth = os.path.join(self.frames_folders[idx], pth)
    #             imgs.append(cv2.imread(img_pth))
    #         imgs_arr.append(img_pth)
    #     return torch.tensor(imgs_arr)


    def _load_data(self):
        # 掃描根目錄下的子文件夾（單字名稱）
        idx = 0
        for label, word in enumerate(os.listdir(self.root_dir)):
            word_path = os.path.join(self.root_dir, word)
            if os.path.isdir(word_path):
                # self.label_mapping[word] = idx  # 單字映射到標籤
                self.label2word[label] = word
                self.word2label[word] = label
                for id, video_file in enumerate(os.listdir(word_path)):
                    video_path = os.path.join(word_path, video_file)
                    vid = Data(video_path, '/'.join(('processed_data', f'{label}', f'{id}')), id, self.frame_size)
                    self.data.append(vid.logits)
                    self.frames_folders.append(vid.frames_path)
                    self.counts.append(vid.count)
                    self.labels.append(label)

        # zero padding
        # max_length = max(len(data) for data in self.data)
        # for i in range(len(self.data)):
        #   self.data[i] = self.data[i] + [torch.zeros(self.data[i][0].shape)] * (max_length - len(self.data[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            vid = self.transform(vid)
        return vid, torch.tensor(label, dtype=torch.long)

