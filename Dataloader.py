import os
import cv2
import torch
import numpy as np
import tqdm
import random
import torchvision.transforms as T
from torch.utils.data import Dataset
from Mediapipe import process_video_and_save
from PIL import Image

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
        video_folders = os.listdir(self.root_dir)
        with tqdm.tqdm(total=len(video_folders), desc="Loading video data") as pbar:
            for label, word in enumerate(os.listdir(self.root_dir)):
                word_path = os.path.join(self.root_dir, word)
                if os.path.isdir(word_path):
                    # self.label_mapping[word] = idx  # 單字映射到標籤
                    self.label2word[label] = word
                    self.word2label[word] = label
                    for id, video_file in enumerate(os.listdir(word_path)):
                        video_path = os.path.join(word_path, video_file)
                        saved_path = '/'.join(('processed_data', f'{label}', f'{id}'))

                        if os.path.exists("processed_data"):
                            logits = self._load_processed_data("processed_data")
                            frame_count = len(os.listdir("processed_data"))
                        else:
                            vid = Data(video_path, saved_path, id, self.frame_size)
                            logits = vid.logits
                            frame_count = vid.count
                            
                        self.data.append(logits)
                        self.frames_folders.append(saved_path)
                        self.counts.append(frame_count)
                        self.labels.append(label)
                pbar.update(1)

        # zero padding
        # max_length = max(len(data) for data in self.data)
        # for i in range(len(self.data)):
        #   self.data[i] = self.data[i] + [torch.zeros(self.data[i][0].shape)] * (max_length - len(self.data[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid = self.data[idx]
        label = self.labels[idx]
        frames = self._load_frames(self.frames_folders[idx])
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        return vid, torch.tensor(label, dtype=torch.long)
    
    def _load_processed_data(self, processed_folder_path = "processed_data"):
        logits = []
        for frame_file in sorted(os.listdir(processed_folder_path)):
            frame_path = os.path.join(processed_folder_path, frame_file)
            if frame_file.endswith('.npy'):
                data = np.load(frame_path)
            else:
                data = cv2.imread(frame_path)
            logits.append(data)
        return logits
    
    def _load_frames(self, video_path):
        # 根據影片路徑讀取每一幀（假設每幀是單獨的圖片）
        frames = []
        for frame_file in sorted(os.listdir(video_path)):
            frame_path = os.path.join(video_path, frame_file)
            frame = Image.open(frame_path)
            frames.append(frame)
        return frames

class VideoDataAugmentation:
    def __init__(self, rotation_range=30, scale_range=(0.8, 1.2), flip_prob=0.5, jitter_prob=0.2):
        self.rotation_range = rotation_range  # 旋轉範圍
        self.scale_range = scale_range  # 縮放範圍
        self.flip_prob = flip_prob  # 翻轉概率
        self.jitter_prob = jitter_prob  # 顏色調整概率

        # 定義增強操作
        self.transform = T.Compose([
            T.RandomRotation(degrees=self.rotation_range),  # 隨機旋轉
            T.RandomHorizontalFlip(p=self.flip_prob),  # 隨機水平翻轉
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) if random.random() < self.jitter_prob else T.Compose([]),
            T.RandomResizedCrop(size=(224, 224), scale=self.scale_range)  # 隨機縮放裁剪
        ])
        self.augmentation_count = 0  # 紀錄已進行的增強次數

    def __call__(self, image):
        image = self.transform(image)
        self.augmentation_count += 1  # 每次進行增強時增加計數
        return image