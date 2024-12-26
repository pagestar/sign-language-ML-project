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
        idx = 0
        video_folders = os.listdir(self.root_dir)
        with tqdm.tqdm(total=len(video_folders), desc="Loading video data") as pbar:
            for label, word in enumerate(os.listdir(self.root_dir)):
                word_path = os.path.join(self.root_dir, word)
                if os.path.isdir(word_path):
                    self.label2word[label] = word
                    self.word2label[word] = label
                    for id, video_file in enumerate(os.listdir(word_path)):
                        video_path = os.path.join(word_path, video_file)
                        saved_path = os.path.join('processed_data', str(label), str(id))

                        # 這裡只在第一次處理時加載 processed_data 資料
                        if os.path.exists(saved_path):
                            logits = self._load_processed_data(saved_path)
                            frame_count = len(logits)  # 幀數
                        else:
                            vid = Data(video_path, saved_path, id, self.frame_size)
                            logits = vid.logits
                            frame_count = vid.count

                        self.data.append(logits)
                        self.frames_folders.append(saved_path)
                        self.counts.append(frame_count)
                        self.labels.append(label)
                        #print(f"Loaded label: {label} for word: {word}")
                    pbar.update(1)


        # zero padding
        # max_length = max(len(data) for data in self.data)
        # for i in range(len(self.data)):
        #   self.data[i] = self.data[i] + [torch.zeros(self.data[i][0].shape)] * (max_length - len(self.data[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = self._load_frames(self.frames_folders[idx])

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # 確保 frames 是 tensor
        frames = torch.stack([T.ToTensor()(frame) for frame in frames])  # (num_frames, channels, height, width)

        logits = self.data[idx]
        label = self.labels[idx]                

        # print("logits:", logits)
        # print("Length of logits:", len(logits))


        logits = torch.tensor(np.stack(logits), dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.long).unsqueeze(0)  # 包裝 label 為 batch_size=1 的張量

        #print(f"Frames shape: {frames.shape}, Label shape: {label.shape}")
        return frames, logits, label


    
    def _load_processed_data(self, processed_folder_path):
        logits = []
        
        if not os.path.exists(processed_folder_path):
            raise ValueError(f"{processed_folder_path} does not exist")

        # 遍歷該路徑下的所有幀資料
        for frame_file in sorted(os.listdir(processed_folder_path)):
            if frame_file.endswith('.jpg'):  # 確保是 .jpg 文件
                frame_path = os.path.join(processed_folder_path, frame_file)
                frame = cv2.imread(frame_path)  # 加載幀
                if frame is not None:
                    logits.append(frame)  # 加入幀到 logits 中
                else:
                    print(f"Warning: {frame_file} could not be loaded.")
        
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
    def __init__(self, rotation_range=10, scale_range=(0.8, 1.2), flip_prob=0.5, jitter_prob=0.2):
        self.rotation_range = rotation_range  # 旋轉範圍
        self.scale_range = scale_range  # 縮放範圍
        self.flip_prob = flip_prob  # 翻轉概率
        self.jitter_prob = jitter_prob  # 顏色調整概率

        # 定義增強操作
        self.transform = T.Compose([
            T.RandomRotation(degrees=self.rotation_range),  # 隨機旋轉
            # T.RandomHorizontalFlip(p=self.flip_prob),  # 隨機水平翻轉
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) if random.random() < self.jitter_prob else T.Compose([]),
            T.RandomResizedCrop(size=(224, 224), scale=self.scale_range)  # 隨機縮放裁剪
        ])
        self.augmentation_count = 0  # 紀錄已進行的增強次數

    def __call__(self, image):
        image = self.transform(image)
        self.augmentation_count += 1  # 每次進行增強時增加計數
        return image