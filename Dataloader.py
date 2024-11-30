import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class Data(object):
    def __init__(self, video_path, idx, frame_size=(224, 224)):
        vid = cv2.VideoCapture(video_path)
        frames = []
        while(vid.isOpened()):
            # Capture frame-by-frame
            ret, frame = vid.read()
            if ret == True:
                # Display the resulting frame
                frame = cv2.resize(frame, frame_size) # 調整圖片大小
                data = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0  # 標準化 [0, 1]
                frames.append(data)
            else:
                break
        vid.release()
        self.logits = ... # todo: process frames with mediapipe
        self.frames = frames
        self.label = idx

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
        self.label_mapping = {}
        self._load_data()

    def _load_data(self):
        # 掃描根目錄下的子文件夾（單字名稱）
        for idx, word in enumerate(os.listdir(self.root_dir)):
            word_path = os.path.join(self.root_dir, word)
            if os.path.isdir(word_path):
                # self.label_mapping[word] = idx  # 單字映射到標籤
                self.label_mapping[idx] = word
                for video_file in os.listdir(word_path):
                    video_path = os.path.join(word_path, video_file)
                    vid = Data(video_path, idx, self.frame_size)
                    self.data.append(vid.frames)
                    self.labels.append(idx)

        max_length = max(len(data) for data in self.data)
        for i in range(len(self.data)):
          self.data[i] = self.data[i] + [torch.zeros(self.data[i][0].shape)] * (max_length - len(self.data[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            vid = self.transform(vid)
        return vid, torch.tensor(label, dtype=torch.long)
