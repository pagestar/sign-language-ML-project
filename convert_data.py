import os
import cv2
import mediapipe as mp
import numpy as np
import time
import torch

# 初始化 MediaPipe 手部辨識和姿勢偵測
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 設定手部辨識和姿勢偵測的參數
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5)

def process(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法讀取影片，請確認路徑正確：{video_path}")
        return None

    all_frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\nDone.\n")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        # 只有當偵測到手部關鍵點時才記錄該幀數據
        if hand_results.multi_hand_landmarks:
            frame_data = []

            # 繪製手部關節並儲存座標
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for landmark in hand_landmarks.landmark:
                    frame_data.extend([landmark.x, landmark.y, landmark.z])

            # 繪製上半身骨架（排除臉部）並儲存座標
            if pose_results.pose_landmarks:
                upper_body_indices = list(range(11, 23))
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for i in upper_body_indices:
                    landmark = pose_results.pose_landmarks.landmark[i]
                    frame_data.extend([landmark.x, landmark.y, landmark.z])

            all_frames_data.append(frame_data)

    cap.release()
    cv2.destroyAllWindows()

    # 找出每幀的最大長度並填充
    max_length = max(len(data) for data in all_frames_data)
    all_frames_data_padded = [np.pad(data, (0, max_length - len(data)), 'constant') for data in all_frames_data]
    all_frames_matrix = np.array(all_frames_data_padded)
    print("Shape of matrix: ", all_frames_matrix.shape)

    # 將矩陣轉換為 RNN 需要的張量格式
    rnn_input = torch.tensor(all_frames_matrix, dtype=torch.float32).unsqueeze(1)  # (frame_num, 1, features)

    # 儲存到文件
    np.savetxt('output_matrix.txt', all_frames_matrix, fmt='%f')
    print(f"RNN Input shape: {rnn_input.shape}")
    
    return rnn_input

def process_videos(video_paths):
    rnn_inputs = []

    for video_path in video_paths:
        # 處理每個影片並返回 RNN 張量
        rnn_input = process(video_path)
        if rnn_input is not None:
            rnn_inputs.append(rnn_input)

    return rnn_inputs
