import cv2
import mediapipe as mp
import os

# 初始化 MediaPipe 手部辨識與姿勢偵測
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 設定 MediaPipe 偵測參數
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def create_folder(folder_path):
    """
    建立存檔資料夾，若不存在則創建。
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def process_video_and_save(video_path, save_path, frame_size = (224, 224)):
    """
    讀取影片並保存每幀偵測到關節點的影像與 MediaPipe 數據。

    Args:
        video_path: 輸入影片路徑。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法讀取影片，請確認路徑正確：{video_path}")
        return

    # 建立輸出資料夾
    # video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_folder = save_path
    create_folder(save_folder)

    mediapipe_data = []  # 存放 MediaPipe 數據
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("影片播放完畢或無法讀取影格")
            break

        # 將影像轉為 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 偵測手部與上半身關鍵點
        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        # 若偵測到手部或上半身，標記並保存影像
        frame_data = []
        frame_paths = []
        if hand_results.multi_hand_landmarks or pose_results.pose_landmarks:
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for landmark in hand_landmarks.landmark:
                        frame_data.append([landmark.x, landmark.y, landmark.z])

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for landmark in pose_results.pose_landmarks.landmark:
                    frame_data.append([landmark.x, landmark.y, landmark.z])

            # 保存圖片
            # save_img_path = os.path.join(save_folder, f"frame_{frame_id:04d}.jpg")
            save_img_path = '/'.join((save_folder, f"frame_{frame_id:04d}.jpg"))
            if (not cv2.imwrite(save_img_path, cv2.resize(frame, frame_size))):
                print(f'save image {save_img_path} failed')
                break
            frame_paths.append(save_img_path)

            # 保存 MediaPipe 數據
            mediapipe_data.append(frame_data)

        frame_id += 1

    cap.release()

    # 返回 MediaPipe 數據與圖片數量
    return mediapipe_data, frame_id
