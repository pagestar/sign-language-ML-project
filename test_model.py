import torch
from Dataloader import FrameDataset
from main import eval_model, print_result
import matplotlib.pyplot as plt

model = torch.load('model.pth')

model.eval()

# Test the model on a sample input  
dataset = FrameDataset(root_dir="data")

print("開始評估模型...")
labels, features = eval_model(model, dataset)
print("評估結果：")
print("特徵向量維度：", features.shape)
#print("模型準確率：", (labels == labels.max(1)[1]).sum().item() / labels.shape[0])
#print("Augmentation count:", transform.augmentation_count)
print_result(labels, dataset)

# 6. 繪製訓練過程圖
'''
plt.plot(loss_list[1:])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Hidden dim: {hidden_dim}, Learning rate: {learning_rate}, layers: {num_layers}")
plt.show()
'''