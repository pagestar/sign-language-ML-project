import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from Dataloader import FrameDataset, VideoDataAugmentation
from GestureModel import GestureModel, eval_model, train_model
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from CNN import FrameCNN, train_cnn, eval_cnn, extract_and_combine_features, test_CNN


'''
Print the predicted and actual labels for each video for GestureModel.
'''
def print_result(labels, dataset):
    labels = labels.cpu().numpy()
    for i in range(len(labels)):
        print(f"Video {i+1}: Predicted = {dataset.label2word[labels[i]]}, Actual = {dataset.label2word[dataset.labels[i]]}")

'''
For dataloader, we need to define a customized collate_fn.
'''
def collate_fn(batch):
    frames = [item[0] for item in batch]
    logits = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # 填充 frames，使每個樣本的長度一致
    frames = pad_sequence(frames, batch_first=True, padding_value=0)

    # 如果 logits 的維度不一致，您也可以考慮對 logits 進行填充或處理
    logits = pad_sequence(logits, batch_first=True, padding_value=0)  # 如果 logits 是序列數據，可以用這行
    # 如果 logits 只是單一的標籤，則可以直接堆疊
    # logits = torch.stack(logits)

    labels = torch.tensor(labels, dtype=torch.long)
    
    return frames, logits, labels


def main():

    root_dir = "data"  
    num_classes = 10
    feature_dim = 128

    '''
    You can adjust the hyperparameters here.
    '''
    hidden_dim = 128
    num_layers = 2
    learning_rate = 5e-4
    epochs = 100

    transform = VideoDataAugmentation()

    '''
    Dataset & Model 
    Maybe something is wrong for dataloader
    '''
    dataset = FrameDataset(root_dir=root_dir, transform=transform)  
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)  

    cnn = FrameCNN(feature_dim=128).cuda()  
    rnn = GestureModel(input_size=128 + 3 * 32 * 32, 
                    hidden_size=hidden_dim, num_classes=num_classes, num_layers=num_layers).cuda()

    '''
    Optimizers & Loss function
    '''
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    '''
    Training CNN
    '''
    print("Start training CNN...")
    _, loss_list = train_cnn(cnn, cnn_optimizer, criterion, dataset, num_epochs=epochs)
    #eval_cnn(cnn, dataset)
    test_CNN(cnn, dataset)
    torch.save(cnn.state_dict(), "trained_cnn.pth")  # Save the model parameters
    # Plot the training loss of CNN
    plt.plot(loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Learning rate: {learning_rate}")
    plt.show()

    '''
    Extract feature maps and combine them with frames.
    Maybe something is wrong in this step.
    '''
    combined_features = extract_and_combine_features(cnn, dataloader)

    '''
    Training RNN
    '''
    labels = torch.tensor(dataset.labels, dtype=torch.long) 
    rnn_dataset = TensorDataset(combined_features, labels)
    rnn_loader = DataLoader(rnn_dataset, batch_size=16, shuffle=True)
    print("Start training RNN...")
    train_model(rnn, criterion, rnn_optimizer, rnn_loader, epochs=epochs)

    '''
    Evaluate the GestureModel
    '''
    print("Evaluate the GestureModel...")
    labels = eval_model(rnn, rnn_loader)
    print("Results:")
    #print("特徵向量維度：", features.shape)
    #print("模型準確率：", (labels == labels.max(1)[1]).sum().item() / labels.shape[0])
    print("Augmentation count:", transform.augmentation_count)
    print_result(labels, dataset)

if __name__ == "__main__":
    main()
