import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from Dataloader import FrameDataset, VideoDataAugmentation
from GestureModel import GestureModel, eval_model, train_model
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from CNN import FrameCNN, train_cnn, eval_cnn, extract_logits, test_CNN


'''
Print the predicted and actual labels for each video for GestureModel.
'''
def print_result(labels, dataset):
    labels = labels.cpu().numpy()
    correct = 0
    total = 0
    for i in range(len(labels)):
        print(f"Video {i+1}: Predicted = {dataset.label2word[labels[i]]}, Actual = {dataset.label2word[dataset.labels[i]]}")
        if labels[i] == dataset.labels[i]:
            correct += 1
        total += 1
    print(f"Accuracy: {correct/total}")

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

def plot_loss(loss_list):
   
    plt.plot(loss_list, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    
    plt.show()


def main():

    root_dir = "data"  
    num_classes = 5
    feature_dim = 128

    '''
    You can adjust the hyperparameters here.
    '''
    hidden_dim = 128
    num_layers = 3
    learning_rate = 1e-3
    epochs = 150

    transform = VideoDataAugmentation()

    '''
    Dataset & Model 
    Maybe something is wrong for dataloader
    '''
    dataset = FrameDataset(root_dir=root_dir, transform=transform)  
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)  

    cnn = FrameCNN(feature_dim=128).cuda()  
    rnn = GestureModel(input_size=128, 
                    hidden_size=hidden_dim, num_classes=num_classes, num_layers=num_layers).cuda()

    '''
    Optimizers & Loss function
    '''
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    '''
    Training CNN
    '''
    print("Start training CNN...")
    _, loss_list, _ = train_cnn(cnn, cnn_optimizer, criterion, dataset, num_epochs=epochs)

    # load the pre-trained CNN model
    # cnn.load_state_dict(torch.load("trained_cnn.pth")) 

    test_CNN(cnn, dataset)
    torch.save(cnn.state_dict(), "trained_cnn.pth")  # Save the model parameters
    # Plot the training loss of CNN
    plot_loss(loss_list)

    '''
    Extract feature maps and combine them with frames.
    Maybe something is wrong in this step.
    '''
    features = extract_logits(cnn, dataloader)
    print("Feature maps shape:", features.shape)

    '''
    Training RNN
    '''
    labels = torch.tensor(dataset.labels, dtype=torch.long) 
    #print(labels)
    rnn_dataset = TensorDataset(features, labels)
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
