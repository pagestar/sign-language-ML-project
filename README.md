# sign-language-ML-project

## Dataset usage

```FrameDataset(root_dir, frame_size=(224, 224), transform=None```
- root_dir: Root directory of dataset (video)
- frame_size: Size of processed dataset

### attribute
```
model = FrameDataset(root_dir)
model.data                  # mediapipe processed data
model.labels                # label of each data
model.counts                # count of frames of each data
model.frames_folders        # where the processed frames stored (避免超出 Memory 上限)

model.word2label            # word to label mapping
model.label2word            # label to word mapping
```
### usage
```
for idx, data in enumerate(dataset):
    imgs = dataset.get_frame_imgs(idx)  # frame images of target index
    logits, lable = data    # (data from mediapipe, label of data (int))
```

## CNN usage
```
FrameClassifier(num_classes, feature_dim=128)
```
- num_classes: total number of labels (classes)
- feature_dim: dimentions (count) of the feature

---
```
train_model(model: FrameClassifier, optimizer, criterion, dataset:FrameDataset, epochs)
```
train the model
- model: the model to train
- return: the model after training
---
```
eval_model(model: FrameClassifier, X)
```
eveluate the value X with model <br/>
return: the evaluated value (predict_label, features)