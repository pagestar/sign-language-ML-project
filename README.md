# Issues

Here are some issues that have been observed and need to be addressed:

## Data Collection

We may want to collect more data to improve the accuracy of our model, and a possible solution is to collect data from different sources. We want to find some words that is from another category or language (ASl, JSL, etc.) but with **similar gestures**.
> See data folder for example

We want to collect at least 10 classes, each with 6-10 videos as possible.

### WLASL

There is a `achieve.zip` in the folder. This is a dataset from WLASL project. To use the dataset, check if the word have the similar gestures first.
- [TSL Dictionary](https://twtsl.ccu.edu.tw/TSL/#result)
- [ASL Dictionary](https://www.handspeak.com/word/)
> The .zip file is too large to be uploaded to GitHub. So I upload on the [Google Drive](https://drive.google.com/file/d/1dVayiPZ3q1Emkbl0_PwYabxd_C0f2vBR/view?usp=drive_link). Or you may want to visit the [WLASL website](https://dxli94.github.io/WLASL/).

If you find such a word, check `WLASL_v0.3.json` and find the word's class.
> An easy way is CTRL+F with `"gloss": "{yopr word}"`.

Then refer to the `video_id` and find in the `videos` folder. A word can have multiple videos, so you may want to check all of them.
> While some of them may be missing.

## CNN
I have add a function `test_CNN` to test the CNN model individually. It's recommended to train more epochs (> 200). 
* I have increased the layers, it may need 20-30 secons for 1 epoch. Note that it's for legacy data.
* You can create a dataloader if you want to speed up by increasing batch size.
* The scheduler now decreases the learning rate by 0.5 if the model doesn't improve for 5 epochs. 
* You can adjust any parameters.

## RNN
A known issues is that the loss has dropped to 0.0000 after only a few epochs (maybe 10), which may suggest that the model is overfitting. 
* I'm not sure if there is any troubles in `extract_and_combine_features`.
* Some suggestions for input data are noted there.
* You can adjust any parameters.

## Other 

Make sure to check the reviews from other groups and TAs.
