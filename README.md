# RNN
The folder contains these files:
* `convert_data.py`: Convert multiple data into a format that can be used by the RNN, supported by Mediapipe. A sample data is provided in the `data` folder.

    > You can find a sample output in `output_matrix.txt`.

* `model.py`: A simple RNN model `GestureRNN` that can be trained on the converted data.
* `train_model.py`: Train and evaluate the `GestureRNN` model on the converted data.
* `main.py`: The main file.

> Haven't evaluate the performance of the model yet.

## Padding

In `main` file, we pad the processed data from Mediapipe to a fixed length. The length is determined by the maximum length of the **tensors**.
```py
def pad_features(input_tensors, target_size):
    '''
    Pad the input tensors to a fixed length, whicxh is the maximum points that Mediapipe has fetched.

    Args:
        input_tensors (list): A list of input tensors (matrices).
        target_size (int): The target size to pad the tensors, expected to be max_feature_size.

    Returns:    
        padded_tensors (list): A list of padded input tensors.
    '''
    padded_tensors = []
    for tensor in input_tensors:
        padding_size = target_size - tensor.shape[2]
        if padding_size > 0:
            padding = torch.zeros((tensor.shape[0], tensor.shape[1], padding_size))
            padded_tensor = torch.cat((tensor, padding), dim=2)
        else:
            padded_tensor = tensor
        padded_tensors.append(padded_tensor)
    return padded_tensors

# Get the maximum feature size, which is the maximum number of points that Mediapipe has fetched.
max_feature_size = max(tensor.shape[2] for tensor in rnn_inputs)
padded_inputs = pad_features(rnn_inputs, max_feature_size)
```

