# About this project

In this model we create a simple CNN from scratch to classify image data. We're using the CIFAR10 dataset, popular amongst computer vision models. It has 10 classes, including vehicles and animals.  
The dataset can be downloaded using the commented part of the code after the initial set up (be sure to update the `ROOT_DIR` to your own directory). If you already have the dataset downloaded, it gets fetched from the `ROOT_DIR`. The dataset is divided into 2 parts: train and test.  
We start by defining transformers, one for the training data and one for validation and test data.

#### For training data:
* Adding random cropping to the image  
* Randomly flips the image horizontally with 50% chance  
* Converts the image to a torch tensor  
* Normalizes the tensor with `mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2470, 0.2435, 0.2616]`, which are known metrics for the data used  

#### For the test data:
* Converts the image to a torch tensor  
* Normalizes the tensor with `mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2470, 0.2435, 0.2616]`, which are known metrics for the data used  

We do an extra split, leaving out 10% of the training data for validation. With now 3 datasets — training, validation and testing — we make our data loaders to be used in the model.

---

## CNN

A simple CNN with:
* 3 convolutional layers with `kernel_size = 3` and `padding = 1`  
* 3 normalization layers for each previous layer  
* added pooling with `kernel_size = 2`, `stride = 2`  
* 2 fully connected layers  
* added a dropout layer with `p=0.25` by default, but can be changed when calling the class  

#### Forward pass:
* used ReLU as an activation function after each convolutional layer  
* used max pooling after each convolutional block to reduce spatial dimensions  
* flattened the output before passing through the fully connected layers  
* applied dropout regularization before the final layer to prevent overfitting  
* the final layer outputs logits for 10 classes corresponding to CIFAR10 categories  

---

## Training and validation

The training process is handled by the `train_and_validate()` function. For each epoch, the model:
* runs a forward pass through the network  
* computes the loss using the CrossEntropyLoss criterion  
* performs backpropagation and updates weights through the Adam optimizer  
* averages the training loss for that epoch  

After training, the model enters evaluation mode and runs through the validation data. It computes both validation loss and accuracy for each epoch. At the end of training, the function plots:
* training vs. validation loss curves  
* validation accuracy per epoch  

These plots are useful to monitor convergence and detect overfitting or underfitting trends.

---

## Hyperparameter tuning

The script tests different configurations of learning rate, weight decay, and dropout using itertools’ product function. For each combination:
* the model is initialized with the given dropout  
* Adam optimizer is set with the current learning rate and weight decay  
* model is trained for 5 epochs  
* the final validation accuracy is recorded  

After iterating through all combinations, the configuration achieving the highest validation accuracy is printed, along with its parameters. The model weights corresponding to that configuration are saved as the best model state.

---

## Testing phase

Once the best hyperparameters are found, the model is reloaded with the best configuration and evaluated on the test set using the `test_loop()` function.  
This function:
* sets the model to evaluation mode  
* iterates through the test data without computing gradients  
* calculates the total accuracy over the full dataset  

The final test accuracy is printed at the end, providing an unbiased estimate of the model’s generalization performance.

---

## Results and interpretation

The model’s performance depends on the chosen hyperparameters and number of epochs. Typically, with a simple CNN and limited tuning, the accuracy ranges between 70% and 80% on CIFAR10.  

To further improve results, additional steps could include:
* increasing the number of epochs  
* adding more convolutional layers  
* experimenting with learning rate schedules  
* introducing data augmentations like color jitter or rotation  
* using pretrained models (e.g., ResNet, VGG) for transfer learning  



