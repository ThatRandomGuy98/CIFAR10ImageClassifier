# CIFAR-10 Image Classifier (PyTorch)

This subfolder contains a **from-scratch CNN** built with **PyTorch** to classify images from the **CIFAR-10** dataset (10 classes of 32×32 RGB images such as airplanes, cats, trucks, etc.).

The goal of the project is **learning and experimentation**:
- Build and understand a custom CNN architecture
- Apply standard data augmentation and normalization
- Implement a clean **training + validation + test** pipeline
- Perform **manual hyperparameter search** (learning rate, weight decay, dropout)
- Visualize training and validation curves to reason about under/overfitting

---

## 1. Dataset

The project uses **CIFAR-10** from `torchvision.datasets`.

- **Train set:** 50,000 images  
- **Test set:** 10,000 images  
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The dataset is downloaded or loaded from a local directory configured in the code via a `ROOT_DIR` variable.

### 1.1. Transforms

Two separate transform pipelines are used:

**Training transforms:**
- Random crop with padding
- Random horizontal flip (50% chance)
- Convert to tensor
- Normalize with known CIFAR-10 stats:  
  `mean = [0.4914, 0.4822, 0.4465]`  
  `std  = [0.2470, 0.2435, 0.2616]`

**Validation / Test transforms:**
- Convert to tensor
- Same normalization as above

The original train set is further split into:
- **Train set**
- **Validation set** (10% of the original train set)

You end up with 3 dataloaders:
- `train_loader`
- `val_loader`
- `test_loader`

---

## 2. Model Architecture (CNN)

The classifier is a **custom convolutional neural network** with:

- **3 convolutional blocks**
  - `kernel_size = 3`, `padding = 1`
  - Each conv followed by a normalization layer
  - Max pooling (`kernel_size = 2`, `stride = 2`) to downsample feature maps
- **2 fully connected (linear) layers**
- **Dropout layer**
  - Default `p = 0.25` (configurable when constructing the model)

### 2.1. Forward pass

The forward pass follows this pattern:

1. Convolution → BatchNorm/Norm → ReLU → MaxPool (repeated for each block)  
2. Flatten the convolutional output  
3. Fully connected layer(s)  
4. Dropout before the final layer for regularization  
5. Output logits for **10 classes** (the CIFAR-10 categories)

The loss function used is **Cross Entropy Loss**, which is standard for multi-class classification.

---

## 3. Training & Validation

Training is handled by a function similar to `train_and_validate(...)` which:

For each epoch:

1. **Training phase**
   - Set model to `train()` mode
   - Loop over `train_loader`
   - Forward pass
   - Compute loss (`nn.CrossEntropyLoss`)
   - Backprop (`loss.backward()`)
   - Update weights (`optimizer.step()`)
   - Track average training loss

2. **Validation phase**
   - Set model to `eval()` mode
   - Disable gradients with `torch.no_grad()`
   - Loop over `val_loader`
   - Compute validation loss
   - Compute validation accuracy
   - Track metrics per epoch

At the end, the script produces plots such as:
- **Training vs validation loss**
- **Validation accuracy over epochs**

These curves make it easier to reason about:
- Underfitting (both losses high)
- Overfitting (train loss low, val loss high)
- Whether more epochs or regularization might help

---

## 4. Hyperparameter Search

The project performs a **manual grid search** over a small hyperparameter space using `itertools.product`, testing different combinations of:

- Learning rate (`lr`)
- Weight decay (`wd`)
- Dropout probability (`dropout`)

For each combination:

1. A fresh model is instantiated (with the chosen dropout)
2. An optimizer (e.g. Adam) is created with the current `lr` and `weight_decay`
3. The model is trained for a fixed number of epochs (e.g. 5)
4. Final **validation accuracy** is recorded

At the end of the search:

- The **best configuration** is identified (highest validation accuracy)
- The corresponding **model `state_dict`** is saved so it can be reloaded later for testing

This acts as a simple but effective hyperparameter tuning loop.

---

## 5. Test Phase

After picking the best hyperparameters, the script:

1. Recreates the model with the best configuration
2. Loads the saved best weights
3. Evaluates on the **test set** using a `test_loop(...)` function:

The test loop typically:

- Sets the model to evaluation mode
- Loops over `test_loader` without gradients
- Computes predictions and compares to labels
- Accumulates the total number of correct predictions
- Prints **final test accuracy** as an estimate of generalization

---

## 6. How to Run

### 6.1. Clone the repository

```bash
git clone https://github.com/ThatRandomGuy98/CIFAR10ImageClassifier.git
cd CIFAR10ImageClassifier/cifar10_classifier
