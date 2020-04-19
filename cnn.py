# %% import standard PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision module to handle image manipulation
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print("PyTorch is running on GPU!")
else:
    device = torch.device("cpu")
    print("PyTorch is running on CPU!")

# %% CNN MODELS
class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()

        # define layers
        self.layer1_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=1), # Keeping estimates of mean/variance during training to normalize data during testing/evaluation [https://arxiv.org/pdf/1502.03167.pdf]
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),    #1@28x28 -> 32@28x28
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),   #32@28x28 -> 32@28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                  #32@28x28 -> 32@14x14
            nn.Dropout(0.2)) # Overfitting is reduced by randomly omitting the feature detectors on every forward call [https://arxiv.org/abs/1207.0580]                            
        self.layer2_conv  = nn.Sequential(
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),   #32@14x14 -> 64@14x14
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),   #64@14x14 -> 64@14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                  #64@14x14 -> 64@7x7
            nn.Dropout(0.3))
        self.layer3_fc = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.Flatten(),
            nn.Linear(in_features=64*7*7, out_features=1000),
            nn.ReLU())
        self.layer4_fc = nn.Sequential(
            nn.Linear(in_features=1000, out_features=100),
            nn.ReLU())
        # don't need softmax here since we'll use cross-entropy as activation.
        self.layer5_out = nn.Linear(in_features=100, out_features=10)

    # define forward function
    def forward(self, t):
        t = self.layer1_conv(t)
        t = self.layer2_conv(t)
        t = self.layer3_fc(t)
        t = self.layer4_fc(t)
        t = self.layer5_out(t)
        return t

# %% PARAMETERS & DATA
lr = 0.001
batch_size = 100
epochs = 20
val_set_ratio = 0.1

criterion = nn.CrossEntropyLoss()

# Using standard FashionMNIST dataset
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)
# train_set, val_set = torch.utils.data.random_split(train_set, [int((1-val_set_ratio)*len(train_set)), int(val_set_ratio*len(train_set))])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

# %% CNN TRAINING & VALIDATION
CNN = CNN_model()
CNN.to(device)

optimizer = optim.Adam(CNN.parameters(), lr=lr)

acc_cnn_train = np.zeros(epochs)
loss_cnn_train = np.zeros(epochs)
acc_cnn_val = np.zeros(epochs)
loss_cnn_val = np.zeros(epochs)
best_val_acc = 0

for epoch in range(epochs):
    #Training Phase
    CNN.train() # set the CNN to training mode
    count = 0
    correct = 0
    total_loss = 0
    
    prog_bar = tqdm(train_loader, desc="Epoch %d (Training the model) " %(epoch), leave=False, ncols = 100)
    for images,labels in prog_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()               # Zero out the gradients in the optimizer
        # forward + backward + optimize
        preds = CNN(images)                 # This calls the forward() function
        loss = criterion (preds, labels)    # Calculate the loss
        loss.backward()                     # Backpropagate the gradients
        optimizer.step()                    # Update the parameters based on the backpropagation

        total_loss += loss
        count += labels.size(0)
        correct += preds.argmax(dim=1).eq(labels).sum().item()
    loss_cnn_train[epoch] = total_loss/len(train_loader) 
    acc_cnn_train[epoch] = (correct/count)*100.0
    print('Epoch %d : Training Accuracy = %.3f%%, Loss = %.3f (%.2fit/s)' % (epoch, acc_cnn_train[epoch], loss_cnn_train[epoch], 1/prog_bar.avg_time) )
    
    # Validation Phase
    CNN.eval() # Set the CNN to test mode
    with torch.no_grad(): # Disables the autograd functionality in the model
        correct = 0
        count = 0
        total_loss = 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            preds = CNN(images)
            loss = criterion (preds, labels)

            total_loss += loss
            count += labels.size(0)
            correct += preds.argmax(dim=1).eq(labels).sum().item()

        loss_cnn_val[epoch] = total_loss/len(test_loader) 
        acc_cnn_val[epoch] = (correct/count)*100.0
        print('Validation Accuracy = %.3f%%, Loss = %.3f' % (acc_cnn_val[epoch], loss_cnn_val[epoch]))
    
    if (best_val_acc < acc_cnn_val[epoch]):
        best_val_acc = acc_cnn_val[epoch]
        torch.save(CNN, 'best_model')

# %% CNN TESTING
CNN_best = torch.load('best_model')
CNN_best.to(device)
CNN_best.eval() # Set the CNN_best to test mode
acc_cnn_test_batch = np.zeros(len(test_loader))
loss_cnn_test_batch = np.zeros(len(test_loader))

with torch.no_grad(): # Disables the autograd functionality in the model
    correct = 0
    count = 0
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        preds = CNN_best(images)
        loss = criterion (preds, labels)

        count += labels.size(0)
        correct += preds.argmax(dim=1).eq(labels).sum().item()
        acc_cnn_test_batch[i] = preds.argmax(dim=1).eq(labels).sum().item()/labels.size(0)*100.0
        loss_cnn_test_batch[i] = loss

        print('Batch %d : Test Accuracy = %.3f%%, Loss = %.3f' % (i,acc_cnn_test_batch[i], loss_cnn_test_batch[i]))
    print('Overall Accuracy = %.3f%%' % ((correct/count)*100.0))
    acc_cnn_test = (correct/count)*100 

# %%  PlOTTING RESULTS
f_test = plt.figure(figsize=(8,5))
plt.plot(acc_cnn_test_batch)
plt.title("Test Accuracy on Different Batches (Overall = %.3f%%)" %(acc_cnn_test))
plt.ylabel("Accuracy (%)")
plt.xlabel('Batch Number')
plt.grid(True, linestyle=':')
plt.xticks(np.arange(0, len(test_loader)))
plt.autoscale(enable=True, axis='x', tight=True)

f_train = plt.figure(figsize=(8,5))
plt.plot(acc_cnn_train, label='Training Accuracy')
plt.plot(acc_cnn_val, label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.ylabel("Accuracy (%)")
plt.xlabel('Number of Epochs')
plt.grid(True, linestyle=':')
plt.xticks(np.arange(0, len(acc_cnn_train),step=2))
plt.autoscale(enable=True, axis='x', tight=True)

f_train_loss = plt.figure(figsize=(8,5))
plt.plot(loss_cnn_train, label='Training Loss')
plt.plot(loss_cnn_val, label='Validation Loss')
plt.title("Training and Validation Loss")
plt.ylabel("Cross Entropy Loss")
plt.xlabel('Number of Epochs')
plt.grid(True, linestyle=':')
plt.xticks(np.arange(0, len(acc_cnn_train),step=2))
plt.autoscale(enable=True, axis='x', tight=True)

plt.show()

