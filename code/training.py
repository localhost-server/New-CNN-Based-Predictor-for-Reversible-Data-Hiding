# Reproducing the method of splitting 'hu' into four sets

# 1. Network: cnnp
# 2. Input: Original 512x512 grayscale images
# 3. Processing: Predict one set using the other three sets
# 4. Output: Predicted images of size 510x510
# 5. Learning rate: Unchanged, consistently maintained
# 6. Training rounds: 120
# 7. Batch size: 1

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import os
import time
import random
import csv
import math

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Importing network structure
from cnnp import *

# batch1: Points of a single set 512x512
# raw1: Points of another set 510x510, obtained by multiplying the original image by mask1 and then reducing it by one pixel
# mask1: Mask of the set
# mask_c:
def trainOneBatch(batch1, target, mask1):
    # Make predictions, output a 510x510 image
    mycnn.train()
    with torch.enable_grad():
        output = mycnn(batch1)
    # Use a mask with output to get 1/4 of the set
    output = output * mask1
    # Calculate loss, only learn the loss of complex parts
    loss = loss_function(output, target)

    optimizer.zero_grad()  # Reset gradients
    loss.backward()  # Backpropagation, to prevent accumulation of gradients from the previous iteration
    optimizer.step()  # Update parameters
    # Output loss
    return loss

# Read images, this function will be called multiple times later
def get_image(file_path1):
    img = cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE)  # Open the image using cv2, it will be in RGB format, convert it to a single-channel grayscale image
    img = img.astype('float64')
    return img

# Training parameters

# Hyperparameters
LR = 0.001  # Learning rate
BATCH_SIZE = 8  # Batch size, to be adjusted adaptively for each image
Weight_Decay = 0  # L2 regularization parameter
EPOCHES = 10000  # Number of training rounds
# Save images, model names
name = "epoch_"
path_num = 4
# x1_lr001_bs4_ep30_wt00001
begin_epoch = 0

# Model loading path name
# name_model = "train_encoder_x1_adam_cnn_encoder5_2"  # Call the well-learned model for further training

# Check if GPU is available
cuda_available = torch.cuda.is_available()  # If available, both model and data will be placed on the GPU

# Instantiate the network, you can change this network as needed
print('===Start training===')
mycnn = CNNP()

# Copy model parameters
best_model_wts = copy.deepcopy(mycnn.state_dict())
best_acc = 0.0  # Record the best accuracy
best_loss = 10000000.0  # Record the best loss

# Training and testing data structures for plotting
train_loss_all = []  # Record the average loss for each round
train_acc_all = []
val_loss_all = []
val_acc_all = []

since = time.time()  # Start time

if cuda_available:
    mycnn.cuda()  # Move to GPU, as optimizer uses the GPU

# Define optimizer and loss function
optimizer = torch.optim.AdamW(mycnn.parameters(), lr=LR, weight_decay=0.001)
loss_function = nn.MSELoss()  # Use Mean Squared Error loss, normalization problem, the more accurate the prediction, the better

# Process training set data, input original images
imgs_train = []  # Construct a list to store images
imgs_train_target = []
imgs_val = []  # Construct a list to store images
imgs_val_target = []

# Only store the addresses of original images
# Read images
files_train_path = []
DIRECTORY_data = "/home/yangx/CNN_RDH/spl/3000/"  # This is the location of your own images
files_data = os.listdir(DIRECTORY_data)
files_data = sorted(files_data)
for file in files_data:
    file_path = DIRECTORY_data + "/" + file
    files_train_path.append(file_path)

files_test_path = []
test_path = '/home/yangx/CNN_RDH/spl/test32/'
test_data = os.listdir(test_path)
test_data = sorted(test_data)
for file in test_data:
    file_path = test_path + "/" + file
    files_test_path.append(file_path)

# Large size mask
mask_1000 = []
img_ans = np.zeros((1000, 1000))
for i in range(0, 1000, 2):
    for j in range(0, 1000, 2):
        img_ans[i, j] = 1
mask_1000.append(img_ans)
Mask_1000 = torch.FloatTensor(mask_1000).unsqueeze(dim=3).permute(0, 3, 1, 2)
if cuda_available:
    Mask_1000 = Mask_1000.cuda()
# Prediction kernel
mykernel = [[0, 1, 0, 1, 0], [1, 2, 4, 2, 1], [0, 4, 0, 4, 0], [1, 2, 4, 2, 1], [0, 1, 0, 1, 0]]
mykernel = np.array(mykernel)
mykernel = mykernel / np.sum(mykernel)

# Start training
print("train")
# Iterate over epochs
optimizer.param_groups[0]['lr'] = 0.001 * 0.95 ** (begin_epoch // 15)
for epoch in range(begin_epoch + 1, EPOCHES):
    print('-' * 10)
    print('Epoch {}/{} shuffle{}'.format(epoch, EPOCHES - 1, len(files_train_path)))
    if epoch % 15 == 0:
        optimizer.param_groups[0]['lr'] = 0.001 * 0.95 ** (epoch // 15)
    print('lr:', optimizer.param_groups[0]['lr'])

    # Each epoch has two training phases
    train_loss = 0.0
    train_corrects = 0
    train_num = 0

    np.random.shuffle(files_train_path)

    count = 0
    batch = []
    y = []
    pred = []

    block_size = 128
    print('block_size:', block_size)
    for j in range(len(files_train_path)):
        file_path = files_train_path[j]
        img = get_image(file_path)
        img = np.rot90(img, k=np.random.randint(4))
        x_position = np.random.randint(512 - block_size + 1)
        y_position = np.random.randint(512 - block_size + 1)
        img =
