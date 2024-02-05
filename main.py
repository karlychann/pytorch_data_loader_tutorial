### Section 1 - Imports and pytorch setup

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from config import *
from custom_hymenoptera_dataset import HymenopteraDataset
from custom_simple_cnn import SimpleCNN

## If you want to keep track of your network on tensorboard, set USE_TENSORBOARD to 1 in the config file (config.py).
if USE_TENSORBOARD:
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=TENSORBOARD_SERVER)
    try:
        cc.remove_experiment(EXP_NAME)
    except:
        pass
    foo = cc.create_experiment(EXP_NAME)


## If you want to use the GPU, set GPU_MODE TO 1 in config file
use_gpu = GPU_MODE
if use_gpu:
    torch.cuda.set_device(CUDA_DEVICE)


### SECTION 2 - data loading and shuffling/augmentation/normalization : all handled by torch automatically.

# Before being fed into the network during training, each image is transformed by undergoing augmentation and normalization.
#    
# Data augmentation is a strategy to reduce overfitting. Instead of just feeding the same images into the network epoch after epoch
# until the network memorizes them, we feed a slightly perturbed version each time so it never sees exactly the same image more than once. 
# For example (under data_transforms below):
# 1. RandomSizedCrop takes a crop of an image at various scales between 0.08 to 0.8 times the size of the image, and then resizes it to 224x224
# 2. RandomHorizontalFlip flips the image horizontally 50% of the time. For example, an image of a horse facing to the left would now be facing to the right.  
#
# ToTensor() converts each image from a PIL.Image into a torch.Tensor so it can be fed into pytorch models 
#   
# Normalize(means, standard_deviations) normalizes each channel (e.g., red green blue a.k.a. RGB) by the dataset's overall mean and standard deviation
# for each of these three values. This allows the the network to receive input with consistent statistics, improving the mathematical stability of the 
# network as it trains. In practice, it takes a long time to compute the exact mean and standard deviation of an entire dataset, so the values below
# (which are the means and stdevs from a large, popular dataset called ImageNet) are used as defaults, and good enough in most cases.
# The mean and standard deviation of photographs isn't going to change too much - but it might be important to recalculate these if you were working 
# with a very different type of images (e.g., x-ray images).
#
# During validation (as opposed to training), we perform normalization but not augmentation - we want to test the network's performance on natural, 
# unperturbed images. Resize() and CenterCrop() just deterministically ensure that we end up with a 224x224 image. 

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

##################
# ADDED: custom torch Datasets for train and val
##################
dsets = {}
for split in ['train', 'val']:
    dsets[split] = HymenopteraDataset(os.path.join(DATA_DIR, split), data_transforms[split])


##################
# ImageFolder version
    
# ImageFolder is a built-in pytorch class that produces a Dataset instance without you having to define your own Dataset class. 
# It only works if data is formatted in a specific way. 
    
# You need to organize your data such that your dataset folder has EXACTLY two folders. Name these 'train' and 'val' (always lower-case)
# The 'train' folder contains training set and 'val' fodler contains validation set on which accuracy is measured.

# The structure within 'train' and 'val' folders will be the same. They both contain one folder per class. All the images of that class 
# are inside the folder named by class name.

# So basically, if your dataset has 3 classes and you're trying to classify between pictures of 1) dogs 2) cats and 3) humans,
# say you name your dataset folder 'data_dir'. Then inside 'data_dir' will be 'train' and 'val'. Further, Inside 'train' will be
# 3 folders - 'dogs', 'cats', 'humans'. All training images for dogs will be inside this 'dogs' folder. 
# Similarly, within 'val' as well there will be the same 3 folders.

## So, the structure looks like this :
# data_dir
#      |- train
#            |- dogs
#                 |- dog_image_1
#                 |- dog_image_2
#                        .....
#            |- cats
#                 |- cat_image_1
#                 |- cat_image_1
#                        .....
#            |- humans
#      |- val
#            |- dogs
#            |- cats
#            |- humans
##################
# dsets = {}
# for split in ['train', 'val']:
#    dsets[split] = datasets.ImageFolder(os.path.join(DATA_DIR, split), data_transforms[split])
    

dset_sizes = {split: len(dsets[split]) for split in ['train', 'val']}

# Define the dataloaders, using our dataset instances for train and val
dset_loaders = {}
for split in ['train', 'val']:
    dset_loaders[split] = torch.utils.data.DataLoader(dsets[split], batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


### SECTION 3 : Writing the functions that do training and validation phase.

# The code below does forward propogation, back propogation, loss calculation, update weights of model, and save the best model at the end!

# As a brief outline:
#
# For the number of specified epochs (in config.py), train_model goes through a training and a validation phase. 
# Hence the nested for loop: an outer loop for multiple epochs, and an inner loop to iterate through batches of data from the dataloader. 
#
# In both training and validation phases, the loaded data is forward propogated through the model.
# In PyTorch, the Dataloader is an iterator that wraps around a Dataset class. There's a __getitem__ function which gets called every
# time the program iterates over the Dataloader - it fetches two tensors, inputs (the images) and labels (which are integers).
# 
# Forward prop is as simple as calling model() as a function and passing in the input.
# 
# The "Variable" class is like a wrappers on top of PyTorch Tensors that keeps track of every mathematical operation that tensor goes through.
# The benefit of this is that you don't need to write the equations for backpropogation, because the history of computations has been tracked
# and pytorch can automatically differentiate it! Thus, 2 things are SUPER important. ALWAYS check for these 2 things.
# 1) NEVER overwrite a pytorch variable, as all previous history will be lost and automatic differentation ("autograd") won't work.
# 2) Variables can only undergo operations that are differentiable.

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=100):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to eval mode for validation (no need to track gradients)

            running_loss = 0.0
            running_corrects = 0

            counter=0
            # Iterate over data, getting one batch of inputs (images) and labels each time.
            for data in dset_loaders[phase]:
                inputs, labels = data

                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()),
                        Variable(labels.long().cuda())
                    except:
                        print(inputs,labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                # Print a line every 10 batches so you have something to watch and don't feel like the program isn't running.
                if counter%10==0:
                    print("Reached batch iteration", counter)

                counter+=1

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # print evaluation statistics
                try:
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)
                except:
                    print('unexpected error, could not calculate loss or do a sum.')
            print('trying epoch loss')
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.item() / float(dset_sizes[phase])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if USE_TENSORBOARD:
                    foo.add_scalar_value('epoch_loss',epoch_loss,step=epoch)
                    foo.add_scalar_value('epoch_acc',epoch_acc,step=epoch)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    print('new best accuracy = ',best_acc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('returning and looping back')
    return best_model


# This function changes the learning rate as the model trains.
# If the learning rate is too high, training tends to be unstable and it's harder to converge on an optimal set of weights. 
# But, if learning rate is too low, learning is too slow and you won't converge in a reasonable time frame. A good compromise 
# is to start out with a high learning rate and then reduce it over time. 
def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (DECAY_WEIGHT**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


### SECTION 4 : DEFINING MODEL ARCHITECTURE.

##################
# MODIFIED: Added
# Set the number of classes in the config file by setting the right value for NUM_CLASSES.
##################
model_ft = SimpleCNN(num_classes=NUM_CLASSES)

##################
# MODIFIED: Deleted
# We use Resnet18 here. If you have more computational power, feel free to swap it with Resnet50, Resnet100 or Resnet152.
# For fine tuning or transfer learning, we will use the pretrained net weights.
##################
# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)


criterion = nn.CrossEntropyLoss()

if use_gpu:
    criterion.cuda()
    model_ft.cuda()

optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.0001)


# Run the functions and save the best model in the function model_ft.
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=NUM_EPOCHS)

# Save model
torch.save(model_ft.state_dict(), 'fine_tuned_best_model.pt')
