# For plotting
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.models as models
from torchvision import datasets, transforms
# For utilities
import os, shutil, time
from pathlib import Path
from PIL import Image


Path = Path.home()

home = str(Path)+'\PycharmProjects\pythonProject\\'

#(x)


import zipfile
import shutil


# Specify the path to the zip file
zip_file_path = home+'\photo_lib.zip'

# Specify the path to the destination folder
destination_folder = home+'total_folder\\test_256\\test_256\\'


"""
Run this part of the code once
# Extract the contents of the zip file
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     #file_list = zip_ref.namelist()
#     zip_ref.extractall(destination_folder)


#print(file_list[0])
# Move 10 files from the extracted folder to another folder
# extracted_files = zip_ref.namelist()[:10]  # Get a list of all the extracted files
# files_to_move = extracted_files[:10]  # Select the first 10 files (adjust as needed)
"""

#Use when need to add more photos to the training se
# extracted_files = os.listdir(destination_folder)
# files_to_move = extracted_files[:500]
#
# for file_name in files_to_move:
#     source_path = destination_folder + file_name
#     destination_path = home +'train\\train_folder\\' + file_name
#     modified_dp = destination_path.replace('/', '\\')
#     shutil.move(source_path, modified_dp)
#


#Path_test = Path()

train_path = home +'train'
#train_path = Path_test / 'train_folder'
val_path = home + 'val'

val_inner = val_path + '/val_folder'

#Debugg
# x = os.scandir(train_path)
#
# with os.scandir(train_path) as itr:
#     for entry in itr :
#         # Check if the entry
#         # is directory
#         if entry.is_dir() :
#             print("% s is a directory." % entry.name)
#         else:
#             print("% s is not a directory." % entry.name)







# # If the image folder doesn't exist, download it and prepare it...
# if x.is_dir():
#     print(f"{x} directory exists.")
# else:
#     print(f"Did not find {x} directory, creating one...")
#     x.mkdir(parents=True, exist_ok=True)



# file_name_v = extracted_files[25]
# source_path_v = destination_folder + file_name_v
# destination_path_v = val_path +'\\' + file_name_v
# modified_dp_v = destination_path_v.replace('/', '\\')
# shutil.move(source_path_v, modified_dp_v)



class ColorizationNet(nn.Module):
    def __init__(self, input_size=128):
        super(ColorizationNet, self).__init__()
        MIDLEVEL_FEATURE_SIZE = 128

        ## First half: ResNet
        resnet = models.resnet18(num_classes=365)
        # Change first conv layer to accept single-channel (grayscale) input
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        # Extract midlevel features from ResNet-gray
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

        ## Second half: Upsampling
        self.upsample = nn.Sequential(
          nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Upsample(scale_factor=2),
          nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Upsample(scale_factor=2),
          nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
          nn.Upsample(scale_factor=2)
        )

    def forward(self, input):

        # Pass input through ResNet-gray to extract features
        midlevel_features = self.midlevel_resnet(input)

        # Upsample to get colors
        output = self.upsample(midlevel_features)
        return output











def shrink_image(image_path, output_path, size):
    # Open the image
    image = Image.open(image_path)

    # Calculate the aspect ratio
    width, height = image.size
    aspect_ratio = width / float(height)

    # Determine the dimensions to fit within the bounding box
    if aspect_ratio > 1:
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        new_width = int(size * aspect_ratio)
        new_height = size

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new blank image with the desired size
    output_image = Image.new('RGB', (size, size))

    # Calculate the position to paste the resized image
    left = int((size - new_width) / 2)
    top = int((size - new_height) / 2)
    right = left + new_width
    bottom = top + new_height

    # Paste the resized image onto the blank image
    output_image.paste(resized_image, (left, top, right, bottom))

    # Save the resized and shrinked image
    output_image.save(output_path)

# Example usage
input_image_path = home + 'Hela_cells.jpg'  # Replace with your input image path
input_image_path = input_image_path.replace('/', '\\')

output_image_path = home +'hela_cells_256.jpg'  # Replace with the desired output image path
output_image_path = output_image_path.replace('/', '\\')
desired_size = 256

shrink_image(input_image_path, output_image_path, desired_size)
shutil.move(output_image_path,val_inner+'\\hela_cells_256.jpg')

use_gpu = torch.cuda.is_available()
model = ColorizationNet()

#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

class GrayscaleImageFolder(datasets.ImageFolder):
  '''Custom images folder, which converts images to grayscale before loading'''
  def __getitem__(self, index):
    path, target = self.imgs[index]
    img = self.loader(path)
    if self.transform is not None:
      img_original = self.transform(img)
      img_original = np.asarray(img_original)
      img_lab = rgb2lab(img_original)
      img_lab = (img_lab + 128) / 255
      img_ab = img_lab[:, :, 1:3]
      img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
      img_original = rgb2gray(img_original)
      img_original = torch.from_numpy(img_original).unsqueeze(0).float()
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img_original, img_ab, target


# # Training
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
train_imagefolder = GrayscaleImageFolder(train_path, train_transforms)
train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=64, shuffle=True)

# Validation
val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
val_imagefolder = GrayscaleImageFolder(val_path, val_transforms)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)


class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial'''
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
  '''Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
  plt.clf() # clear matplotlib
  color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
  color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
  color_image = lab2rgb(color_image.astype(np.float64))
  grayscale_input = grayscale_input.squeeze().numpy()
  if save_path is not None and save_name is not None:
    plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))



def validate(val_loader, model, criterion, save_images, epoch):
  model.eval()

  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  already_saved_images = False
  for i, (input_gray, input_ab, target) in enumerate(val_loader):
    data_time.update(time.time() - end)

    # Use GPU
    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    # Run model and record loss
    output_ab = model(input_gray) # throw away class predictions
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

    # Save images to file
    if save_images and not already_saved_images:
      already_saved_images = True
      for j in range(min(len(output_ab), 10)): # save at most 5 images
        save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
        save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
        to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

    # Record time to do forward passes and save images
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to both value and validation
    if i % 25 == 0:
      print('Validate: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
             i, len(val_loader), batch_time=batch_time, loss=losses))

  print('Finished validation.')
  return losses.avg



epoch_list = []
train_losses = []
val_losses = []

def train(train_loader, model, criterion, optimizer, epoch, epoch_list,train_losses):




  print('Starting training epoch {}'.format(epoch))
  model.train()

  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()






  for i, (input_gray, input_ab, target) in enumerate(train_loader):

    # Use GPU if available
    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    # Record time to load data (above)
    data_time.update(time.time() - end)

    # Run forward pass
    output_ab = model(input_gray)
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

    # Compute gradient and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    # Record the loss value
    train_losses.append(loss.item())






    # Record time to do forward and backward passes
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to value, not validation
    if i % 25 == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
              epoch, i, len(train_loader), batch_time=batch_time,
             data_time=data_time, loss=losses))

  print('Finished training epoch {}'.format(epoch))
  epoch_list.append(epoch)
  return epoch_list



# Move model and loss function to GPU
if use_gpu:
  criterion = criterion.cuda()
  model = model.cuda()



# Make folders and set parameters
os.makedirs('outputs/color', exist_ok=True)
os.makedirs('outputs/gray', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
save_images = True
best_losses = 1e10
epochs = 100


# Train model
for epoch in range(epochs):
  # Train for one epoch, then validate
  train(train_loader, model, criterion, optimizer, epoch,epoch_list, train_losses)
  with torch.no_grad():
    losses = validate(val_loader, model, criterion, save_images, epoch)
    val_losses.append(losses)
  # Save checkpoint and replace old best model if current model is better
  if losses < best_losses:
    best_losses = losses
    torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch+1,losses))
# Show images
import matplotlib.image as mpimg
image_pairs = [('outputs/color/img-3-epoch-99.jpg', 'outputs/gray/img-3-epoch-0.jpg'),
               ('outputs/color/img-1-epoch-99.jpg', 'outputs/gray/img-1-epoch-0.jpg'),
               ('outputs/color/img-2-epoch-99.jpg', 'outputs/gray/img-2-epoch-0.jpg')]
#
#
# x = np.linspace(0, 100, 100)  # Range of predicted values
#  # MSE loss for each predicted value
# plt.plot(x, epoch_list)
# plt.xlabel('Predicted Values')
# plt.ylabel('Mean Squared Error (MSE)')
# plt.title('MSE Loss Function')
# plt.grid(True)
# plt.show()
#


# mlp for regression with mse loss function


# Plot the loss function graph

plt.plot(epoch_list, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Over Epochs')
plt.legend()
plt.show()


for c, g in image_pairs:
  color = mpimg.imread(c)
  gray  = mpimg.imread(g)
  f, axarr = plt.subplots(1, 2)
  f.set_size_inches(15, 15)
  axarr[0].imshow(gray, cmap='gray')
  axarr[1].imshow(color)
  axarr[0].axis('off'), axarr[1].axis('off')
  plt.show()

# Plot the loss function graph
plt.plot(epoch_list, train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
