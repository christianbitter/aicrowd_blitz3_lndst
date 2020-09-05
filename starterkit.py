import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

root='/home/local/DFPOC/imvha/aicrowd/blitz3/lndst/data'

#UNET NETWORK
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

model = UNet(3, 1).cuda()
learningRate = 0.001
optimizer = torch.optim.RMSprop(model.parameters(), lr=learningRate, weight_decay=1e-8, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()

# loading data
images_path=os.path.join(root,"train_images/")
groundTruth_path=os.path.join(root, "train_gt/")
saving_bestModel_path=os.path.join(root, "unet_1.pt")

# split
class LandsatDataset(Dataset):
    def __init__(self, training="True"):
        water_arr = os.listdir(images_path)
        training_images_count = 800

        if (training == "True"):
            print("training")
            # how many images do you want to have in training
            self.arr = water_arr[:training_images_count]
        else:
            print("Validation")
            self.arr = water_arr[training_images_count:]

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):

        image_name = self.arr[idx]
        input_path = images_path + image_name
        gt_path = groundTruth_path + image_name[:-3] + "png"
        gt_img = cv2.imread(gt_path, cv2.COLOR_BGR2GRAY)
        gt_img = gt_img.reshape((1, 400, 400))
        input_img = cv2.imread(input_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.reshape((3, 400, 400))
        return input_img, gt_img

X = LandsatDataset("True")
Y = LandsatDataset("False")

print(X.__len__())

# dataloader
batch_size=8
train_loader = torch.utils.data.DataLoader(X,batch_size=batch_size,shuffle=True)
val_loader = torch.utils.data.DataLoader(Y,batch_size=batch_size,shuffle=True)

'''
# first training iteration
epochs = 1
best_loss = 1000
for epoch in range(epochs):

    model.train()
    print('epochs {}/{} '.format(epoch + 1, epochs))
    running_loss = 0.0
    running_loss_v = 0.0

    for idx, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for idx, (inputs_v, labels_v) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs_v = inputs_v.type(torch.FloatTensor)
            labels_v = labels_v.type(torch.FloatTensor)
            inputs_v = inputs_v.cuda()
            labels_v = labels_v.cuda()
            outputs_v = model(inputs_v)
            loss_v = criterion(outputs_v, labels_v)
            running_loss_v += loss_v

        if (running_loss_v / len(val_loader)) < best_loss:
            best_loss = running_loss_v / len(val_loader)
            out = torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dev_loss': best_loss,
            }, f=saving_bestModel_path)
    print('loss : {:.4f}   val_loss : {:.4f}'.format((running_loss / len(train_loader)),
                                                     (running_loss_v / len(val_loader))))

# saving weights
torch.save(model.state_dict(),saving_bestModel_path)
'''

'''
# second training iteration
saving_bestModel_path = os.path.join(root, "unet_2.pt")
epochs = 1
best_loss = 1000
for epoch in range(epochs):

    model.train()
    print('epochs {}/{} '.format(epoch + 1, epochs))
    running_loss = 0.0
    running_loss_v = 0.0

    for idx, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for idx, (inputs_v, labels_v) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs_v = inputs_v.type(torch.FloatTensor)
            labels_v = labels_v.type(torch.FloatTensor)
            inputs_v = inputs_v.cuda()
            labels_v = labels_v.cuda()
            outputs_v = model(inputs_v)
            loss_v = criterion(outputs_v, labels_v)
            running_loss_v += loss_v

        if (running_loss_v / len(val_loader)) < best_loss:
            best_loss = running_loss_v / len(val_loader)
            out = torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dev_loss': best_loss,
            }, f=saving_bestModel_path)
    print('loss : {:.4f}   val_loss : {:.4f}'.format((running_loss / len(train_loader)),
                                                     (running_loss_v / len(val_loader))))
'''

output_mask_path = os.path.join(root, 'iteration2/masks/')
'''
#Run Prediction
checkpoint = torch.load(os.path.join(root, "unet_2.pt"))
model = UNet(3,1).cuda()
model.load_state_dict=checkpoint['model']

checkpoint['epoch']

#Lets load a single image and see how well our model is performing
test_data_path = os.path.join(root, "test_images/")
# If each pixel value is above 0.5, we mark it as water, else mark it as land
threshold = 0.5

for j in os.listdir(test_data_path):
  test_image_path = test_data_path + j
  print(test_image_path)
  img_plot = cv2.imread(test_image_path)
  img_plot = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
  img = img_plot.reshape((1,3,400,400))
  img = torch.tensor(img).float()
  model.eval()
  with torch.no_grad():
      img = img.cuda()
      output = model(img)
      probs = torch.sigmoid(output)
      # print("probs output ",probs.shape)
  probs = probs.squeeze(0)
  # print("probs after squeeze",probs.shape)
  probs = probs.reshape((400,400)).detach().cpu().numpy()

  probs = probs.flatten()
  binary_probs = []
  for i in probs:
    if (i>threshold):
        binary_probs.append(1)
    else:
        binary_probs.append(0)
  binary_probs = np.asarray(binary_probs)
  # print("binary_probs ",binary_probs.shape)
  output_mask = binary_probs.reshape((400,400))
  # print("output_mask shape ",output_mask.shape)
  output_filename = output_mask_path+j[:-3]+'png'
  print(output_filename)
  cv2.imwrite(output_filename,output_mask)
'''

#Models output mask
#plt.imshow(output_mask)

#input image
#plt.imshow(img_plot)

#Preparing submission.npy file
#output_mask_path is the path which leads to the masks that you have generated for all the images (Kindly iterate through the previous cell and generate masks for all the images)
#make sure that your flattened numpy array consists of image pixels from image_0 to image_477
submission_file_path = os.path.join(root, "submission.npy")
main_array = []
for i in range(len(os.listdir(output_mask_path))):
  image_name = "image_" + str(i) + ".jpg"
  gt_mask = output_mask_path + image_name[:-3] + "png"
  print(gt_mask)
  gt_img = cv2.imread(gt_mask,cv2.COLOR_BGR2GRAY)
  flattened =gt_img.flatten()
  main_array.append(flattened)

main_array = np.asarray(main_array)
print(main_array.shape)
main_array_flat = np.reshape(main_array,(-1))
print(main_array_flat.shape)
print(type(main_array_flat))

with open(submission_file_path, 'wb') as f:
    np.save(f,main_array_flat)
