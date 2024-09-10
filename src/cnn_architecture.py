import torch.nn as nn
import torch.nn.functional as F
import torch

class XrayCNN(nn.Module):
  def __init__(self,num_classes=2):
    super(XrayCNN,self).__init__()

    #convolution layers
    self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
    self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
    self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
    self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)

    #pooling layer
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    #connected layers
    self.fc1 = nn.Linear(512*9*9,1024)
    self.fc2 = nn.Linear(1024,512)
    self.fc3 = nn.Linear(512,num_classes)

    #dropout
    self.dropout = nn.Dropout(0.25)

    self.bn1 = nn.BatchNorm2d(32)
    self.bn2 = nn.BatchNorm2d(64)
    self.bn3 = nn.BatchNorm2d(128)
    self.bn4 = nn.BatchNorm2d(256)
    self.bn5 = nn.BatchNorm2d(512)

  def forward(self,x):
   x = self.pool(F.relu(self.bn1(self.conv1(x))))
   x = self.pool(F.relu(self.bn2(self.conv2(x))))
   x = self.pool(F.relu(self.bn3(self.conv3(x))))
   x = self.pool(F.relu(self.bn4(self.conv4(x))))
   x = self.pool(F.relu(self.bn5(self.conv5(x))))

   size = x.size()[1:]  
   num_features = 1
   for s in size:
       num_features *= s

   x = x.view(-1,num_features)

   x = F.relu(self.fc1(x))
   x = self.dropout(x)
   x = F.relu(self.fc2(x))
   x = self.dropout(x)

   x = self.fc3(x)

   return x