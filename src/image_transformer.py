from torchvision import datasets, transforms
import torch
from PIL import Image

class image_transform:
    def __init__(self,image):
        self.image = image

    def image_transformer(self):
        test_image_transform = transforms.Compose([
                        transforms.Resize((300,300)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomRotation(degrees=35),
                        transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor(),
                        transforms.RandomAffine(degrees=0, translate=(0.2,0.2)),
                        transforms.Normalize(mean=[0.485],std=[0.229])
                        ])    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img = Image.open(self.image)
        img = test_image_transform(img)
        img = img.unsqueeze(0).to(device)
        return img