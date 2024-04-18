# Importing all the relevant libraries
import torch
import numpy as np
import torch.nn.parallel
import torchvision.transforms as transforms
from PIL import Image
import torchvision
import time
import os
from spiking_kitti import *
from sklearn.model_selection import train_test_split
from torch.utils.data import *
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Image path where images are stored
image_path = '/Users/Vijay/DCSNN/dataset/'


# Split the dataset into training and validation

def train_val_dataset(dataset, val_split=0.20):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


# Apply transformation and preprocessing to the dataset
transform = transforms.Compose([
    transforms.Resize((32, 32), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.557, 0.549, 0.5534])

])

dataset = ImageFolder(image_path, transform=transform)
print(len(dataset))
print(type(dataset))

datasets = train_val_dataset(dataset)

print(len(datasets['train']))
print(len(datasets['val']))

dataloaders = {x: DataLoader(datasets[x], batch_size=64, shuffle=True, drop_last=True) for
               x in ['train', 'val']}

# print(type(dataloaders))
x, y = next(iter(dataloaders['train']))
print(x.shape, y.shape)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# images, labels = next(iter(test_loader))
# print(images.shape, labels.shape)

# print(len(test_set))
# print(torch.__version__)

# Visualize the original test images
# constant for classes
classes = ('bicycle', 'bus', 'car', 'motorcycle', 'person',
           'truck')


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
        plt.show()
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


# get some random training images
#images, labels = next(iter(dataloaders['val']))
# print(images.shape, labels.shape)

# create grid of images
#img_grid = torchvision.utils.make_grid(images)

# show images
#matplotlib_imshow(img_grid, one_channel=True)

device = torch.device('cuda')
model = VGGModel(n_steps=10).to(device)

# Loading the .pt file

model.load_state_dict(
    torch.load('/Users/Vijay/DCSNN/state_dict_VGG_6_new_100ep.pt'))


# Set inference to evaluation mode

def eval_on_validation(model, loader):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            accuracy = 100. * float(correct) / float(total)
            # for image, label in zip(inputs, predicted):
            # for img0 in image.cpu().numpy():
            # plt.imshow(img0, cmap='binary')
            # plt.title('Predicted Image')
            # print(targets.cpu().numpy())
            # plt.savefig('pred.png')
            # plt.show()

    return accuracy


inference_accuracy = []

for i in range(5):
    tsince = time.time()
    infer_acc = eval_on_validation(model, dataloaders['val'])
    inference_accuracy.append(infer_acc)
    ttime_elapsed = time.time() - tsince
    print('test time elapsed {}s'.format(np.mean(ttime_elapsed)))
    print("Infer_accs: ", inference_accuracy[-1])
