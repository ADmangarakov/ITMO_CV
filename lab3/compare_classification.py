# -*- coding: utf-8 -*-
"""compare_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vk9oqxR2Xo7vyQiTqoTmoMQWBsMR7Eo8
"""


import torch
import torchvision
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
resnet50 = model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

resnet50.eval().to(device)
vgg16.eval()
alexnet.eval()

"""Prepare data

"""

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Download ImageNet labels
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

from os import listdir
from os.path import isfile, join
MYPATH = 'data/'
data_images = [f for f in listdir(MYPATH) if isfile(join(MYPATH, f))]

from os import listdir
from os.path import isfile, join
MYPATH = 'data/'
lables = [f for f in listdir(MYPATH) if not isfile(join(MYPATH, f))]
lables

data_images = []
for l in lables:
  for d in listdir(join(MYPATH, l)):
    data_images.append(join(MYPATH, l, d))

acc_5 = [0, 0, 0]
acc_1 = [0, 0, 0]
for i in data_images:
  input_image = Image.open(i)
  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  with torch.no_grad():
    vgg_output = vgg16(input_batch)
    alexnet_output = alexnet(input_batch)
    resnet50_output = resnet50(input_batch)

  # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
  model_names = ['vgg', 'alexnet', 'resnet50']
  probabilities = []
  probabilities.append(torch.nn.functional.softmax(vgg_output[0], dim=0))
  probabilities.append(torch.nn.functional.softmax(alexnet_output[0], dim=0))
  probabilities.append(torch.nn.functional.softmax(resnet50_output[0], dim=0))
  # Read the categories

  with open("imagenet_classes.txt", "r") as f:
      categories = [s.strip() for s in f.readlines()]
  # Show top categories per image
  plt.imshow(input_image)
  plt.show()
  for p in range(len(probabilities)):
    print('\n' + model_names[p] + "| Actual: " + i.split('/')[1] + '\nPredicted:')
    top5_prob, top5_catid = torch.topk(probabilities[p], 5)
    top1_prob, top1_catid = torch.topk(probabilities[p], 1)
    for k in range(top5_prob.size(0)):
        print(categories[top5_catid[k]], top5_prob[k].item())
        if categories[top5_catid[k]] == i.split('/')[1]:
          acc_5[p] +=1
    if categories[top1_catid] == i.split('/')[1]:
      acc_1[p] +=1
i = 0
for n in model_names:
  print(n + 'top 1 accuracy = ' + str(acc_1[i]/len(data_images)))
  print(n + 'top 5 accuracy = ' + str(acc_5[i]/len(data_images)))
  i+=1


