import torch
import torchvision
import cv2
import numpy as np

from classifier import Classifier


# eval the model, load state_dict
model = Classifier()
model.eval()

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
image = cv2.imread('dog.jpg')
image = cv2.resize(image, (224, 224))            # resize
image = image / 255.0
image = (image - imagenet_mean) / imagenet_std   # normalize
image = image.astype(np.float32)                 # float64 -> float32
image = image.transpose(2, 0, 1)                 # HWC -> CHW
image = np.ascontiguousarray(image)              # contiguous array memory
image = image[None, ...]                         # CHW -> 1CHW
image = torch.from_numpy(image)                  # numpy -> torch

with torch.no_grad():
    probability   = model(image)

predict_class = probability.argmax(dim=1).item()
confidence    = probability[0, predict_class]

labels = open("labels.txt",encoding='utf-8').readlines()   # '金鱼\n',.....
labels = [item.replace("\n", "") for item in labels]       # '金鱼',.....
print(predict_class, confidence, labels[predict_class])

# assert False
###### export onnx
input = torch.zeros(1,3,224,224)
torch.onnx.export(model,(input),'classifier.onnx',
                  input_names=['images'],
                  output_names=['output'],
                  dynamic_axes={
                      'images':{0:'batch'},
                      'output':{0:'batch'}
                  },opset_version=11)