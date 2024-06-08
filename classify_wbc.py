import torch
import torchvision
from torchvision.io import read_image  
from torchvision.transforms import v2
from torch import nn    

import os
import argparse
import pdb

# Defining class mapping
# creating label tensors
class_names = ['Basophil', 'Neutrophil', 'Monocyte', 'Eosinophil', 'Lymphocyte']

class_map = {}
for i, class_name in enumerate(class_names):
    class_map[i] = class_name


# Define transformation as per resnet specs
resnet_transform = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])

# custom final layer for my modified resnet18, needed to load the model
class FinalBlock(nn.Module):
    def __init__(self):
        super(FinalBlock, self).__init__()
        
        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        x = self.fc(x)
        return x


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    return device
    

def load_image(image_path, device):
    assert os.path.exists(image_path), f"Image file not found at {image_path}"
    img_as_tensor = read_image(image_path)
    img_as_tensor = resnet_transform(img_as_tensor)  
    img_as_tensor = img_as_tensor.to(device)  # move to given device
    img_as_tensor = img_as_tensor.unsqueeze(0)  # add batch dimension
    return img_as_tensor
        

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    args = parser.parse_args()
    assert os.path.exists(args.image_path), f"Image file not found at {args.image_path}"
    image_path = args.image_path

    # get device 
    device = get_device()
    print(f"Using device: {device}")

    # load image
    image_as_tensor = load_image(image_path, device)
    assert image_as_tensor.shape == (1, 3, 224, 224), f"Expected shape (1, 3, 224, 224) but got {image_as_tensor.shape}"

    # load the model
    model_path = 'resnet18_epoch_47_time_04_06_2024_19_42_45' # path to the best-performing model
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    # make prediction
    with torch.no_grad():
        pred = model(image_as_tensor)
        pred_softmax = torch.softmax(pred, dim=1)
        print("Model output: ", {class_map[key]: pred_softmax[0][key].item() for i, key in enumerate(class_map.keys())})
        pred_class = torch.argmax(pred_softmax).item()
        pred_class_name = class_map[pred_class]
        print(f"Interpretation: the model is {pred_softmax[0][pred_class].item()*100:.2f}% confident that the image is of a(n) {pred_class_name}")

        

if __name__ == '__main__':
    main()