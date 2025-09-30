import os

import numpy as np
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import json
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.models import inception_v3
from tqdm import tqdm


class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        root = "D:/WorkSpace/DeepRover-main/DeepRover-main/data/ImageNet"
        samples_dir = os.path.join(root, split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]

def evaluate(model, data_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 获取模型的预测输出
            outputs = model(inputs)

            # 获取预测的类别
            _, predicted = torch.max(outputs, 1)

            # 累加正确的预测数量
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f'Accuracy on ImageNet validation set: {accuracy:.2f}%')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = inception_v3(pretrained=True).to(device)
    # model = resnet50(pretrained=True).to(device)
    # model.eval()
    model = resnet50(pretrained=True)  # 不使用ImageNet预训练模型
    num_classes = 1000  # ImageNet验证集有1000个类别
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 修改最后一层

    save_path = './models/finetuned'
    # 加载已经微调的模型参数
    model.load_state_dict(torch.load(os.path.join(save_path, 'resnet50_finetuned.pt')))  # 假设微调模型已经保存为resnet50_finetuned.pth
    model = model.cuda()  # 使用GPU
    model.eval()  # 设置模型为评估模式


    # imagenet_size = 224
    imagenet_path = "D:/WorkSpace/DeepRover-main/DeepRover-main/imagenet_dataset"
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    torch.manual_seed(1)

    dataset = ImageNetKaggle(imagenet_path, "val", val_transform)
    imagenet_loader = DataLoader(
        dataset,
        batch_size=64,
        # batch_size=n_ex,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    evaluate(model, imagenet_loader, device)


