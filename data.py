import os
import random

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from imagenet_data import ImageNetKaggle
import scipy.io as sio
import torchvision

def random_select(x_test_list, y_test_list, n_ex_to_load):

    np.random.seed(0)

    index_list = []
    for index in range(0, len(x_test_list)):
        index_list.append(index)

    if n_ex_to_load > len(x_test_list):
        n_ex_to_load = len(x_test_list)

    image_num = 0
    x_test_select = []
    y_test_select = []
    while image_num < n_ex_to_load:
        index = np.random.randint(0, len(index_list))
        sel_index = index_list[index]
        x_test_select.append(x_test_list[sel_index])
        y_test_select.append(y_test_list[sel_index])
        index_list.remove(sel_index)
        image_num = image_num + 1

    x_test = np.array(x_test_select)
    y_test = np.array(y_test_select)
    return x_test, y_test

def load_cifar10_trades(n_ex_to_load):
    X_data = np.load('./data/CIFAR10/cifar10_X.npy')
    Y_data = np.load('./data/CIFAR10/cifar10_Y.npy')
    print(X_data.shape)
    X_data = np.transpose(X_data, axes=[0, 3, 1, 2])

    # x_test_list = np.ndarray.tolist(X_data)
    # y_test_list = np.ndarray.tolist(Y_data)
    #
    # x_test, y_test = random_select(x_test_list, y_test_list, n_ex_to_load)
    indices_per_class = {}
    for i, label in enumerate(Y_data):
        if label not in indices_per_class:
            indices_per_class[label] = []
        indices_per_class[label].append(i)

    n_ex_to_load = min(n_ex_to_load, len(indices_per_class))

    # 从每个类别中随机选择一个样本
    selected_indices = []
    for label in indices_per_class:
        if len(selected_indices) >= n_ex_to_load:
            break
        selected_index = np.random.choice(indices_per_class[label])
        selected_indices.append(selected_index)

    # 根据选择的索引获取样本
    x_test = X_data[selected_indices].tolist()
    y_test = Y_data[selected_indices].tolist()
    return x_test, y_test

def load_mnist_trades(n_ex_to_load):
    X_data = np.load('./data/MNIST/mnist_X.npy')
    Y_data = np.load('./data/MNIST/mnist_Y.npy')
    print(X_data.shape)
    X_data = np.transpose(X_data, axes=[0, 3, 1, 2])

    x_test_list = np.ndarray.tolist(X_data)
    y_test_list = np.ndarray.tolist(Y_data)

    x_test, y_test = random_select(x_test_list, y_test_list, n_ex_to_load)

    return x_test, y_test

# def load_mnist(n_ex_to_load):
#     test_dataset = torchvision.datasets.MNIST(root='.\data\MNIST',
#                                                train=False,
#                                                transform=transforms.ToTensor(),
#                                                download=True)
#     x_test_list = []
#     y_test_list = []
#     for image, label in test_dataset:
#         image_array = np.array(image)
#         x_test_list.append(image_array)
#         y_test_list.append(label)
#
#     x_test, y_test = random_select(x_test_list, y_test_list, n_ex_to_load)
#     return x_test, y_test

# def load_cifar10(n_ex_to_load):
#     test_dataset = torchvision.datasets.CIFAR10(root='.\data\CIFAR10',
#                                                  train=False,
#                                                  transform=transforms.ToTensor(),
#                                                  download=True)
#     x_test_list = []
#     y_test_list = []
#     for image, label in test_dataset:
#         image_array = np.array(image)
#         x_test_list.append(image_array)
#         y_test_list.append(label)
#
#     x_test, y_test = random_select(x_test_list, y_test_list, n_ex_to_load)
#     return x_test, y_test

def load_cifar10(n_ex_to_load):
    # 加载 CIFAR-10 测试集
    test_dataset = torchvision.datasets.CIFAR10(root='./data/CIFAR10',
                                                train=False,
                                                transform=transforms.ToTensor(),
                                                download=True)

    selected_images = {}  # 存储每个类别随机选中的图片

    # 遍历数据集，直到每个类别选到一张图片
    for image, label in test_dataset:
        if label not in selected_images:  # 如果该类别还没有选中的图片
            selected_images[label] = image.numpy()  # 转换为 NumPy 数组存储
        if len(selected_images) == n_ex_to_load:  # 10 个类别全部选中，停止遍历
            break

            # 按类别顺序返回 10 张图片和对应标签
    x_test = np.array([selected_images[i] for i in range(n_ex_to_load)])  # 按类别顺序组织
    y_test = np.array(list(selected_images.keys()))

    return x_test, y_test

def load_svhn(n_ex_to_load):
    test_dataset = torchvision.datasets.SVHN(root='.\data\SVHN',
                                              split='test',
                                              transform=transforms.ToTensor(),
                                              download=True)
    x_test_list = []
    y_test_list = []

    for image, label in test_dataset:
        image_array = np.array(image)
        x_test_list.append(image_array)
        y_test_list.append(label)

    x_test, y_test = random_select(x_test_list, y_test_list, n_ex_to_load)
    return x_test, y_test

# def load_svhn(n_ex_to_load):
#     # 加载 CIFAR-10 测试集
#     test_dataset = torchvision.datasets.SVHN(root='./data/SVHN',
#                                                 split='test',
#                                                 transform=transforms.ToTensor(),
#                                                 download=True)
#
#     selected_images = {}  # 存储每个类别随机选中的图片
#
#     # 遍历数据集，直到每个类别选到一张图片
#     for image, label in test_dataset:
#         if label not in selected_images:  # 如果该类别还没有选中的图片
#             selected_images[label] = image.numpy()  # 转换为 NumPy 数组存储
#         if len(selected_images) == n_ex_to_load:  # 10 个类别全部选中，停止遍历
#             break
#
#             # 按类别顺序返回 10 张图片和对应标签
#     x_test = np.array([selected_images[i] for i in range(n_ex_to_load)])  # 按类别顺序组织
#     y_test = np.array(list(selected_images.keys()))
#
#     return x_test, y_test

def load_mnist(n_ex_to_load):
    test_dataset = torchvision.datasets.MNIST(root='.\data\MNIST',
                                               train=False,
                                               transform=transforms.ToTensor(),
                                               download=True)
    # x_test_list = []
    # y_test_list = []
    #
    # for image, label in test_dataset:
    #     image_array = np.array(image)
    #     x_test_list.append(image_array)
    #     y_test_list.append(label)
    #
    # x_test, y_test = random_select(x_test_list, y_test_list, n_ex_to_load)
    selected_images = {}  # 存储每个类别随机选中的图片

    # 遍历数据集，直到每个类别选到一张图片
    for image, label in test_dataset:
        if label not in selected_images:  # 如果该类别还没有选中的图片
            selected_images[label] = image.numpy()  # 转换为 NumPy 数组存储
        if len(selected_images) == n_ex_to_load:  # 10 个类别全部选中，停止遍历
            break

            # 按类别顺序返回 10 张图片和对应标签
    x_test = np.array([selected_images[i] for i in range(n_ex_to_load)])  # 按类别顺序组织
    y_test = np.array(list(selected_images.keys()))

    return x_test, y_test
def load_imagenet(n_ex, size=224):
    imagenet_size = size
    imagenet_path = "imagenet_dataset"
    val_transform = transforms.Compose(
        [
            transforms.Resize(imagenet_size),
            transforms.CenterCrop(imagenet_size),
            transforms.ToTensor()
        ]
    )

    torch.manual_seed(1)

    dataset = ImageNetKaggle(imagenet_path, "val", val_transform)

    imagenet_loader = DataLoader(
        dataset,
        batch_size=int(n_ex / 1.5),
        # batch_size=n_ex,
        num_workers=4,
        shuffle=True,
        drop_last=False,
        pin_memory=True
    )

    x_test, y_test = next(iter(imagenet_loader))

    return np.array(x_test, dtype=np.float32), np.array(y_test)

batch_size_dictionary = {'cifar10': 512,
                         'svhn': 512,
                         'imagenet': 32,
                         'mnist': 512,
}
region_size_dictionary = {'cifar10': 8,#256+512=768Q
                          'mnist': 4,#49+24*4=145Q
                          'svhn': 8,#256+512=768Q
                          'imagenet': 32,#196*(1+2+4)=1372Q 588Q
}
region_stop_dictionary = {'cifar10': 2,
                          'mnist': 1,
                          'svhn': 2,
                          'imagenet': 8,
}




