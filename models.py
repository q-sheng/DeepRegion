import torch
import numpy as np
import math
import utils
from torchvision import models as torch_models
from torch.nn import DataParallel

from model_definitions.trades_model.wideresnet import WideResNet as WideResNet_trades
from model_definitions.trades_model.wideresnet import SmallCNN
from model_definitions.trades_model.vgg16 import VGG16
from model_definitions.trades_model.LeNet import LeNet1, LeNet5, LeNet5_SVHN

class Model:
    def __init__(self, batch_size, gpu_memory):
        self.batch_size = batch_size
        self.gpu_memory = gpu_memory

    def predict(self, x):
        raise NotImplementedError('custom model should implement this method')

    def objective(self, y, logits):
        """
        the objective function calculates logits(i)-logits(j),
        where i is the correct class, i !=j, and j is the class with largest logits value.
        """
        logits_correct = (logits * y).sum(1, keepdims=True)
        logits_difference = logits_correct - logits
        logits_difference[y] = np.inf # 将正确类别的差值设为无穷大
        objective_val = logits_difference.min(1, keepdims=True) # 计算最小差值

        return objective_val.flatten()

class CustomModel_Trades(Model):
    """
    We use this class for models trained by TRADES
    """

    def __init__(self, model_name, batch_size, gpu_memory):
        super().__init__(batch_size, gpu_memory)
        if model_name in ['svhn-defended-trades','svhn-undefended-trades']:
            self.mean = np.reshape([0.4376821, 0.4437697, 0.47280442], [1, 3, 1, 1])
            self.std = np.reshape([0.19803012, 0.20101562, 0.19703614], [1, 3, 1, 1])
        else:
            if model_name in ['cifar10-undefended-trades','cifar10-defended-trades', 'cifar10-vgg16']:
                self.mean = np.reshape([0.49139968, 0.48215841, 0.44653091], [1, 3, 1, 1])
                self.std = np.reshape([0.24703223, 0.24348513, 0.26158784], [1, 3, 1, 1])
            else:
                if model_name in ['imagenet_resnet', 'imagenet_inception']:
                    self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
                    self.std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])
        self.mean, self.std = self.mean.astype(np.float32), self.std.astype(np.float32)

        if model_name in ['cifar10-defended-trades','svhn-defended-trades','cifar10-undefended-trades', 'svhn-undefended-trades', 'cifar10-vgg16']:
            model = model_definitions_dictionary[model_name](34, 10, 10)
            model = model.cuda()
            checkpoint = torch.load(model_directory_dictionary[model_name] + '.pt')
            model.load_state_dict(checkpoint)
        else:
            if model_name in ['imagenet_resnet', 'imagenet_inception']:
                model = model_definitions_dictionary[model_name](weights="DEFAULT")
                model = DataParallel(model.cuda())

        model.float()
        model.eval()
        self.model = model
        self.model_name = model_name

    def predict(self, x):
        x = x.astype(np.float32)

        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []
        with torch.no_grad():
            for i in range(n_batches):
                x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
                x_batch_torch = torch.as_tensor(x_batch, device=torch.device('cuda'))
                logits = self.model(x_batch_torch).cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits

class CustomModel(Model):
    """
    We use this class for models trained without TRADES
    """

    def __init__(self, model_name, batch_size, gpu_memory):
        super().__init__(batch_size, gpu_memory)
        if model_name in ['svhn-defended-trades','svhn-undefended-trades', 'svhn-lenet5']:
            self.mean = np.reshape([0.4376821, 0.4437697, 0.47280442], [1, 3, 1, 1])
            self.std = np.reshape([0.19803012, 0.20101562, 0.19703614], [1, 3, 1, 1])
        else:
            if model_name in ['cifar10-undefended-trades','cifar10-defended-trades', 'cifar10-vgg16']:
                self.mean = np.reshape([0.49139968, 0.48215841, 0.44653091], [1, 3, 1, 1])
                self.std = np.reshape([0.24703223, 0.24348513, 0.26158784], [1, 3, 1, 1])
            else:
                if model_name in ['imagenet_resnet', 'imagenet_inception']:
                    self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
                    self.std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])
                else:
                    if model_name in ['mnist-defended-trades','mnist-lenet1','mnist-lenet5']:
                        self.mean = np.reshape([0.13251461], [1, 1, 1, 1])
                        self.std = np.reshape([0.31048027], [1, 1, 1, 1])

        self.mean, self.std = self.mean.astype(np.float32), self.std.astype(np.float32)

        if model_name in ['cifar10-defended-trades','svhn-defended-trades','cifar10-undefended-trades',
                          'svhn-undefended-trades', 'cifar10-vgg16','mnist-defended-trades','mnist-lenet1','mnist-lenet5', 'svhn-lenet5']:
            # model = model_definitions_dictionary[model_name](34, 10, 10)
            model = model_definitions_dictionary[model_name]()
            model = model.cuda()
            checkpoint = torch.load(model_directory_dictionary[model_name] + '.pt')
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
        else:
            if model_name in ['imagenet_resnet', 'imagenet_inception']:
                model = model_definitions_dictionary[model_name](weights="DEFAULT")
                model = DataParallel(model.cuda())

        model.float()
        model.eval()
        self.model = model
        self.model_name = model_name

    def predict(self, x):
        x = (x - self.mean) / self.std
        # x = x / 255.0
        x = x.astype(np.float32)

        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []
        with torch.no_grad():
            for i in range(n_batches):
                x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
                x_batch_torch = torch.as_tensor(x_batch, device=torch.device('cuda'))
                logits = self.model(x_batch_torch).cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits


model_directory_dictionary = {
                     'cifar10-defended-trades': 'models/cifar10/cifar10-defended-trades',
                     'cifar10-undefended-trades': 'models/cifar10/cifar10-undefended-trades',
                     'cifar10-vgg16': 'models/cifar10/cifar10-vgg16',
                     'svhn-defended-trades': 'models/svhn/svhn-defended-trades',
                     'svhn-undefended-trades': 'models/svhn/svhn-undefended-trades',
                     'svhn-lenet5': 'models/svhn/svhn-lenet5',
                     'mnist-defended-trades': 'models/mnist/mnist-defended-trades',
                     'mnist-lenet1': 'models/mnist/mnist-lenet1',
                     'mnist-lenet5': 'models/mnist/mnist-lenet5',
                   }

model_definitions_dictionary = {
                     'mnist-defended-trades': SmallCNN,
                     'mnist-lenet1': LeNet1,
                     'mnist-lenet5': LeNet5,
                     'cifar10-defended-trades': WideResNet_trades,
                     'cifar10-undefended-trades': WideResNet_trades,
                     'cifar10-vgg16': VGG16,
                     'svhn-defended-trades': WideResNet_trades,
                     'svhn-undefended-trades': WideResNet_trades,
                     'svhn-lenet5': LeNet5_SVHN,
                     'imagenet_resnet': torch_models.resnet50,
                     'imagenet_inception': torch_models.inception_v3,
                    }

models_defined = list(model_definitions_dictionary.keys())