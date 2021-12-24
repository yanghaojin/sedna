# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## 1. 配置训练节点ID、聚合服务ip&port、数据集路径、训练超参数
import os
import random
import string
import sys

# TODO: should allocated by sedna.

# client_id = ''.join(random.choice(string.digits) for i in range(10))
# # aggregation service parameters
# sys.argv.extend(['-i', client_id])  # client id

# os.environ['AGG_IP'] = "49.0.251.158"  # aggregation server ip
os.environ['AGG_PORT'] = "30363"  # aggregation server websocket port
os.environ['TRANSMITTER'] = "ws"
# os.environ['PARTICIPANTS_COUNT']="2"


# datasets parameters
os.environ['TRAIN_DATASET_URL'] = "/kaggle/input/magnetic-tile-defect-datasets/1.txt"

# training parameters
os.environ['learning_rate'] = "0.001"
os.environ['batch_size'] = "32"
os.environ['epochs'] = "2"

# worker parameters.
# os.environ['DATA_PATH_PREFIX']="/home/data"
# os.environ['LC_SERVER']="http://localhost:9100"
# os.environ['HOSTNAME']="edge1"  # client name
os.environ['MODEL_URL'] = "/home/data/model"
os.environ['NAMESPACE'] = "default"
os.environ['WORKER_NAME'] = "trainworker-nf8jw"
os.environ['JOB_NAME'] = "surface-defect-detection"
os.environ['MODEL_NAME'] = "surface-defect-detection-model"
os.environ['DATASET_NAME'] = "edge1-surface-defect-detection-dataset"

## 3. 编写训练模型，导入训练数据，启动本地训练。


import os
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from sedna.algorithms.aggregation import FedAvgV2
from sedna.algorithms.client_choose import SimpleClientChoose
from sedna.common.config import BaseConfig
from sedna.common.config import Context
from sedna.core.federated_learning import FederatedLearningV2

# os.environ['BACKEND_TYPE'] = 'TORCH'

simple_chooser = SimpleClientChoose(per_round=2)

# It has been determined that mistnet is required here.
fedavg = FedAvgV2()

# The function `get_transmitter_from_config()` returns an object instance.
transmitter = FederatedLearningV2.get_transmitter_from_config()


class SddDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.images = x
        self.labels = y

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class myDataset:
    def __init__(self, trainset=None, testset=None) -> None:
        self.customized = True
        self.trainset = SddDataset(trainset[0], trainset[1])
        self.testset = SddDataset(testset[0], testset[1])


class Estimator:
    def __init__(self) -> None:
        self.model = self.build()
        self.pretrained = None
        self.saved = None
        self.hyperparameters = {
            "type": "basic",
            "rounds": int(Context.get_parameters("exit_round", 5)),
            "target_accuracy": 0.97,
            "epochs": int(Context.get_parameters("epochs", 5)),
            "batch_size": int(Context.get_parameters("batch_size", 32)),
            "optimizer": "SGD",
            "learning_rate": float(Context.get_parameters("learning_rate", 0.01)),
            # The machine learning model
            "model_name": "sdd_model",
            "momentum": 0.9,
            "weight_decay": 0.0
        }

    @staticmethod
    def build():
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(6272, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2))

        return model


def readFromTxt(path):
    data_x = []
    data_y = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            x, y = image_process(line)
            data_x.append(x)
            data_y.append(y)
    return data_x, data_y


def image_process(line):
    file_path, label = line.split(',')
    original_dataset_url = (
            BaseConfig.original_dataset_url or BaseConfig.train_dataset_url
    )
    root_path = os.path.dirname(original_dataset_url)
    file_path = os.path.join(root_path, file_path)
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.PILToTensor()])
    x = Image.open(file_path)
    x = transform(x) / 255.
    y = int(label)

    return [x, y]


def main():
    train_dataset_url = BaseConfig.train_dataset_url
    # we have same data in the trainset and testset
    test_dataset_url = BaseConfig.train_dataset_url

    #     train_data = TxtDataParse(data_type="train", func=image_process)
    #     train_data.parse(train_dataset_url)
    train_data = readFromTxt(train_dataset_url)
    data = myDataset(trainset=train_data, testset=train_data)
    estimator = Estimator()

    fl = FederatedLearningV2(
        data=data,
        estimator=estimator,
        aggregation=fedavg,
        transmitter=transmitter)

    fl.register()
    fl.train()


if __name__ == '__main__':
    main()
