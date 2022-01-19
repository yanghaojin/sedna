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

# TODO: set in environment variables (if working offline)
# TODO: set manually on Kaggle
# sys.argv.extend(['-i', '1'])  # client id
# os.environ['AGG_IP']="x.x.x.x" # aggregation server ip

# TODO: should allocated by sedna.
# client_id = ''.join(random.choice(string.digits) for i in range(10))
# # aggregation service parameters
# sys.argv.extend(['-i', client_id])  # client id
# os.environ['AGG_IP'] = "x.x.x.x"  # aggregation server ip
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

from sedna.algorithms.aggregation import FedAvgV2
from sedna.algorithms.client_choose import SimpleClientChoose
from sedna.common.config import Context
from sedna.core.federated_learning import FederatedLearningV2

import pickle

import numpy as np
import torchvision.transforms as tt

from typing import Type, Union

from imgaug import augmenters as iaa
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

import cifar_resnet as models

# on Kaggle: add CIFAR as inputs
# offline: download cifar manually
from torchvision.datasets import CIFAR100 as download_helper
download_helper(".", train=True, download=True)

from numpy.random import default_rng

# task to solve for participants:
def parallel_seeded_shuffle(seed, *args):
    for array in args:
        fixed_seed_rng = default_rng(seed)
        fixed_seed_rng.shuffle(array)


test_arrays = np.arange(10), np.arange(10)
parallel_seeded_shuffle(42, *test_arrays)
assert np.allclose(test_arrays[0], test_arrays[1]), "arrays are not shuffled the same way"
assert np.allclose(np.array([5, 6, 0, 7, 3, 2, 4, 9, 1, 8]), test_arrays[0]), "the seed is not used correctly"


# participants enter birthdate as their seed
my_birthday = 19800101
partition_fraction = 1/3  # we used one third of all cifar100 data ("randomly" chosen)


class CIFAR100(Dataset):
    def __init__(self, dataset_path: Path, image_transforms: tt.Compose, image_augmentations: Union[None, Type[iaa.Augmenter]] = None):
        super().__init__()
        data = pickle.load(dataset_path.open("rb"), encoding="bytes")
        self.images = data[b"data"]
        self.labels = data[b"fine_labels"]

        # possible task for students:
        parallel_seeded_shuffle(my_birthday, self.images, self.labels)
        partition_size = int(len(self.images) * partition_fraction)
        self.images = self.images[:partition_size]
        self.labels = self.labels[:partition_size]

        self.image_transforms = image_transforms
        self.image_augmentations = image_augmentations

        assert len(self.images) == len(self.labels), "Number of images and labels is not equal!"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple:
        image = self.images[index]
        label = self.labels[index]

        image = np.reshape(image, (3, 32, 32))
        image = np.transpose(image, (1, 2, 0))

        if self.image_augmentations is not None:
            image = self.image_augmentations.augment_image(image)
        image = self.image_transforms(Image.fromarray(image))
        return image, label


image_transformations = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(
        mean=(0.5074, 0.4867, 0.4411),
        std=(0.2011, 0.1987, 0.2025)
    )
])

train_augmentations = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.CropAndPad(px=(-4, 4), pad_mode="reflect")
])

# on Kaggle: "/kaggle/input/cifar100/train"
# on Kaggle: "/kaggle/input/cifar100/test"
train_dataset = CIFAR100(Path("./cifar-100-python/train"), image_transformations, train_augmentations)
test_dataset = CIFAR100(Path("./cifar-100-python/test"), image_transformations)

# os.environ['BACKEND_TYPE'] = 'TORCH'

simple_chooser = SimpleClientChoose(per_round=2)

# It has been determined that mistnet is required here.
fedavg = FedAvgV2()

# The function `get_transmitter_from_config()` returns an object instance.
transmitter = FederatedLearningV2.get_transmitter_from_config()


class CIFAR100Partition:
    def __init__(self, trainset, testset) -> None:
        self.customized = True
        self.trainset = trainset
        self.testset = testset


class Estimator:
    def __init__(self) -> None:
        self.model = self.build()
        self.pretrained = None
        self.saved = None
        self.hyperparameters = {
            "type": "basic",
            "rounds": int(Context.get_parameters("exit_round", 5)),
            "target_accuracy": 0.6,
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
        return models.resnet20(num_classes=100)


def main():
    data = CIFAR100Partition(trainset=train_dataset, testset=test_dataset)
    print("One third of train data:", len(train_dataset))
    print("One third of test data:", len(test_dataset))
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
