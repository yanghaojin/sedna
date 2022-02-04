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

# Contents are in utility script on Kaggle:
import os
# os.environ['AGG_IP']="x.x.x.x" # aggregation server ip
os.environ['AGG_PORT'] = "30363"  # aggregation server websocket port
os.environ['TRANSMITTER'] = "ws"

# worker parameters.
os.environ['MODEL_URL'] = "/home/data/model"
os.environ['NAMESPACE'] = "default"
os.environ['WORKER_NAME'] = "trainworker-nf8jw"
os.environ['JOB_NAME'] = "mooc-cifar100-resnet"
os.environ['MODEL_NAME'] = "mooc-cifar100-resnet-model"
os.environ['DATASET_NAME'] = "cifar100"

####################################################################################################################

import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')

## 3. 编写训练模型，导入训练数据，启动本地训练。

####################################################################################################################

# on Kaggle: add CIFAR as inputs, this code is not needed
# offline: download cifar manually
from torchvision.datasets import CIFAR100 as download_helper
download_helper(".", train=True, download=True)

####################################################################################################################
import numpy as np

def parallel_seeded_shuffle(seed, *args):
    # args is a list of all arrays that should be shuffled

    # TODO: use the default random number generator of numpy, set a seed and shuffle all given arrays in the exact same way
    # HINT: our function should work with any number of arrays
    ### BEGIN SOLUTION
    for array in args:
        fixed_seed_rng = np.random.default_rng(seed)
        fixed_seed_rng.shuffle(array)
    ### END SOLUTION
####################################################################################################################
# do not change the test seed
TEST_SEED = 42
for test_arrays in [(np.arange(10), np.arange(10)), (np.arange(10), np.arange(10), np.arange(10))]:
    print("Arrays before shuffling with fixed seed:", test_arrays)
    parallel_seeded_shuffle(TEST_SEED, *test_arrays)
    print("Arrays after shuffling with fixed seed: ", test_arrays)

    assert not np.allclose(test_arrays[0], np.arange(10)), "The arrays were not shuffled."
    for i in range(1, len(test_arrays)):
        assert np.allclose(test_arrays[0], test_arrays[i]), "Arrays 0 and {} are not shuffled the same way.".format(i)
    assert np.allclose(np.array([5, 6, 0, 7, 3, 2, 4, 9, 1, 8]), test_arrays[0]), "The arrays are not shuffled correctly based on TEST_SEED=42."
print("Test passed.")
####################################################################################################################
# TODO: Store your birthdate as one integer in the format YYYYMMDD in the variable `MY_RANDOM_SEED`.
#       This ensures that (almost) everyone in the course has a different random seed and thus different share of data.
### BEGIN SOLUTION
MY_RANDOM_SEED = 19800101
### END SOLUTION
data = np.arange(10)
labels = np.arange(10)
print("Arrays before shuffling with fixed seed:", data, labels)
parallel_seeded_shuffle(MY_RANDOM_SEED, data, labels)
print("Arrays after shuffling with fixed seed: ", data, labels)
assert np.allclose(data, labels), "Arrays are not shuffled the same way."
print("Test passed.")
####################################################################################################################
from pickle import load as pload

from torchvision import transforms as tt
from typing import Optional
from imgaug import augmenters as iaa
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class CIFAR100Partition(Dataset):
    def __init__(self, dataset_path: Path, image_transforms: tt.Compose, train: bool = True,
                 image_augmentations: Optional[iaa.Sequential] = None):
        super().__init__()
        data = pload(dataset_path.open("rb"), encoding="bytes")
        self.images = data[b"data"]
        self.labels = data[b"fine_labels"]

        self.image_transforms = image_transforms
        self.image_augmentations = image_augmentations

        if not train:
            # use all test data
            assert len(self.images) == len(self.labels), "Number of images and labels is not equal!"
            return

        # TODO: shuffle the training data (images and labels), then use only half of it
        ### BEGIN SOLUTION
        PARTITION_FRACTION = 1/2
        partition_size = int(len(self.images) * PARTITION_FRACTION)
        parallel_seeded_shuffle(MY_RANDOM_SEED, self.images, self.labels)
        self.images = self.images[:partition_size]
        self.labels = self.labels[:partition_size]
        ### END SOLUTION

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
cifar_train_set = CIFAR100Partition(Path("./cifar-100-python/train"), image_transformations, image_augmentations=train_augmentations)
cifar_test_set = CIFAR100Partition(Path("./cifar-100-python/test"), image_transformations, train=False)

print("Number of training samples:", len(cifar_train_set))
print("Number of test samples:    ", len(cifar_test_set))

assert len(cifar_train_set) == 25000, "We should use 25000 training samples (half of all data)."
assert len(cifar_test_set) == 10000, "We should use all 10000 test samples."

####################################################################################################################
import cifar100_resnets as models


class CustomDataset:
    def __init__(self, trainset, testset) -> None:
        self.customized = True
        self.trainset = trainset
        self.testset = testset


class Estimator:
    def __init__(self) -> None:
        self.model = self.build()
        self.pretrained = None
        self.saved = None
        self.use_client = "edge_ai_client"
        self.hyperparameters = {
            "type": "edge_ai_trainer",
            "rounds": 10,
            "target_accuracy": 0.5,
            "epochs": 2,
            "batch_size": 128,
            "optimizer": "SGD",
            "learning_rate": 0.1,
            "lr_schedule": "StepLR",
            "model_name": "cifar100_resnet",
            "momentum": 0.9,
            "weight_decay": 1e-4
        }

    @staticmethod
    def build():
        return models.resnet20(num_classes=100)


####################################################################################################################
from plato.trainers import registry as trainer_registry
from plato.trainers.basic import Trainer as BasicTrainer

from plato.clients import registry as client_registry
from plato.clients.simple import Client as SimpleClient


class EdgeAiTrainer(BasicTrainer):
    def train(self, trainset, sampler, cut_layer=None) -> float:
        logging.info("Edge AI trainer started training.")
        training_time = super().train(trainset, sampler, cut_layer=cut_layer)
        logging.info("Edge AI trainer finished training, saving the model.")
        self.save_model()
        return training_time


class EdgeAiClient(SimpleClient):
    pass


client_registry.registered_clients["edge_ai_client"] = EdgeAiClient
trainer_registry.registered_trainers["edge_ai_trainer"] = EdgeAiTrainer

####################################################################################################################

from sedna.algorithms.aggregation import FedAvgV2
from sedna.core.federated_learning import FederatedLearningV2


our_dataset = CustomDataset(trainset=cifar_train_set, testset=cifar_test_set)
# create an instance of our estimator
estimator = Estimator()
# our aggregation method
fedavg = FedAvgV2()
# get configured model transmitter
transmitter = FederatedLearningV2.get_transmitter_from_config()


fl = FederatedLearningV2(
    data=our_dataset,
    estimator=estimator,
    aggregation=fedavg,
    transmitter=transmitter)
fl.train()
