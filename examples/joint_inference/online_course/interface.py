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

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import cifar100_resnets as models

LOG = logging.getLogger(__name__)
os.environ['BACKEND_TYPE'] = 'PYTORCH'


class CIFAR100Net(nn.Module):

    def __init__(self, model_type: str = "resnet18", temperature: int = 1):
        super().__init__()
        model_class = getattr(models, model_type)
        self.feature_extractor = model_class(num_classes=100)
        self.temperature = temperature

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        activations = self.feature_extractor(images)
        return activations / self.temperature


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def accuracy(predictions: torch.Tensor, labels: torch.Tensor, reduce_mean: bool = True) -> torch.Tensor:
    predicted_props = F.softmax(predictions, dim=1)
    predicted_classes = torch.argmax(predicted_props, dim=1)
    correct_predictions = torch.sum(predicted_classes == labels)
    if reduce_mean:
        return correct_predictions / len(labels)
    return correct_predictions


class Estimator:

    def __init__(self, node_type: str = "", **kwargs):
        """
        initialize logging configuration
        """
        self.model = None
        self.device = get_device()
        if node_type == "cloud":
            self.model = CIFAR100Net("resnet110")
        else:
            self.model = CIFAR100Net("resnet20")

    def load(self, model_url=""):
        checkpoint = torch.load(model_url, map_location=get_device())
        LOG.info(f"Load pytorch checkpoint {model_url} finsihed!")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        LOG.info("Load pytorch state dict finished")

        self.model = self.model.to(get_device())
        self.model.eval()

    def predict(self, data, **kwargs):
        image = to_device(data, self.device)
        predictions = self.model(image)
        props = F.softmax(predictions, dim=1)
        props_arr = props.detach().cpu().numpy().flatten().tolist()
        return props_arr
