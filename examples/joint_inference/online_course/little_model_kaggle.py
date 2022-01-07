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

import time
import logging
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Type, List, Union
from imgaug import augmenters as iaa
from pathlib import Path
from PIL import Image

import torchvision.transforms as tt

from sedna.core.joint_inference import JointInference
from interface import Estimator, get_device, accuracy

LOG = logging.getLogger(__name__)


class CIFAR100(Dataset):

    def __init__(self, dataset_path: Path, image_transforms: tt.Compose,
                 image_augmentations: Union[None, Type[iaa.Augmenter]] = None,
                 length : int = 10000):
        super().__init__()
        data = pickle.load(dataset_path.open("rb"), encoding="bytes")
        self.images = data[b"data"][:length]
        self.labels = data[b"fine_labels"][:length]

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


def main():

    inference_instance = JointInference(
        estimator=Estimator,
        hard_example_mining={
            "method": "CrossEntropy",
            "param": {
                "threshold_cross_entropy": 0.85
            }
        }
    )

    # testing data
    SAMPLE_NUM = 512
    test_dataset = CIFAR100(Path("/kaggle/input/cifar100/test"), image_transformations, length=SAMPLE_NUM)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # collaborative testing
    num_remote_samples = 0
    duration = .0
    num_correct_predictions = 0
    for image, label in test_data_loader:
        start = time.time()
        is_hard_example, final_result, edge_result, cloud_result = (
            inference_instance.inference(image)
        )
        if get_device() == torch.device("cuda"):
            torch.cuda.current_stream().synchronize()
        duration += (time.time()-start)*1000.0

        if is_hard_example: num_remote_samples += 1

        torch_result = torch.from_numpy(np.asarray(final_result).reshape(1, -1))
        num_correct_predictions += float(accuracy(torch_result, label, reduce_mean=False).item())

    print("collaborative acc: {:.2f}, processed sample number (teacher/student): {}/{},  avg_inference_time: {:.3f} ms, total time: {:.3f} seconds.".format(
        num_correct_predictions/SAMPLE_NUM, num_remote_samples, SAMPLE_NUM-num_remote_samples, duration/SAMPLE_NUM, duration/1000.0))


if __name__ == '__main__':
    main()
