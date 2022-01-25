## Modified Example for Federated Learning of CIFAR 100

This example should later be able to run on Kaggle. For testing the script we can run from a local machine.

Installation required (run commands from the git root directory):
```bash
pip install -r lib/requirements.txt
pip install torch torchvision imgaug matplotlib
```

Add the `lib` folder to the python path.

To run the script, pass `AGG_IP` as environment variable and client id (later manully entered in Kaggle):
```bash
cd examples/federated_learning/online_course
AGG_IP="x.x.x.x" python pytorch/train_cifar_on_kaggle.py -i 1
```

The script should download the cifar dataset and try connecting to the server.
The hyperparameters are maybe not perfect for CIFAR100.
With GPU on Kaggle we can try to increase the `batch_size` to 128.
We usually use Adam with learning rate `0.001`.

### Changes for Kaggle

To run on Kaggle we need to follow the Kaggle notebook example and install with:
```python
!pip install -r /kaggle/input/kubeedge-sedna/lib/requirements.txt
import sys
import logging
sys.path.append('/kaggle/input/kubeedge-sedna/lib')
# !python /kaggle/input/kubeedge-sedna/examples/federated_learning/online_course/init.py
```

Also we need to adapt the script, check comments with "kaggle" in them.
