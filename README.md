# DeepFake Detection

Repository for work on my Master's Thesis project. The goal is to examine state-of-the-art methods for face swap detection and look to develop something new and possibly better.

After conducting a review of some state-of-the-art models on the face manipulation detection task, I found the best performers to be ResNeXt101_32x8d and InceptionV3. The details can be found in my master's thesis. Based on this research, I proposed my own models - Reseption V1 and Reseption V2, which are smaller models based on a combination of residual connections and inception modules.

# Project Contents

* models/ - definitions for different types of pytorch models to be tested

* tools/ - module of tools/functions created for the purpose of the project

* outputs/ - various outputs such as logs, graphs, etc.

# ToDo:

* Classification mode

# Dependcies:

This implementation relies on PyTorch with cuda enabled to work efficiently. You can choose to install manually or with Conda. The following is my manual installation process determined through trial and error for my machine specifically.

Install CUDA and cuDNN:
* [CUDA (10.2)](https://developer.nvidia.com/cuda-toolkit-archive)
* [cuDNN (7.6.5 for CUDA 10.2 on Windows 10)](https://developer.nvidia.com/rdp/cudnn-download) according to [tensorflow compatibility tables for GPU usage](https://www.tensorflow.org/install/source)
* [cuDNN install guide](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

It is worth setting up a virtual environment so as not to mess with your default python packages.
* python -m venv <venv_path>
* <venv_path>/Scripts/Activate.ps1 (Activates the virtual environment in Windows PowerShell)

Set up an appropriaye version of PyTorch with torchvision, for me on Windows:
* pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html

Finally install other requirements:
* pip install -r requirements.txt