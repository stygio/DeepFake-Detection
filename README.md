# DeepFake Detection

Repository for work on my Master's Thesis project. The goal is to examine state-of-the-art methods for face swap detection and look to develop something new and possibly better.

# Project Contents

* models/ - definitions for different types of pytorch models to be tested

* tools/ - module of tools/functions created for the purpose of the project

* unit_tests/ - tests for various implemented functions, which can be run after navigating to the directory

* outputs/ - various outputs such as logs, graphs, etc.

# ToDo:

* Validation run

* Training entire network, not just FC layer

* Handling frames without detected faces more intelligently?

# Dependcies:

* pip3 install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html