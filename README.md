# LightCNLib
A lightweight convolutionary neural network library with built in (Num/Cuda)py with built-in LeNet support

# Usage 
1. Navigate to LightCNLib
2. Run setup.py with python setup.py sdist bdist_wheel
3. Run pip install on the build files 

# About
This library will automatically detect cuda capable devices and choose either numpy or cudapy. It is comprised of layers similar to the pytorch library, see the layer.py abstract class.