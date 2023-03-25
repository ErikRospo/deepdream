# deepdream

## Installation
To install the models, run setup.py
### Caffe
`sudo apt install caffe python3-caffe`
### CUDA
If using CUDA, don't install cafe with apt. Compile it using the links in the troubleshooting section.
copy the compiled caffe folder from `/build/python/caffe/` to the repository.
### PIP dependencies
`pip install -r requirements.txt`
### Extra Models
https://github.com/BVLC/caffe/wiki/Model-Zoo

## Running
`python3 main.py` 



## troubleshooting
https://gist.github.com/CyanLetter/a72e4be744aef1ed603a7d0df1632972
https://gist.github.com/ewnd9/3d3f688f8c6d3fe643f1

### can't compile PyCaffe
####  error: ‘CV_LOAD_IMAGE_COLOR’ was not declared in this scope
1. replace `CV_LOAD_IMAGE_COLOR` with `cv::IMREAD_COLOR`
2. replace `CV_LOAD_IMAGE_GRAYSCALE` with `cv::IMREAD_GRAYSCALE`
3. add `#include "opencv2/imgcodecs/imgcodecs.hpp"` to the top of all files with the error. 


### can't import caffe_pb2 or `TypeError: Couldn't build proto file into descriptor pool: duplicate file name caffe/proto/caffe.proto`  
`export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"`