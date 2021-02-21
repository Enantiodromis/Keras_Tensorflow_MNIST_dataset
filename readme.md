# Deep Discriminant Nueral Network (MNIST dataset)

## Tools & Tech:
- Python 
- Keras
- TensorFlow

## Building and training a deep nueral network for classifying the MNIST dataset. 
The MNIST dataset consists of 60,000 28x28 pixel images of handwritten digits. 

The set of images in the [MNIST database](http://yann.lecun.com/exdb/mnist/) is a combination of two of NIST's databases: Special Database 1 and Special Database 3. Special Database 1 and Special Database 3 consist of digits written by high school students and employees of the United States Census Bureau.

## Architecture Approach
The chosen implemented archictecture is a variation of the [LeNet-5 architecture](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).

### Original LeNet 
LeNet5 contains the basic modules of deep learning:
1. Convolution layer
2. Pooling layer
3. Full link layer

LeNet5 is comprised of 7 layers:

Layer   | Layer Type     | Activation
:-----: | :----:         | :-----:
Input   | Image          | -
1       | Convolution    | tanh
2       | Average Pooling| tanh
3       | Convolution    | tanh
4       | Average Pooling| tanh
5       | Convolution    | tanh
6       | FC             | tanh
Output  | FC             | softmax

<br>
### Variations on LeNet5

#### - __ReLU-softmax__
ReLU-softmax was used inplace of tanh-softmax.
- After benchmarking:
    - ReLU-softmax
    - sigmoid sigmoid
    - tanh-softmax

ReLU-softmax returned the best performance on training and test data, and was therefore implemented
<br>

#### -  __Max Pooling inplace of Average Pooling__
Instead of using average pooling, max pooling was implemented in order to reduce computation cost. The background of the MNIST dataset is black, max pooling [performs better](https://iq.opengenus.org/maxpool-vs-avgpool/) than average pooling for darker backgrounds. 
<br>

#### - __Batch Normalisation__
[Batch Normalisation](https://www.baeldung.com/cs/batch-normalization-cnn) was used between the layers of the network in an effort to speed up training and use higher learning rates. 
<br>

#### - __Dropout__
In an effort to reduce interdependent learning amongst nuerons and minimize overfitting, [Dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5) was used within the network.
<br>

#### - __Additional dense layer__
A additional dense layer, with activation function ReLU and output size of 256 was used.
<br>

__The resulting architecture:__
Layer   | Layer Type     | Activation
:-----: | :----:         | :-----:
Input   | Image          | -
1       | Convolution    | relu
2       | Convolution    | relu
3       | BatchNormalization | relu
4       | Max Pooling  | -
5       | Dropout    | -
6       | Convolution    | relu
7       | Convolution    | relu
8       | BatchNormalization | relu
9       | Max Pooling | -
10       | Dropout    | -
11      | FC             | relu
12      | BatchNormalization | relu
13      | FC             | relu
14      | BatchNormalization | relu
15      | FC             | relu
16      | BatchNormalization | relu
17      | Dropout    | -
Output  | FC             | softmax
