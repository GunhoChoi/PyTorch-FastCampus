# PyTorch FastCampus

PyTorch로 시작하는 딥러닝 입문 CAMP (www.fastcampus.co.kr/data_camp_pytorch/) 1기 강의자료

<p align="center">
<img src="./data_camp_pytorch.png" width="60%">
</p>

Requirements
-------------------------
- python 3.6
- Pytorch (http://pytorch.org/)
- Numpy
- matplotlib


Optional
--------------------------
- visdom (https://github.com/facebookresearch/visdom)

설치방법 [PyTorch & Jupyter Notebook](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/01_DL%26Pytorch/PyTorch_AWS%EC%84%A4%EC%B9%98.pdf)
-------------------------------------
- AWS p2.xlarge(Tesla K80 GPU)
- CUDA 8.0
- CuDNN 5.1
- Anaconda
- PyTorch
- Jupyter Notebook

강의자료
--------------------------
## 1강 [Deep Learning & PyTorch](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/01_DL%26Pytorch/%EB%94%A5%EB%9F%AC%EB%8B%9D%26%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98.pdf)

1) [프레임워크 비교](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/01_DL%26Pytorch/codes/0_Framework_Comparison.ipynb)

2) [파이토치 기본 사용법](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/01_DL%26Pytorch/codes/1_pytorch_tensor_basic.ipynb)

## 2강 [Linear Regression & Neural Network](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/02_Regression%26NN/Regression%26NN.pdf)

1) [Automatic Gradient Calculation](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/02_Regression%26NN/0_Linear_code/0_Variable_Autograd.ipynb)

2) [시각화 툴 Visdom 소개](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/02_Regression%26NN/Visdom_Tutorial.ipynb)

3) [선형회귀모델](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/02_Regression%26NN/0_Linear_code/1_linear_regression.ipynb)

4) [선형회귀모델의 한계](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/02_Regression%26NN/0_Linear_code/2_linear_nonlinear.ipynb)

5) [인공신경망 모델 - 2차함수근사](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/02_Regression%26NN/1_NN_code/1d_data/0_neural_quadratic.ipynb)

6) [인공신경망 모델 - 3차함수근사](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/02_Regression%26NN/1_NN_code/1d_data/1_neural_cubic.ipynb)

7) [인공신경망 모델 - 2D데이터](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/02_Regression%26NN/1_NN_code/2d_data/neural_2d.ipynb)

## 3강 [Convolutional Neural Network - Basic](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/03_CNN_Basics/%5B%ED%8C%A8%EC%8A%A4%ED%8A%B8%EC%BA%A0%ED%8D%BC%EC%8A%A4%5D%20PyTorch%EB%A1%9C%20%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94%20%EB%94%A5%EB%9F%AC%EB%8B%9D%20%EC%9E%85%EB%AC%B8%20CAMP_3%ED%9A%8C%EC%B0%A8.pdf)

1) [CNN 기본 모듈](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/03_CNN_Basics/0_MNIST/0_Basic_Modules.ipynb)

2) [NN으로 MNIST 풀어보기](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/03_CNN_Basics/0_MNIST/1_Linear_mnist.ipynb)

3) [CNN으로 MNIST 풀어보기](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/03_CNN_Basics/0_MNIST/3_CNN_clean.ipynb)

4) [CNN으로 CIFAR10 풀어보기](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/03_CNN_Basics/1_CIFAR/CNN_CIFAR10.ipynb)

## 4강 [Convolutional Neural Network - Advanced](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/04_CNN_Advanced/CNN_Advanced.pdf) 

1) [Custom Data 불러오기](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/04_CNN_Advanced/0_Custom_DataLoader.ipynb)

2) [VGGNet 구현해보기](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/04_CNN_Advanced/1_VGGNet.ipynb)

3) [GoogLeNet 구현해보기](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/04_CNN_Advanced/2_GoogleNet.ipynb)

4) [ResNet 구현해보기](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/04_CNN_Advanced/3_ResNet.ipynb)

## 5강 [Recurrent Neural Network - Basic](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/05_RNN/RNN.pdf)

1) [RNN 직접 만들어보기](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/05_RNN/0_Basic/Simple_Char_RNNcell.ipynb)

2) [LSTM 튜토리얼](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/05_RNN/1_LSTM/0_LSTM_Practice.ipynb)

3) [LSTM으로 문장 기억하기](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/05_RNN/1_LSTM/1_Char_LSTM.ipynb)

4) [nn.Embedding 사용법](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/05_RNN/2_Char_RNN/0_Embedding_Practice.ipynb)

5) [Shakespeare 문체 모방하기-RNN](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/05_RNN/2_Char_RNN/1_Char_RNN_Naive.ipynb)

6) [Shakespeare 문체 모방하기-GRU](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/05_RNN/2_Char_RNN/2_Char_RNN_GRU.ipynb)

7) [Shakespeare 문체 모방하기-LSTM](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/05_RNN/2_Char_RNN/3_Char_RNN_LSTM.ipynb)

## 6 [Problem & Solutions](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/06_Prob%26Solutions/Problem%26Solutions.pdf)

1) [Weight Regularization](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/06_Prob%26Solutions/0_Weight_Regularization.ipynb)

2) [Dropout](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/06_Prob%26Solutions/1_Dropout.ipynb)

3) [Data Augmentation](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/06_Prob%26Solutions/2_Data_Augmentation.ipynb)

4) [Weight Initialization](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/06_Prob%26Solutions/3_Weight_Initialization.ipynb)

5) [Learning Rate Scheduler](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/06_Prob%26Solutions/4_Learning_Rate_Decay.ipynb)

6) [Data Normalization](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/06_Prob%26Solutions/5_Data_Normalization.ipynb)

7) [Batch Normalization](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/06_Prob%26Solutions/6_Batch_Normalization.ipynb)

8) [Gradient Descent Variants](https://github.com/GunhoChoi/PyTorch_FastCampus/blob/master/06_Prob%26Solutions/7_Gradient_Descent_Variants.ipynb)
