# NpTorch
- PyTorch implementation  from scratch using Numpy.
- Inspired from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). 
- What's different from micrograd?
    - **VECTORIZATION**
    - all mathematical operations take advantage of numpy vectorized algebra 
    - making it orders of magnitude faster than micrograd
    - NpTorch speeds are comparable to the speeds of PyTorch (cpu as tested on i5 7200U :( )
- What's the use ?
    - TBH, none, it is a rudimentary attempt to clone pytorch functionality in order to understand its working.

## Functionality :
- Autograd (i.e. backpropagation) engine 
- Activation function 
    - ReLU, 
    - Tanh, 
    - Sigmoid, 
    - Softmax
- Regularization layer 
    - Dropout,
- Loss Function 
    - MSE, 
    - Cross-entropy
- (will add more as I implement them in future)
