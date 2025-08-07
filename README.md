# neural-network
<br>
A basic implementation of neural network in python.

No warranty, that it will work on your system.
Not responsible ony any damage caused on your system.
This implementation is not intended for large-scale applications. In particular, no GPU support!

## Basic ideas behind it
Multi-layer Perceptron (MLP) is a supervised learning algorithm that learns a function
by training on a dataset, where M is the number of dimensions for input and O is the number of dimensions for output.
Given a set of features and a target, it can learn a non-linear function approximator for either classification or regression.
It is different from logistic regression, in that between the input and the output layer, there can be one or more non-linear layers, called hidden layers.

## Results
The trained data will be the [!MINST dataset](/assets/mnist-img/MnistExamples.png)

## Expected Output !!! CPU - HEAVY !!!

[!Figure_1](/assets/Figure_1.png)



## Usage
```bash

source .venv/bin/activate

python3 src/nn.py

```

<br>

[The Sauce](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
