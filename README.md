# CIFAR10Net
A neural network for classifying the CIFAR 10 dataset.

## Model Architechture
The model has two convolutional layers and two fully connected layers. Each convolutional layer uses the leaky ReLU activation function and is followed by a pool layer of size 2 and stride 2. The fully connected layers use the sigmoid activation function to coerce outputs to [0, 1]. They also have droupout with probability 0.5. The output layer has 10 neurons, each representing the probability the image belongs to that class.

## Data
The data is the CIFAR-10 dataset obtained [here](https://www.cs.toronto.edu/~kriz/cifar.html). The images are resized into shape (3, 32, 32) and the data is normalized to range [0, 1].
