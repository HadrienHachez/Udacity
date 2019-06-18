# Artificial Intelligence Lab

We are here building a minimal version of self driving car. Here, we have a front camera view. This will transfer input to the computer. Then Deep Learning algorithm in computer predicts the steering angle to avoid all sorts of collisions. Predicting steering angle can be thought of as a regression problem. We will feed images to Convolutional Neural Network and the label will be the steering angle in that image. Model will learn the steering angle from the as per the turns in the image and will finally predicts steering angle for unknown images.

## Libraries and Dependencies

- Keras, Tensorflow, scikit-learn (Machine Learning libraries)
- openCV, PIL (Image Processing libraries)
- pandas, numpy, matplotlib, base64, io (data stucture and data analysis libraries)
- socketio, eventlet (session with the simulator)

## Starting the simulation

1. Clone the repository: `git clone https://github.com/julienkessels/udacity-self-driving-car`
2. `cd udacity-self-driving-car`
3. Run the simulator available [here](https://github.com/udacity/self-driving-car-sim)
4. `python drive.py [1,2]` 0:  1:simple CNN, 2: custom vgg

## Model Architecture

We have implemented a convolutional neural network.

It is a relatively simple model of only X layers.
The architecture is a combination of Convolutional layers followed by Fully-Connected layers, since the input data is a raw RGB image.
This time, the architecture is applied to a **regression problem** (predicting
steering angle) instead of classification, so no activation function
or softmax must be applied at the last layer, which will have only one neuron.

The implemented network consists of the following layers:

- **Input**. Image of size (66, 200, 3).
- **Convolutional 1**. 24 filters of size 5x5. The filter is applied with strides of (3, 3)
- **Convolutional 2**. 36 filters of size 5x5. Strides of (2, 2).
- **Convolutional 3**. 48 filters of size 3x3. Strides of (2, 2).
- **Convolutional 4**. 64 filters of size 3x3.

- **Flatten**.
- **Dropout** to mitigate the effects of overfitting.

- **Fully Connected 1**, with 60 neurons.
- **Fully Connected 2**, with 20 neurons.
- **Fully Connected 4**, with 1 neuron, being the output.

All the layers, except for the output layer, have a **ELU activation function**.
The motivation to prefer it over ReLU is that it has a continuous derivative
and x = 0 and does not kill negative activations. The result is a bit smoother
steering output.

We have used quite an agressive **Dropout with probability of keeping = 0.25**,
in order to prevent overfitting and have a smooth output.

## Code overview

In this project, you will find three different files, each of them specific to a particular task. An additional file (the jupyter Notebook, will explain how they work)

- [`utils.py`](utils.py)

This file's main purpose is to provide preprocessing and data augmentation techniques for our machine learning model.

## Contributors

- Hadrien Hachez (15306)


## Acknowledge

Thanks to Julien and Saikou without who I was not able to present something for the exam.

## References

https://github.com/julienkessels/udacity-self-driving-car/blob/master/drive.py

http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

https://medium.com/@cacheop/using-deep-learning-to-clone-driving-behavior-30100c0782f6

https://towardsdatascience.com/deep-learning-for-self-driving-cars-7f198ef4cfa2

https://github.com/hminle/car-behavioral-cloning-with-pytorch/

https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7

https://github.com/llSourcell/How_to_simulate_a_self_driving_car
