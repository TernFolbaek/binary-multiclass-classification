## Binary and Multiclass Classification
In this project, we explore two different types of classification: **binary classification** and **multiclass classification**. These tasks are separated into two distinct files:

- **Binary Classification**: This approach is used when the model's output is one of two possible values. It’s often applied to tasks like yes/no or true/false predictions.
- **Multiclass Classification**: This method is used when the model needs to classify inputs into more than two categories. It is typically used for problems like recognizing handwritten digits (0-9) or categorizing images into various classes.

---

## `nn-binary-classification.ipynb`
This Jupyter notebook focuses on implementing a neural network for **binary classification**. It walks through the steps involved in classifying elements where the output can be one of two values (e.g., 0 or 1). A key part of the process is using the **sigmoid function** to handle the binary nature of the output in the linear regression algorithm.

### Keras Implementation

#### Load Test Data
The first step involves loading our test data. In this example, we use the **MNIST** dataset, a well-known open-source library containing images of handwritten digits. The data is pre-labeled, making it ideal for training and testing our binary classification model.

#### Define the Keras Model
We define a **Keras** model with three layers:
- **Input Layer**: 25 neurons, using a sigmoid activation function.
- **Hidden Layer**: 15 neurons, also using a sigmoid activation function.
- **Output Layer**: 1 neuron, using a sigmoid activation function to produce a binary output.
  
This model setup is designed to learn from the data and predict a binary outcome.

#### Examine Weights
Next, we examine the **shape of the weights** in our model. Since each image in the MNIST dataset is 28x28 pixels, each image is represented as a vector of **784 (28 * 28) values**. This allows us to see how the input data is being transformed through the layers of the network.

#### Compile & Train the Model
We then **compile** the model, specifying the loss function and optimizer, and **train** it using the training data. This step allows the model to learn the relationships between input features and the binary output.

#### Make Predictions
After training, we use the model to **make predictions** on new data, using the learned weights to classify each input into one of two categories.

---

## Own Implementation

#### Define Sigmoid Function
The **sigmoid function** is used to transform inputs into a value between 0 and 1, making it ideal for binary classification tasks. It is defined as:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

This function is applied to the weighted sums produced by each neuron, turning them into probabilities that indicate how likely an input is to belong to a particular class.

### Define Our Own Dense Layer
We create a custom implementation of a **dense layer**, which is a core building block of neural networks. A dense layer connects each input to every neuron in the layer through a weight matrix and a bias term. For each input:
- We calculate a weighted sum using the input values and their respective weights.
- We add a bias to this sum to shift the output.
- Finally, we apply an activation function, such as the sigmoid function, to introduce non-linearity into the model.

This custom implementation helps deepen our understanding of how dense layers work under the hood, particularly the process of transforming inputs through matrix multiplication and activation functions.

### Define Our Sequential Network
In our own implementation, we construct a simple **sequential neural network** using the custom dense layers. The network contains three layers, each of which processes the data sequentially:
1. **First Layer**: Takes the initial input and applies the dense layer transformation with a sigmoid activation function, producing the first set of activations.
2. **Second Layer**: Takes the output of the first layer as input and applies another dense layer transformation, again using the sigmoid function.
3. **Third Layer**: The final layer transforms the activations from the second layer into the output of the network, using a sigmoid function to produce a binary prediction.

This sequential flow mimics how information is processed in a Keras model but allows us to manually control each step. The network's final output is a value between 0 and 1, representing the probability that the input belongs to a particular class. This probability can be interpreted as a prediction, with a threshold to determine whether the output is classified as a 0 or 1.

### Make Predictions
After building the model, we use it to **make predictions** by feeding it new data. By manually implementing this process, we gain a deeper understanding of how neural networks work, from the calculation of weighted sums to the role of activation functions in transforming those sums into meaningful predictions.

## `nn-multiclass-classification.ipynb`

### Define softmax function
We begin by defining our softmax function which looks as follows:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

Our softmax function takes all the logits (the logit is the ouput of a neuron before being sent into an activation function) from a layer, where the numerator is $z_i$ which is the logit of a single neuron and is the output of the following format $\vec{w} \cdot \vec{x} + b$. Whilst the denominator is the accumulated exponentiated output of all other neurons from that layer.
### Reshape Data

### Define our network
We define our network which is of 3 layers, first layer containing 25 neurons then 15 neurons then 10. The activation function for the first two layers is relu, allowing the ouput of all positive values, but no values under 0, and at last a linear activation which allows us to compile with ``SparseCategoricalCrossentropy`` with with_logits=true allowing for more accurate outputs.

### Visualize predictions
We at last visualize our predictions for each handwritten digit, with the true label target y and our prediction.
