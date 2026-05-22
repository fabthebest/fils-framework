# Module 04 — Deep Learning

**The FILS Framework**
Part of the open-source AI/ML curriculum for complete beginners.

---

## Module Overview

This module introduces deep learning: what neural networks actually are, how they learn, and how the most important architectures work. Every concept is delivered as a BRIDGIA card — seven cases per concept, from Hook to Application.

**Concepts in this module:**

| Card | Concept | Difficulty | Relevance |
|------|---------|------------|-----------|
| 24 | Neurons and Perceptrons | 3/5 | 5/5 |
| 25 | Neural Network Architecture | 4/5 | 5/5 |
| 26 | Backpropagation | 4/5 | 5/5 |
| 27 | Convolutional Neural Networks | 4/5 | 5/5 |
| 28 | RNNs and LSTMs | 4/5 | 4/5 |
| 29 | Transfer Learning | 3/5 | 5/5 |
| 30 | Generative Adversarial Networks | 5/5 | 4/5 |

---

---

## Card 24 — Neurons and Perceptrons

**Difficulty:** 3/5 | **Relevance:** 5/5 | **Skins:** Human Body / Restaurant

---

### Case 1 — Hook

Your brain contains roughly 86 billion neurons. None of them is intelligent on its own. Each one does something almost embarrassingly simple: it receives signals from neighbors, adds them up with some weighting, and either fires or stays quiet.

That is the entire mechanism. No reasoning. No memory. No understanding. Just: weighted sum, threshold, fire or not.

So how does a collection of such simple switches produce the ability to recognize faces, understand language, and drive cars? The question this card answers is: what exactly does a single artificial neuron do, and why is it the right building block for everything that follows?

---

### Case 2 — Mental Image

**Human Body skin:** A single nerve cell in your spinal cord receives incoming signals from dozens of other neurons through its dendrites. Each incoming signal arrives with a different strength — some strong, some weak, some inhibitory. The cell sums all of these weighted contributions. If the total crosses a threshold, the neuron fires an action potential down its axon to the next cell. If it does not cross the threshold, nothing happens. The neuron has made a binary decision based on a weighted vote from its neighbors.

An artificial neuron (perceptron) is exactly this: take inputs, multiply each by a learned weight, add them all up, add a bias term, pass through an activation function, produce one output.

**Restaurant skin:** A head taste-tester receives five small plates — sweetness, acidity, saltiness, bitterness, umami. Each dimension matters, but not equally: in this dish, umami matters twice as much as sweetness. The tester weighs each flavor proportionally, sums his impression, and makes a single binary decision: serve this dish to the dining room or send it back to the kitchen.

The weights encode what matters. The threshold encodes how demanding the tester is. The output is always one bit: yes or no.

---

### Case 3 — Decryption

A perceptron computes:

```
output = activation(w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b)
```

- `x_i` are the inputs (features)
- `w_i` are learnable weights (initialized randomly, updated by training)
- `b` is the bias term (shifts the activation threshold)
- `activation` is a non-linear function — sigmoid, ReLU, or tanh

Without the activation function, stacking neurons produces nothing more powerful than a single linear transformation. The activation function introduces non-linearity, which is what allows deep networks to learn complex patterns.

The perceptron is the 1958 version: a single neuron with a step activation function. It can only separate linearly separable classes. The modern neuron uses smooth activation functions (ReLU being the standard) and is trained with gradient descent rather than the original Rosenblatt rule.

---

### Case 4 — Minimal Code

```python
import numpy as np
import tensorflow as tf

# A single perceptron: 3 inputs -> 1 output
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(3,))
])

# One prediction pass on a single sample
x = np.array([[0.5, 1.2, -0.3]])
output = model.predict(x, verbose=0)
print(f"Weights: {model.layers[0].get_weights()[0].flatten().round(3)}")
print(f"Bias:    {model.layers[0].get_weights()[1].round(3)}")
print(f"Output:  {output[0][0]:.4f}")
```

**Expected output:**
```
Weights: [ 0.312 -0.748  0.521]
Bias:    [0.]
Output:  0.5813
```

Note: weights are random at initialization, so exact values vary. The output will always be between 0 and 1 because of the sigmoid activation.

---

### Case 5 — (See output above)

---

### Case 6 — Analysis and Intuition

- The weight on each input encodes how much influence that input has on the output. A large positive weight means the input strongly pushes the output toward 1. A large negative weight pushes it toward 0.
- The bias allows the neuron to fire even when all inputs are zero. Without it, the decision boundary always passes through the origin.
- A single neuron with sigmoid activation is equivalent to logistic regression. The only thing that makes deep learning powerful is stacking many such neurons in layers and training them jointly.
- ReLU (Rectified Linear Unit) — `max(0, x)` — replaced sigmoid as the standard activation for hidden neurons. It is faster to compute, avoids the vanishing gradient problem, and produces sparse activations (many zeros).

---

### Case 7 — Traps and Limits

**Trap 1 — Thinking one neuron is powerful**
A single perceptron cannot solve XOR. It cannot separate any non-linearly separable problem. The power comes from networks, not from individual neurons.

**Trap 2 — Confusing weights with importance scores**
After training, large weights do not simply mean "important features." Weights are entangled with the scale of inputs, with the weights of other neurons, and with the training dynamics. Do not interpret individual weights as feature importance.

**Trap 3 — Using sigmoid activations in hidden layers**
Sigmoid saturates near 0 and 1, producing near-zero gradients during backpropagation. For hidden layers, use ReLU. Reserve sigmoid for binary output neurons.

**Mirror Mode: Where the Analogies Break**

The nerve cell analogy breaks at the speed of update. A biological neuron strengthens or weakens its synapses over minutes, hours, and days through long-term potentiation. An artificial neuron's weights are updated thousands of times per second during training. The timescale of learning is incomparable.

The taste-tester analogy breaks at the discreteness of decision. The tester deliberates; they have memory, context, and aesthetic judgment. The perceptron has none of these — it is a fixed arithmetic operation applied to a vector. There is no deliberation, only arithmetic.

---

### Case 8 — Application Exercise

**Exercise: Building intuition for weights**

Create a single neuron with two inputs and sigmoid activation. Manually set the weights to `[2.0, -2.0]` and the bias to `0.0`.

1. Evaluate it on four inputs: `[0,0]`, `[1,0]`, `[0,1]`, `[1,1]`.
2. Print each output and describe in one sentence what the neuron is detecting.
3. Change the bias to `1.0` and repeat. What changes?
4. What logical function does this neuron approximate?

**Success condition:** You can read a weight vector and bias and predict, without running any code, whether a given input will produce a high or low output.

---

---

## Card 25 — Neural Network Architecture

**Difficulty:** 4/5 | **Relevance:** 5/5 | **Skins:** Restaurant / Orchestra

---

### Case 1 — Hook

A single neuron can draw one straight line through data. Two neurons can combine two lines. But classifying handwritten digits requires distinguishing 10 shapes across thousands of possible stroke variations. No small number of lines can do that.

So you stack neurons into layers, and layers into networks. The question this card answers is: what does the structure of a neural network look like, why does it have the shape it does, and what does each layer actually contribute to the final answer?

---

### Case 2 — Mental Image

**Restaurant skin:** A Michelin-starred kitchen is organized as a brigade. At the front, a prep station receives raw ingredients — washed vegetables, measured spices, raw protein. This is the input layer: it does not cook anything, it only passes structured data into the pipeline.

In the middle, three cooking stations work in parallel and in sequence: one handles heat and browning, one handles sauces and reductions, one handles seasoning and finishing. Each station takes the output of the previous one, transforms it, and passes it on. These are the hidden layers. No single station sees the final dish; each one builds an intermediate representation.

At the back, the plating station receives the fully processed components and arranges them into one final presentation. This is the output layer: it takes the high-level features built by the hidden layers and produces the final answer — the dish, the class label, the prediction.

**Orchestra skin:** An orchestra has four sections: strings, woodwinds, brass, percussion. Each section processes the score in its own way — different timbres, different frequency ranges, different roles. The full sound at any moment is a combination of what all sections produce simultaneously. The conductor (the loss function during training) listens to the result and shapes how each section plays. No section produces the music alone; the music is the sum of layered contributions.

---

### Case 3 — Decryption

A feedforward neural network has three types of layers:

**Input layer:** Passes raw features into the network. No computation occurs here. Width equals the number of input features.

**Hidden layers:** Each applies an affine transformation followed by a non-linear activation:
```
h = activation(W * h_prev + b)
```
Multiple hidden layers allow the network to build hierarchical representations. Early layers learn simple features; deeper layers combine them into complex ones.

**Output layer:** Produces the final prediction. Activation depends on the task:
- Binary classification: sigmoid (output between 0 and 1)
- Multi-class classification: softmax (outputs sum to 1, one per class)
- Regression: linear (no activation)

**Universal Approximation Theorem:** A feedforward network with one hidden layer of sufficient width can approximate any continuous function. In practice, depth is more efficient than width — deeper networks need exponentially fewer neurons to represent certain functions.

---

### Case 4 — Minimal Code

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Network for classifying 10 handwritten digit classes (MNIST-style)
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # hidden layer 1
    Dense(64,  activation='relu'),                       # hidden layer 2
    Dense(10,  activation='softmax')                     # output: 10 classes
])

model.summary()
```

**Expected output:**
```
Model: "sequential"
_________________________________________________________________
 Layer (type)            Output Shape         Param #
=================================================================
 dense (Dense)           (None, 128)          100480
 dense_1 (Dense)         (None, 64)           8256
 dense_2 (Dense)         (None, 10)           650
=================================================================
Total params: 109,386
Trainable params: 109,386
Non-trainable params: 0
```

Note: 784 inputs * 128 neurons + 128 biases = 100,480 parameters in the first layer alone.

---

### Case 6 — Analysis and Intuition

- The first hidden layer learns to detect simple patterns in the raw input. For images, these might be edges and gradients.
- Each subsequent hidden layer combines the patterns of the previous one into increasingly abstract representations.
- The output layer does not learn features; it learns to combine the final hidden layer's representation into a probability distribution over classes.
- Deeper is not always better. Very deep networks suffer from vanishing gradients (covered in Card 26) unless architectural tricks like batch normalization or skip connections are used.
- Width (neurons per layer) controls capacity within a layer. Depth (number of layers) controls the complexity of the composable hierarchy.

---

### Case 7 — Traps and Limits

**Trap 1 — Treating depth as a hyperparameter to maximize**
Adding more layers always increases model capacity, but also increases training difficulty, memory cost, and overfitting risk. Start shallow. Deepen only when underfitting is confirmed.

**Trap 2 — Ignoring the output activation**
Using softmax for regression or linear for classification are common mistakes. The output activation determines what the loss function expects. These must match.

**Trap 3 — Fully connected layers on image data**
Flattening a 224x224 image and feeding it to a Dense layer produces 150,528 inputs. The first layer would need millions of parameters just to cover reasonable widths. This is why convolutions exist (Card 27).

**Mirror Mode: Where the Analogies Break**

The kitchen brigade analogy breaks at the direction of feedback. In a real kitchen, a cook at the sauce station can ask the prep station to change how they cut the vegetables. In a standard feedforward network, information flows in only one direction — forward during prediction, backward only during training. The cooks cannot communicate horizontally; each station only ever receives from the station behind it.

The orchestra analogy breaks at coordination. Musicians listen to each other and adjust in real time. Neurons in a layer do not hear each other — each neuron in a hidden layer is computed independently, in parallel, with no lateral communication (unless you use attention mechanisms or other specialized architectures).

---

### Case 8 — Application Exercise

**Exercise: Counting parameters**

Build three networks for a 20-feature input, 3-class output problem:
- Network A: one hidden layer of 50 neurons
- Network B: two hidden layers of 25 neurons each
- Network C: three hidden layers of 16 neurons each

1. Call `model.summary()` on each.
2. Record the total parameter count.
3. Which has the most parameters? Which has the deepest representation hierarchy?
4. Train all three on any small dataset (e.g., sklearn's `make_classification`). Which converges fastest? Which overfits first?

**Success condition:** You can predict a network's parameter count from its layer widths before running `summary()`, and you can explain why parameter count alone does not determine model quality.

---

---

## Card 26 — Backpropagation

**Difficulty:** 4/5 | **Relevance:** 5/5 | **Skins:** Restaurant / Construction

---

### Case 1 — Hook

You have a neural network with 109,000 parameters. You run one prediction. It is wrong. Now what?

Somehow, each of the 109,000 weights must be nudged in the direction that would have made the prediction less wrong. But how do you know which weights caused the error, and by how much each should move? The network is a chain of multiplications and additions. The error came out at the end of that chain. To fix it, you must trace responsibility backward through every operation that contributed to it.

This is backpropagation. The question this card answers is: how does the error at the output become a precise gradient for every single weight in the network?

---

### Case 2 — Mental Image

**Restaurant skin:** A head chef tastes the final dish and notes that it is too salty. He does not just shout "fix it" at the kitchen generally. He walks backward through the brigade: the plating station added a finishing salt — reduce by half. The sauce station used a heavily reduced stock — that concentrated the salt — reduce the reduction time. The prep station over-salted the marinade — halve the marinade salt. Each station receives a precise, quantified correction based on exactly how much their contribution affected the final taste.

This backward walk, with proportional corrections at each station, is backpropagation.

**Construction skin:** A quality inspector examines a completed wall and finds a crack. He does not call the whole project a failure and start over. He traces the crack backward through the construction log: the crack started at a joint that was filled with insufficient mortar — correction noted. The insufficient mortar was caused by a mix that was too dry — correction noted. The dry mix was caused by incorrect water measurement at the batch stage — correction noted. Each stage gets a specific adjustment, weighted by how much that stage contributed to the final defect.

---

### Case 3 — Decryption

Backpropagation applies the chain rule of calculus to compute the gradient of the loss function with respect to every weight in the network.

For a network with loss `L` and a weight `w` in layer `k`:

```
dL/dw = dL/d_output * d_output/d_hidden_k * d_hidden_k/dw
```

This chain extends backward through every layer. The key insight is that these partial derivatives can be computed efficiently by reusing intermediate values calculated during the forward pass.

**The algorithm in three steps:**
1. Forward pass: compute predictions and cache all intermediate activations.
2. Compute loss: measure how wrong the prediction was.
3. Backward pass: propagate the error gradient from the output layer back to the input layer, computing `dL/dw` for every weight using the cached activations.

The gradients are then used by an optimizer (SGD, Adam, RMSprop) to update weights:
```
w = w - learning_rate * dL/dw
```

You never need to implement backprop manually — frameworks compute it automatically via automatic differentiation. But understanding it is essential for debugging training failures.

---

### Case 4 — Minimal Code

```python
import tensorflow as tf
import numpy as np

# Simple binary classification: learn XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([[0],[1],[1],[0]],         dtype=np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=500, verbose=0)

print(f"Final loss:     {history.history['loss'][-1]:.4f}")
print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Predictions:    {model.predict(X, verbose=0).flatten().round(2)}")
```

**Expected output:**
```
Final loss:     0.0231
Final accuracy: 1.0000
Predictions:    [0.02 0.98 0.98 0.03]
```

XOR is the classic problem a single perceptron cannot solve (it is not linearly separable). With one hidden layer and 500 backpropagation steps, the network solves it.

---

### Case 6 — Analysis and Intuition

- Backprop only works because every operation in the network is differentiable. This is why activation functions must have computable derivatives — and why the step function used in the original perceptron cannot be used in modern networks.
- The learning rate controls how large each update step is. Too large: the optimizer overshoots and the loss oscillates. Too small: training takes thousands of additional epochs.
- Vanishing gradients: in deep networks, gradients can shrink exponentially as they propagate backward. By the time they reach the first layers, the update is nearly zero — early layers stop learning. ReLU activations and careful initialization help prevent this.
- Exploding gradients: the reverse problem. Gradients grow exponentially, causing weight updates so large they destabilize training. Gradient clipping and batch normalization are the standard fixes.

---

### Case 7 — Traps and Limits

**Trap 1 — Assuming backprop finds the global minimum**
Backprop finds a local minimum of the loss surface, not necessarily the global one. In practice, local minima in high-dimensional loss surfaces are often good enough — but there is no guarantee of optimality.

**Trap 2 — Forgetting to compile before training**
In Keras, `model.fit()` without `model.compile()` raises an error. Compile specifies the optimizer and loss function — both are required for backprop to run.

**Trap 3 — Monitoring only training loss**
Backpropagation minimizes training loss by definition. A training loss of 0.001 tells you backprop is working. It tells you nothing about generalization. Always monitor validation loss in parallel.

**Mirror Mode: Where the Analogies Break**

The head chef analogy breaks at precision. A real chef makes qualitative adjustments based on experience and judgment — "a bit less salt." Backpropagation computes exact numerical derivatives. The correction for each weight is not an opinion; it is a mathematically derived quantity telling the weight precisely how much to change to reduce the loss by the maximum amount per unit of change.

The construction inspector analogy breaks at simultaneity. An inspector traces a crack sequentially, station by station. Backpropagation computes all gradients in a single backward sweep through the computational graph, reusing cached values from the forward pass for efficiency. The corrections are derived in parallel across the entire network, not investigated one stage at a time.

---

### Case 8 — Application Exercise

**Exercise: Watching the loss fall**

Train the XOR network from the code example above, but this time run with `verbose=1` for 200 epochs.

1. Plot `history.history['loss']` over 500 epochs. What shape does the curve take?
2. Try learning rates of 0.001, 0.01, and 0.1 by passing `optimizer=tf.keras.optimizers.Adam(learning_rate=X)`. How does the curve change?
3. Reduce the hidden layer to 2 neurons. Can the network still solve XOR? At what epoch does it converge?

**Success condition:** You can look at a loss curve and diagnose whether the learning rate is too high (oscillating loss), too low (flat loss), or appropriate (smooth descent to a low value).

---

---

## Card 27 — Convolutional Neural Networks

**Difficulty:** 4/5 | **Relevance:** 5/5 | **Skins:** Video Game / City

---

### Case 1 — Hook

You want a neural network to recognize cats in photographs. A 224x224 pixel image has 150,528 numbers. A fully connected first layer with 1,000 neurons would need 150 million parameters just for that one layer. And none of those parameters would know that a pixel at position (50,50) is related to the pixel at (51,50) — a neighbor one step away.

Fully connected layers treat every pixel as independent. But images are not made of independent pixels. Structure — edges, textures, shapes — is local. The solution is a different kind of layer that explicitly exploits that locality.

---

### Case 2 — Mental Image

**Video Game skin:** A scout in a real-time strategy game cannot see the entire map at once. She has a small viewing window — say, a 3x3 tile area — that she moves systematically across the terrain. Within each window, she checks: is there a river here? A forest? An enemy building? She records her findings and moves the window one step right. After scanning every position, she hands her notes to a strategist who combines local observations into a global map.

A convolutional filter is that scanning window. It slides across the image, detecting one specific pattern at every position. Stacking many filters means scanning for many patterns simultaneously — edges, corners, textures, gradients — building a rich feature map from local evidence.

**City skin:** A city surveillance system covers a large area with a grid of cameras, each covering one city block. Each camera is tuned to detect a specific event type — a parked car, a pedestrian crossing, a lit window. No camera sees the whole city, but together they produce a complete picture by aggregating local observations. A central system then combines block-level summaries into district-level patterns.

---

### Case 3 — Decryption

A convolutional layer applies a set of learned filters to an input image (or feature map). Each filter is a small matrix (typically 3x3 or 5x5) of learned weights that slides across the input with a stride. At each position, the filter computes the dot product between its weights and the patch of the input it covers. The result is a single number placed in the output feature map.

Key properties:
- **Weight sharing:** the same filter is used at every position. A filter that detects a vertical edge will detect it anywhere in the image, not just in the position it was trained on.
- **Translation invariance:** because the filter scans the entire input, the network becomes robust to where in the image a pattern appears.
- **Local connectivity:** each output unit depends only on a small patch of the input, not the entire image.

A typical CNN architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Dense -> Output.

Pooling (max pooling) downsamples feature maps by taking the maximum value in each region, reducing spatial dimensions while preserving dominant features.

---

### Case 4 — Minimal Code

```python
import tensorflow as tf

# Small CNN for 28x28 grayscale images (MNIST-style), 10 classes
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
```

**Expected output:**
```
Model: "sequential"
_________________________________________________________________
 Layer (type)            Output Shape         Param #
=================================================================
 conv2d (Conv2D)         (None, 26, 26, 32)   320
 max_pooling2d           (None, 13, 13, 32)   0
 conv2d_1 (Conv2D)       (None, 11, 11, 64)   18496
 max_pooling2d_1         (None, 5, 5, 64)     0
 flatten (Flatten)       (None, 1600)         0
 dense (Dense)           (None, 64)           102464
 dense_2 (Dense)         (None, 10)           650
=================================================================
Total params: 121,930
```

Notice: the two convolutional layers together use only 18,816 parameters to process 28x28 images. A fully connected equivalent would need millions.

---

### Case 6 — Analysis and Intuition

- The first convolutional layer typically learns edge detectors — patterns sensitive to horizontal, vertical, and diagonal edges.
- Deeper convolutional layers combine edges into textures, and textures into shapes.
- Max pooling discards spatial precision in exchange for robustness. After pooling, a feature "I detected a vertical edge" no longer depends on the exact pixel location of that edge.
- CNNs generalize from 2D images to any structured grid data: audio spectrograms (1D convolution over time), 3D medical volumes (3D convolution), video frames (3D convolution over space and time).

---

### Case 7 — Traps and Limits

**Trap 1 — Applying CNNs to non-spatial data**
Tabular data with 20 features does not have spatial structure. Reshaping it into a 2D array to apply Conv2D introduces false spatial relationships between features. For tabular data, use Dense layers.

**Trap 2 — Ignoring padding**
Without padding (`padding='valid'`), each convolutional layer reduces the spatial dimensions. After several layers, small images can shrink to nothing. Use `padding='same'` to maintain spatial dimensions when needed.

**Trap 3 — Too many filters too early**
Starting with 512 filters in the first layer on small images wastes parameters. Filters in early layers detect simple low-level patterns that do not require high numbers. A standard progression is 32 -> 64 -> 128.

**Mirror Mode: Where the Analogies Break**

The scout/viewing-window analogy breaks at the nature of the patterns detected. The scout knows in advance what she is looking for — rivers, forests, enemies. A convolutional filter does not know what pattern it will detect; it learns the optimal pattern during training. The filter's weights are initialized randomly and shaped by backpropagation. The scout has a fixed concept of "enemy building"; the filter develops its concept of "useful pattern" from data.

The city surveillance analogy breaks at independence. Each city camera operates independently; they do not combine their outputs during observation, only afterward. In a CNN, each filter is applied simultaneously to the entire feature map, and the outputs of all filters from one layer collectively form the input to the next layer. The cameras of the analogy are independent sensors; CNN filters are jointly trained components of a single computation.

---

### Case 8 — Application Exercise

**Exercise: Visualizing what filters detect**

Load the MNIST dataset using `tf.keras.datasets.mnist.load_data()`. Train the CNN from the code example for 3 epochs.

1. Extract the weights of the first convolutional layer: `model.layers[0].get_weights()[0]`.
2. Visualize the 32 filters as 3x3 grayscale patches using matplotlib.
3. Pick one filter that looks like an edge detector. Find a test image where this filter produces a high activation. Show both the original image and the feature map.

**Success condition:** You can describe in plain English what pattern at least two of your trained filters appear to detect, and explain why the filter shape is 3x3 rather than, say, 1x1.

---

---

## Card 28 — RNNs and LSTMs

**Difficulty:** 4/5 | **Relevance:** 4/5 | **Skins:** Orchestra / Restaurant

---

### Case 1 — Hook

You are reading the sentence: "The musician picked up the violin she had been playing since..."

At the word "since," you already expect the sentence to end with a time reference. You know this because you have been tracking the structure of the sentence since the first word. Your understanding at any point depends on everything that came before.

A feedforward network processes each input in isolation. It has no notion of sequence. Feed it word 7 of a sentence and it has no memory of word 1. But language, music, speech, and time series are fundamentally sequential — the past matters for understanding the present.

---

### Case 2 — Mental Image

**Orchestra skin:** A musician playing a symphony cannot read only the bar in front of them. The phrasing of the current bar depends on what was played in the last 10 bars — the dynamic arc, the thematic development, the harmonic tension built up over time. A musician who started mid-piece with no memory of what preceded would play technically correct notes but musically meaningless ones. An RNN is a musician who, at each timestep, holds a summary of everything played so far — a hidden state — and uses it to interpret the current input.

**Restaurant skin:** A skilled waiter does not treat each course as an isolated event. When you arrive at dessert, he remembers that you ordered a dry red wine with your main course, that you mentioned you prefer dark chocolate, and that you declined the cheese plate. He uses this cumulative memory — his hidden state — to recommend the right dessert pairing. A waiter with no memory of previous courses would be starting from scratch at every interaction.

---

### Case 3 — Decryption

A Recurrent Neural Network (RNN) processes sequential input one step at a time, maintaining a hidden state `h_t` that summarizes the history up to step `t`:

```
h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
```

The hidden state is passed forward at each step, giving the network a form of memory.

**The vanishing gradient problem in RNNs:** when backpropagating through long sequences, gradients are multiplied by the same weight matrix at each step. If this matrix has eigenvalues less than 1, gradients vanish exponentially as they propagate backward through time — the network cannot learn dependencies separated by more than ~10 steps.

**LSTM (Long Short-Term Memory)** solves this with gating mechanisms:
- **Forget gate:** decides what fraction of the previous memory to discard
- **Input gate:** decides what new information to write into memory
- **Output gate:** decides what part of the memory to expose as the hidden state

The cell state (a separate memory vector) can persist information over hundreds of timesteps without vanishing, because the gradient path through the cell state involves addition rather than repeated multiplication.

---

### Case 4 — Minimal Code

```python
import tensorflow as tf
import numpy as np

# Predict next value in a sine wave: sequence of 10 steps -> next step
T = 500
t = np.linspace(0, 50, T)
sine = np.sin(t).astype(np.float32)

X = np.array([sine[i:i+10] for i in range(T-11)]).reshape(-1, 10, 1)
y = np.array([sine[i+10]   for i in range(T-11)])

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=32, verbose=0)
print(f"Final MSE: {model.evaluate(X, y, verbose=0):.6f}")
```

**Expected output:**
```
Final MSE: 0.000183
```

The LSTM learns to predict the next sine value from the last 10 values with very low error because a sine wave is a perfectly regular sequence.

---

### Case 6 — Analysis and Intuition

- Vanilla RNNs work for short sequences (fewer than 20 steps). For anything longer — paragraphs of text, minutes of audio, long time series — use LSTM or GRU.
- GRU (Gated Recurrent Unit) is a simplified LSTM with two gates instead of three. It is faster to train and performs comparably on many tasks.
- For text, the sequence length is the number of tokens, not characters. A sentence of 20 words is a sequence of length 20.
- LSTMs process sequences one step at a time, which cannot be parallelized. Transformers (the architecture behind GPT and BERT) replaced LSTMs for most NLP tasks precisely because they can process the entire sequence in parallel using attention.

---

### Case 7 — Traps and Limits

**Trap 1 — Using RNN for non-sequential data**
Tabular data with no temporal or sequential structure does not benefit from an RNN. You are adding parameters and training complexity with no signal to exploit.

**Trap 2 — Not setting `return_sequences=True` when stacking LSTM layers**
By default, an LSTM layer returns only the final hidden state — a single vector. If you stack two LSTM layers, the first must return the full sequence: `LSTM(32, return_sequences=True)`. Otherwise the second layer receives only one timestep and cannot function as a sequence processor.

**Trap 3 — Ignoring sequence length at inference**
An LSTM trained on sequences of length 10 expects sequences of length 10 at inference. Variable-length sequences require padding, masking, or a different architecture.

**Mirror Mode: Where the Analogies Break**

The orchestra musician analogy breaks at the fidelity of memory. A musician remembers the full emotional and harmonic arc of the piece, with nuance and interpretive context. An LSTM's hidden state is a fixed-dimensional vector — a compressed summary of the past, not a faithful record of it. Long sequences can exhaust the capacity of that vector, causing old information to be overwritten. The musician's memory degrades gracefully; the LSTM's has a hard capacity limit.

The waiter analogy breaks at retrieval precision. When the waiter recalls that you preferred dark chocolate, he retrieves a specific, distinct memory. An LSTM does not store discrete memories that can be retrieved by content. It maintains a single blended state vector that is updated at every step. The effect of your first sentence is diffused and mixed into every subsequent update. There is no separate slot labeled "customer preference."

---

### Case 8 — Application Exercise

**Exercise: Long versus short memory**

Create two sequences:
- Short dependency: `y[t] = x[t-1] + noise` (target depends on 1 step ago)
- Long dependency: `y[t] = x[t-10] + noise` (target depends on 10 steps ago)

1. Train a vanilla `SimpleRNN` layer on both. Compare final MSE.
2. Train an `LSTM` layer on both. Compare final MSE.
3. Which model struggles more on the long dependency? Why?

**Success condition:** You observe that SimpleRNN fails on the 10-step dependency and can explain this in terms of the vanishing gradient, while LSTM maintains reasonable performance.

---

---

## Card 29 — Transfer Learning

**Difficulty:** 3/5 | **Relevance:** 5/5 | **Skins:** Construction / Restaurant

---

### Case 1 — Hook

Training a CNN to recognize images from scratch requires millions of labeled examples and days of GPU computation. Most practitioners have neither. But somewhere, someone already trained a network on 1.2 million images across 1,000 categories — and made it freely available.

Why start from nothing? The question this card answers is: what exactly is being transferred when you load a pretrained model, and when is it legitimate to reuse someone else's learned weights for your own problem?

---

### Case 2 — Mental Image

**Construction skin:** You acquire a partially built building: solid foundation, structural walls, plumbing, electrical — everything below the fifth floor. The top floor still needs interior finishing tailored to your specific business. You do not tear the building down and start from concrete. You hire a crew to fit out only the top floor while leaving the existing structure untouched. The pretrained lower layers are the existing structure. Your task-specific output head is the top-floor renovation.

**Restaurant skin:** A Michelin-trained chef has spent 15 years mastering knife technique, sauce-making, heat control, and flavor balance. You hire her to cook your specific regional cuisine. You do not retrain her on the basics — you teach her only the new dishes specific to your menu. She applies her existing expertise to your new context. The general culinary knowledge transfers; only the application-specific details need to be learned from scratch.

---

### Case 3 — Decryption

Transfer learning reuses weights learned on a large source task as the starting point for a different target task. It is effective because early layers of deep networks learn general-purpose features (edges, textures, shapes for CNNs; grammar and semantics for language models) that are useful across many tasks.

**Standard procedure for image classification:**
1. Load a pretrained model (e.g., VGG16, ResNet50, MobileNet) with weights from ImageNet training.
2. Freeze the base layers (`layer.trainable = False`).
3. Replace the final classification head with a new dense layer matching your number of classes.
4. Train only the new head on your small dataset.

**Fine-tuning:** after training the head, unfreeze the last few base layers and continue training with a very small learning rate. This adapts the pretrained features slightly to your domain without catastrophically forgetting what they learned.

**Domain proximity matters:** transfer learning works well when the source and target domains are similar (both are natural images, or both are English text). It degrades when they are distant (e.g., ImageNet weights transferred to X-ray classification require more fine-tuning).

---

### Case 4 — Minimal Code

```python
import tensorflow as tf

# Load MobileNetV2 pretrained on ImageNet, exclude the top classification layer
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False  # freeze all pretrained layers

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 new classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print(f"\nTrainable params:     {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
print(f"Non-trainable params: {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}")
```

**Expected output:**
```
...
Total params: 2,261,829
Trainable params: 6,405
Non-trainable params: 2,255,424

Trainable params:     6,405
Non-trainable params: 2,255,424
```

Only 6,405 parameters are trained — the new classification head. The 2.25 million pretrained parameters are frozen.

---

### Case 6 — Analysis and Intuition

- The closer your task is to the source task, the more layers you can freeze and the less data you need. Five-class flower classification with 500 images works well because ImageNet already contains flowers.
- If your domain is very different from ImageNet (e.g., satellite imagery, medical scans), freeze fewer layers and fine-tune more of the network. The low-level features still transfer; the mid-level ones may not.
- Transfer learning is why deep learning became practical for small teams with limited data. Before it, training a competitive image classifier required a large dataset and substantial compute. With transfer learning, competitive accuracy on a new task with 500 examples per class is achievable in minutes.
- For language, transformer models pretrained on large text corpora (BERT, GPT, RoBERTa) are fine-tuned on specific tasks. The principle is identical: general representations transferred to a specific application.

---

### Case 7 — Traps and Limits

**Trap 1 — Fine-tuning with a high learning rate**
When unfreezing pretrained layers to fine-tune, use a learning rate 10-100x smaller than you used for the head. A normal learning rate destroys the pretrained weights in the first epoch.

**Trap 2 — Assuming transfer always helps**
If your task domain is very different from the source (e.g., using ImageNet weights for sonar spectrograms), transfer may provide no benefit — or even harm performance by biasing the initialization toward irrelevant features. When in doubt, benchmark against random initialization.

**Trap 3 — Using the wrong input preprocessing**
Pretrained models expect inputs normalized in a specific way. MobileNetV2 expects pixel values scaled to [-1, 1]. VGG16 expects BGR channels with ImageNet mean subtracted. Use the preprocessing function bundled with the model (`tf.keras.applications.mobilenet_v2.preprocess_input`) rather than your own normalization.

**Mirror Mode: Where the Analogies Break**

The construction analogy breaks at cost. Reusing an existing building's structure saves construction cost but introduces constraints — the floor plan, ceiling heights, load-bearing walls are fixed by the original design. Pretrained layers also introduce constraints: the input dimensions, the feature representations, and the expected normalization are all fixed by the original training. You cannot arbitrarily change the architecture of the base model without retraining it.

The hired-chef analogy breaks at the granularity of transfer. When the chef brings her expertise, she brings everything simultaneously — technique, instinct, muscle memory — in an integrated way. In transfer learning, you transfer discrete numerical values (weights) layer by layer. You can selectively freeze or unfreeze individual layers, transfer only part of a model, or insert new layers in the middle. The chef analogy implies an all-or-nothing transfer of a unified skill set; the technical reality is a granular, layer-by-layer parameter reuse.

---

### Case 8 — Application Exercise

**Exercise: Frozen versus fine-tuned**

Using MobileNetV2 as a base, build two models for a binary classification task (cats vs. dogs, or any two-class dataset):

- Model A: freeze all base layers, train only the head for 10 epochs.
- Model B: after training the head for 5 epochs, unfreeze the last 20 layers of the base and train for 5 more epochs with a learning rate of `1e-5`.

1. Compare final validation accuracy for both.
2. Plot training and validation loss for both. Which shows more overfitting?
3. How many parameters are trainable in each configuration?

**Success condition:** You observe measurable accuracy improvement from fine-tuning, and can explain why a very small learning rate is required when unfreezing pretrained layers.

---

---

## Card 30 — Generative Adversarial Networks

**Difficulty:** 5/5 | **Relevance:** 4/5 | **Skins:** Restaurant / Construction

---

### Case 1 — Hook

Every model covered so far learns a mapping: input goes in, label or value comes out. But what if you want to generate new data — a realistic photograph of a face that does not exist, a synthetic medical scan, a new sample that matches the distribution of your training set?

Discriminative models classify. Generative models create. And the most powerful way to train a generative model is to give it an adversary — a second model whose sole purpose is to catch it cheating.

---

### Case 2 — Mental Image

**Restaurant skin:** A culinary school holds an ongoing competition. A student chef (the generator) attempts to produce dishes that could pass as being from a three-star restaurant. A professional food critic (the discriminator) must determine which dishes are genuinely from the three-star kitchen and which are fakes from the student.

At first, the student's dishes are obviously inferior. The critic catches them immediately and explains why — too rough, wrong texture, weak plating. The student learns from this feedback and improves. As the student improves, the critic must develop finer and finer detection skills to keep catching fakes. Both get better simultaneously, driven by each other's improvement. Eventually, the student produces dishes that even the expert critic cannot reliably distinguish from the real thing.

**Construction skin:** A team of forensic document examiners (discriminator) must detect counterfeit building permits. A forger (generator) keeps producing permits, and the examiners keep catching them and explaining the tells. The forger incorporates each piece of feedback — better paper stock, more accurate seal, correct date format — until the forgeries become indistinguishable from authentic documents.

---

### Case 3 — Decryption

A GAN consists of two neural networks trained in opposition:

**Generator G:** Takes random noise `z` as input and produces synthetic samples `G(z)`. Its goal is to produce samples that the discriminator classifies as real.

**Discriminator D:** Takes a sample (real or generated) and outputs a probability that it is real. Its goal is to correctly classify real samples as real and generated samples as fake.

**Training objective:**
```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

The discriminator maximizes its ability to distinguish real from fake. The generator minimizes the discriminator's ability to do so. This is a minimax game.

**Training procedure:**
1. Sample a batch of real data.
2. Sample noise, generate a batch of fake data via `G(z)`.
3. Train D on both: predict real as 1, fake as 0.
4. Train G to fool D: generate fakes and push D's output toward 1.
5. Repeat.

**Key failure mode — mode collapse:** the generator learns to produce only a small variety of outputs that consistently fool the discriminator, ignoring the full diversity of the real data distribution.

---

### Case 4 — Minimal Code

```python
import tensorflow as tf
import numpy as np

# Toy GAN: learn to generate 1D Gaussian samples (target: mean=2, std=0.5)
def make_generator():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1)
    ])

def make_discriminator():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

G = make_generator()
D = make_discriminator()
d_opt = tf.keras.optimizers.Adam(0.01)
g_opt = tf.keras.optimizers.Adam(0.01)
bce = tf.keras.losses.BinaryCrossentropy()

for step in range(1000):
    real = np.random.normal(2.0, 0.5, (64, 1)).astype(np.float32)
    noise = np.random.randn(64, 4).astype(np.float32)
    with tf.GradientTape() as tape:
        fake = G(noise, training=True)
        d_loss = bce(tf.ones((64,1)), D(real)) + bce(tf.zeros((64,1)), D(fake))
    D.optimizer = d_opt
    d_opt.apply_gradients(zip(tape.gradient(d_loss, D.trainable_variables), D.trainable_variables))
    with tf.GradientTape() as tape:
        g_loss = bce(tf.ones((64,1)), D(G(noise, training=True)))
    g_opt.apply_gradients(zip(tape.gradient(g_loss, G.trainable_variables), G.trainable_variables))

samples = G(np.random.randn(1000, 4).astype(np.float32)).numpy()
print(f"Generated mean: {samples.mean():.3f}  (target: 2.000)")
print(f"Generated std:  {samples.std():.3f}   (target: 0.500)")
```

**Expected output:**
```
Generated mean: 1.987  (target: 2.000)
Generated std:  0.493   (target: 0.500)
```

The generator has learned to produce samples matching the target distribution without ever seeing the target distribution directly — only by learning from the discriminator's feedback.

---

### Case 6 — Analysis and Intuition

- GANs are notoriously difficult to train. The generator and discriminator must improve at roughly equal rates. If the discriminator becomes too good too fast, the generator receives near-zero gradients and stops learning. If the generator becomes too good too fast, the discriminator loses signal.
- Mode collapse is the most common failure: the generator produces one type of output that consistently fools the discriminator, ignoring all other modes of the real distribution.
- Wasserstein GAN (WGAN) replaces the standard loss with a different distance measure that provides more stable gradients and better diagnoses of training progress.
- Modern image generation (Stable Diffusion, DALL-E) uses diffusion models rather than GANs. GANs dominated image synthesis from 2014 to ~2021 but have been largely replaced for generation tasks. They remain relevant for discriminative augmentation, domain adaptation, and anomaly detection.

---

### Case 7 — Traps and Limits

**Trap 1 — Evaluating GANs with training loss**
A GAN loss that is decreasing for both G and D does not mean the samples are improving. The losses measure the equilibrium state of the game, not the quality of generated outputs. Visual inspection and quantitative metrics (Frechet Inception Distance, FID) are required.

**Trap 2 — Ignoring the discriminator's strength relative to the generator**
If D is much stronger than G at the start (e.g., a deep discriminator vs. a shallow generator), D will quickly learn to detect all fakes with near-certainty, providing near-zero gradient to G. Balance network capacity between G and D.

**Trap 3 — Expecting stability from the default Adam parameters**
Standard Adam (`beta_1=0.9`) often destabilizes GAN training. A common empirical fix is `beta_1=0.5` for both optimizers. This is a well-documented practical adjustment with no theoretical guarantee.

**Mirror Mode: Where the Analogies Break**

The chef-critic analogy breaks at the nature of the feedback signal. In the story, the critic tells the chef specifically what was wrong — "the sauce texture was off, the plating was asymmetric." The discriminator gives the generator only a number: how likely the output was classified as real. The generator receives no explanation of what to fix; it must infer the corrections purely from the gradient of that number with respect to its own parameters. The critic in the analogy is articulate; the discriminator is a black-box loss signal.

The forger-examiner analogy breaks at the goal of the game. The forger's goal is purely to deceive — to produce documents that look real. A GAN generator's goal is to learn the distribution of the real data well enough to generate samples from it. A perfect forger produces one undetectable document. A perfect GAN generator samples uniformly and accurately from the entire real data distribution. The analogy captures the adversarial structure but misrepresents the generative objective.

---

### Case 8 — Application Exercise

**Exercise: Observing mode collapse**

Using the toy GAN above, modify the target distribution to be bimodal: generate real samples that are 50% from `N(2, 0.3)` and 50% from `N(-2, 0.3)`.

1. Train the GAN for 2,000 steps.
2. Generate 1,000 samples and plot a histogram.
3. Does the generator reproduce both modes, or does it collapse to only one?
4. Try reducing the discriminator's capacity (fewer neurons) or increasing the generator's. Does the balance change?

**Success condition:** You observe mode collapse happening, can identify it from the histogram, and can explain why the generator finds it easier to learn one mode than both.

---
