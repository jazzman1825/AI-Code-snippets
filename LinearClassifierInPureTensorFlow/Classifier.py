import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],                    #Generate the first class of points: 1000 random 2D points. cov=[[1,
    cov=[[1, 0.5],[0.5, 1]],        #0.5],[0.5, 1]] corresponds to an oval-like point cloud oriented
    size=num_samples_per_class)     #from bottom left to top right.
positive_samples = np.random.multivariate_normal(       
    mean=[3, 0],                            #Generate the other class of
    cov=[[1, 0.5],[0.5, 1]],                #points with a different mean and
    size=num_samples_per_class)             #the same covariance matrix.

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
np.ones((num_samples_per_class, 1), dtype="float32")))

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()

input_dim = 2 #The inputs will be 2D points.
output_dim = 1 #The output predictions will be a single score per sample (close to 0 if the sample is predicted to be in class 0, and close to 1 if the sample is predicted to be in class 1).
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

def model(inputs):
    return tf.matmul(inputs, W) + b

def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions) #per_sample_losses will be a tensor with the same shape as targets and predictions, containing per-sample loss scores.
    return tf.reduce_mean(per_sample_losses) #We need to average these per-sample loss scores into a single scalar loss value: this is what reduce_mean does.

learning_rate = 0.1

def training_step(inputs, targets):
    with tf.GradientTape() as tape: #Forward pass, inside a gradient tape scope
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b]) #Retrieve the gradient of the loss with regard to weights.
    W.assign_sub(grad_loss_wrt_W * learning_rate) #Update the weights
    b.assign_sub(grad_loss_wrt_b * learning_rate) #Update the weights
    return loss

for step in range(100):
    loss = training_step(inputs, targets)
    print(f"Loss at step {step}: {loss:.4f}")

predictions = model(inputs)
x = np.linspace(-1, 4, 100)
y = - W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()