import numpy as np
import torch 
import matplotlib.pyplot as plt

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
def mse(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

print(" ====Softmax Output====")
print(softmax(np.array([2.0, 1.0, 0.1])))
print("-" * 30)

print(" ====Autograd Check====")
w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)
b = torch.tensor(0.0, requires_grad=True)
t = torch.tensor(10.0)
loss = (w * x + b - t) ** 2
loss.backward()
print(f"w_grad: {w.grad.item()}")
print(f"b_grad: {b.grad.item()}")
print("-" * 30)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

def train_network(lr, epochs=6000):
    np.random.seed(10)
    W1 = np.random.randn(2, 4) * 2.0
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 2.0
    b2 = np.zeros((1, 1))

    losses = []
    for epoch in range(epochs):
        Z1 = np.dot(X, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)

        losses.append(cross_entropy(y, A2))
        dZ2 = A2 -y
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    return losses, A2

losses_05, final_preds = train_network(lr=0.5)
losses_001, _ = train_network(lr=0.01)
losses_5, _ = train_network(lr=5.0)

print("====XOR Predictions====")
print(np.round(final_preds.flatten(), 2))
print("-" * 30)

plt.figure(figsize=(8, 5))
plt.plot(losses_05, label='lr=0.5')
plt.plot(losses_001, label='lr=0.01')
plt.plot(losses_5, label='lr=5.0')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('XOR Loss Analysis')
plt.savefig('task1_loss_curves.png')

print("System : Plot image generated successfully.")


