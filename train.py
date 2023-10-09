import matplotlib.pyplot as plt
import numpy as np

# Simulate some training data
epochs = np.linspace(1, 100, 100)
train_loss = np.exp(-epochs/50) + 0.1 * np.random.randn(100) + 0.2
val_loss = np.exp(-epochs/50) + 0.1 * np.random.randn(100)

def train_model(epochs, train_loss, val_loss):
    pass

def generate_plot(epochs, train_loss, val_loss):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, '-o', label='Training Loss')
    plt.plot(epochs, val_loss, '-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot.png')
    plt.close()

if __name__ == "__main__":
    train_model(epochs, train_loss, val_loss)
    generate_plot(epochs, train_loss, val_loss)
