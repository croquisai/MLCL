from mlcl import Tensor, Conv2D, Linear, relu
import numpy as np

class CNN:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = Conv2D(in_channels=16, out_channels=32, kernel_size=5)
        self.fc1 = Linear(32 * 20 * 20, 128)
        self.fc2 = Linear(128, 10)
    
    def forward(self, x):
        x = relu(self.conv1.forward(x))

        x = relu(self.conv2.forward(x))

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        x = relu(self.fc1.forward(x))
        x = self.fc2.forward(x)
        
        return x

def main():
    batch_size = 4
    x = Tensor(np.random.randn(batch_size, 1, 28, 28))

    model = CNN()

    output = model.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
