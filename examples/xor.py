from mlcl import Tensor, Linear, sigmoid, relu, BinaryCrossEntropyLoss, SGD, OpenCLManager

X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=True)
y = Tensor([[0], [1], [1], [0]])

layer1 = Linear(2, 8)
layer2 = Linear(8, 1)

criterion = BinaryCrossEntropyLoss()
optimizer = SGD([layer1.weights, layer1.bias, layer2.weights, layer2.bias], 
               learning_rate=0.1,
               momentum=0.9,
               clip_value=1.0)

for epoch in range(1000):
    h1 = relu(layer1.forward(X))
    out = sigmoid(layer2.forward(h1))
    loss = criterion(out, y)
    
    if epoch % 10 == 0:
        print('Epoch', epoch, 'Loss', loss.data[0], end='\r')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("\nFinal predictions:")
print("Input [0, 0] ->", out.data[0][0])
print("Input [0, 1] ->", out.data[1][0])
print("Input [1, 0] ->", out.data[2][0])
print("Input [1, 1] ->", out.data[3][0])
