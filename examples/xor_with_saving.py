from mlcl import Tensor, Linear, sigmoid, relu, BinaryCrossEntropyLoss, SGD, OpenCLManager
from mlcl.core.model_io import ModelIO

model_io = ModelIO("saved_models")

X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=True)
y = Tensor([[0], [1], [1], [0]])

layer1 = Linear(2, 8)
layer2 = Linear(8, 1)

criterion = BinaryCrossEntropyLoss()
optimizer = SGD([layer1.weights, layer1.bias, layer2.weights, layer2.bias], 
               lr=0.1,
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

print("\nFinal predictions before saving:")
print("Input [0, 0] ->", out.data[0][0])
print("Input [0, 1] ->", out.data[1][0])
print("Input [1, 0] ->", out.data[2][0])
print("Input [1, 1] ->", out.data[3][0])

model_params = [layer1.weights, layer1.bias, layer2.weights, layer2.bias]
model_io.save("xor_model", model_params)

new_layer1 = Linear(2, 8)
new_layer2 = Linear(8, 1)

new_params = [new_layer1.weights, new_layer1.bias, new_layer2.weights, new_layer2.bias]
model_io.apply("xor_model", new_params)

h1 = relu(new_layer1.forward(X))
out = sigmoid(new_layer2.forward(h1))

print("\nPredictions after loading saved model:") # just a test
print("Input [0, 0] ->", out.data[0][0])
print("Input [0, 1] ->", out.data[1][0])
print("Input [1, 0] ->", out.data[2][0])
print("Input [1, 1] ->", out.data[3][0])
