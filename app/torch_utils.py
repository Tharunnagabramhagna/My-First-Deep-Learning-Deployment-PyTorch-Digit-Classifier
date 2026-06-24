import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

# step-1 : Load Image

# hyper-parameters

input_size = 784  # 1 batch => 28x28
hidden_size = 500
num_classes = 10

# Custom NeutralNet class


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        # no sigmoid is applied at the end
        return out


model = NeuralNet(input_size, hidden_size, num_classes)

PATH = "app/mnist_ffn.pth"
model.load_state_dict(torch.load(PATH, weights_only=False))
model.eval()

# step-2 : Image -> Tensor

def transform_Image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28,28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081))])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# step-3 : Prediction

def get_prediction(image_tensor):
    images = image_tensor.reshape(-1,28*28)
    output = model(images)
        # max returns (value, index)
    _, predicted = torch.max(output.data, 1)
    return predicted
