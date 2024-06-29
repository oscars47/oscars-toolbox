import torch
import torch.nn as nn
import torch.nn.functional as F
# -------------------------------- #
# --- imports for KAN ---- #
from fastkan import FastKANLayer

###################################

#### ----- standard MLP ----- ####
class MLP(nn.Module):
    '''Generalized MLP class for any number of hidden layers and neurons in each layer'''
    def __init__(self, neurons_ls):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()  # initialize list to store the layers
        for i in range(1, len(neurons_ls)):  # build the hidden layers
            self.layers.append(nn.Linear(neurons_ls[i - 1], neurons_ls[i]))

    def forward(self, x):  # propagate input through the network
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)  # Last layer without ReLU for logits
        return x
    
#### ----- standard CNN ----- ####
class TestCNN(nn.Module):
    def __init__(self, num_classes=10, num_channels=3):
        super(TestCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.act3(self.conv3(x))
        x = self.drop3(x)
        x = self.act4(self.conv4(x))
        x = self.pool4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class TestCNNKAN(nn.Module):
    def __init__(self, num_classes=10, num_channels=3):
        super(TestCNNKAN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.kanlayer = FastKANLayer(64, num_classes)
        # self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.act3(self.conv3(x))
        x = self.drop3(x)
        x = self.act4(self.conv4(x))
        x = self.pool4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.kanlayer(x)
        return x
    
class CNN(nn.Module):
    '''Generalized CNN class for any number of convolutional and single linear layer for output'''
    def __init__(self, img_size, in_channels, num_classes, conv_layers):
        '''
        img_size is a tuple containing the height and width of the input images,
        in_channels is the number of channels in the input data,
        num_classes is the number of classes in the dataset,
        conv_layers is a list of tuples containing the number of out_channels, kernel_size, and stride for each convolutional layer'''

        super(CNN, self).__init__()
        self.conv_layers = nn.ModuleList()  # Initialize list to store the convolutional layers
        self.conv_layers.append(nn.Conv2d(in_channels, conv_layers[0][0], conv_layers[0][1], conv_layers[0][2]))  # First convolutional layer
        for i in range(1, len(conv_layers)):  # Build the rest of the convolutional layers
            self.conv_layers.append(nn.Conv2d(conv_layers[i-1][0], conv_layers[i][0], conv_layers[i][1], conv_layers[i][2]))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Define a pooling layer

        self.flatten = nn.Flatten()  # Flatten the data before the linear layers

        # Use a dummy input to determine the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *img_size)
            dummy_output = dummy_input
            for layer in self.conv_layers:
                dummy_output = layer(dummy_output)
                dummy_output = self.pool(dummy_output)  # Apply pooling layer
                print(f"Shape after conv layer and pooling: {dummy_output.shape}")  # Debugging statement
            self.flattened_size = dummy_output.numel()

        # Adding intermediate linear layers to reduce dimensions appropriately
        self.fc1 = nn.Linear(self.flattened_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

        # Debugging statement
        print(f"Flattened size: {self.flattened_size}")

    def forward(self, x):  # Propagate input through the network
        for layer in self.conv_layers:
            x = F.relu(layer(x))
            x = self.pool(x)  # Apply pooling layer
            print(f"Shape after conv layer and pooling: {x.shape}")  # Debugging statement
        x = self.flatten(x)
        print(f"Shape after flattening: {x.shape}")  # Debugging statement
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
###################################
### ------- modified CNN ------- ###
class KCNN(nn.Module):
    '''Generalized CNN class for any number of convolutional and single linear layer for output'''
    def __init__(self, img_size, in_channels, num_classes, conv_layers):
        '''
        img_size is a tuple containing the height and width of the input images,
        in_channels is the number of channels in the input data,
        num_classes is the number of classes in the dataset,
        conv_layers is a list of tuples containing the number of out_channels, kernel_size, and stride for each convolutional layer,
        linear_layers is a list of the number of neurons in each linear layer'''

        super(KCNN, self).__init__()
        self.conv_layers = nn.ModuleList() # initialize list to store the convolutional layers
        self.conv_layers.append(nn.Conv2d(in_channels, conv_layers[0][0], conv_layers[0][1], conv_layers[0][2])) # first convolutional layer
        for i in range(1, len(conv_layers)): # build the rest of the convolutional layers
            self.conv_layers.append(nn.Conv2d(conv_layers[i-1][0], conv_layers[i][0], conv_layers[i][1], conv_layers[i][2]))

        self.flatten = nn.Flatten() # flatten the data before the linear layers
        
        # Calculate the output size after the last convolutional layer
        def calculate_conv_output_size(input_size, kernel_size, stride, padding=0):
            return (input_size - kernel_size + 2 * padding) // stride + 1

        input_height, input_width = img_size

        for layer in conv_layers:
            input_height = calculate_conv_output_size(input_height, layer[1], layer[2])
            input_width = calculate_conv_output_size(input_width, layer[1], layer[2])

        flattened_size = conv_layers[-1][0] * input_height * input_width
        self.outlayer = FastKANLayer(flattened_size, num_classes) # output layer
        nn.Linear(flattened_size, num_classes) # output layer

    def forward(self, x): # propagate input through the network
        for layer in self.conv_layers:
            x = F.relu(layer(x))
        x = self.flatten(x)
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        return x
