import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Model(nn.Module):
    """
    The architecture for our model. The model consists of 3 convolution layers
    and 2 flat layers. Additionally, the dueling model has 2 additional flat
    layers that are used to extract the value and advantages measures.

    @param learningRate: the model learning rate
    @param inputSize: a tuple containg the number of frames and new image size
    @param numActions: the number of actions available to the agent
    """
    def __init__(self, learningRate, inputSize, numActions, mainName, targetName):
        super(Model, self).__init__()
        self.learningRate = learningRate
        self.inputSize = inputSize
        self.numActions = numActions
        self.mainModel = mainName
        self.targetModel = targetName

        ## Create network architecture from OpenAI paper
        self.conv1 = nn.Conv2d(in_channels = 4,  out_channels = 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(in_channels = 32,  out_channels = 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1)

        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, self.numActions)

        self.optimizer = T.optim.RMSprop(self.parameters(), lr = learningRate)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    """
    The feedforward algorithm for the network. The feedforward algorithm allows
    us to determine the values and advantages for a state using the weights in
    our model

    @param state: the image we would like to assess
    @return the value of the state and advantages of each action
    """
    def forward(self, state):
        state = T.Tensor(state).to(self.device)
        ## Reshape since torch takes in the channels as the first argument
        state = state.view(-1, self.inputSize[2], self.inputSize[0], self.inputSize[1])
        ## Feedforward network
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))

        ## Flatten the array and push through FC MLP layers
        conv3State = conv3.view(-1, 64 * 7 * 7)
        flat = F.relu(self.fc1(conv3State))
        flatTwo = F.relu(self.fc2(flat))

        ## Values and Advantages from forward pass
        values = self.V(flatTwo)
        advantages = self.A(flatTwo)

        return values, advantages

    def saveModelMain(self):
        print('--- Saving Main Model ---')
        T.save(self.state_dict(), self.mainModel)

    def saveModelTarget(self):
        print('--- Saving Target Model ---')
        T.save(self.state_dict(), self.targetModel)

    def loadModelMain(self):
        print('--- Loading Main Model ---')
        self.load_state_dict(T.load(self.mainModel, map_location = T.device('cpu')))

    def loadModelTarget(self):
        print('--- Loading Target Model ---')
        self.load_state_dict(T.load(self.targetModel, map_location = T.device('cpu')))
