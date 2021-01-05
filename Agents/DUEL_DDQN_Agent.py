import torch as T
import numpy as np
import random
from Networks.DUEL_DDQN import Model

class DQNAgent:
    """
    The main agent used in the program. The agent learns the actions to take at
    specific states by updating the Q Values in each state. This agent splits
    the state information into value and advantages (actions). Doing so allows us
    to learn the value of a specific state irrespective of the actions in that
    state. We then use the quality of the available actions and pair that with the
    state value to determine the overall Q Value. The agent also has a target model.
    We use the main model to determine the correct action for the nextState and
    use the target model to determine the action's value. This allows our model
    to converge with more stability

    @param numActions: the number of actions available to the agent
    @param inputSize: the number of frames and new processed image size
    @param possibleActions: possible actions the agent can take
    """
    def __init__(self, numActions, inputSize, possibleActions, mainName, targetName, epsDecay, targetUpdate):
        self.inputSize = inputSize
        self.numActions = numActions
        ## One hot encoded actions
        self.possibleActions = possibleActions
        self.gamma = 0.95
        self.learningRate = 1e-4
        self.epsilon = 1.0
        self.epsilonMin = 1e-1
        self.epsilonDecay = epsDecay
        self.numFrames = 4
        self.stepCounter = 0.0
        ## How frequently to update the target model
        self.updateTarget = targetUpdate
        self.batchSize = 32
        ## Create models. They are the same initially
        self.qModel = Model(self.learningRate, inputSize, numActions, mainName, targetName)
        self.targetModel = Model(self.learningRate, inputSize, numActions, mainName, targetName)

    """
    Using an epsilon greedy approach, we will determine the action to take at a
    certain state.

    @param state: the current state our agent is in
    @return an action (number) and it's one hot encoded counterpart
    """
    def chooseAction(self, state):
        coinFlip = np.random.rand()
        explore = self.epsilon
        ## Epsilon greedy approach to choose action
        if explore > coinFlip:
            chooseAct = random.randrange(self.numActions)
        else:
            values, advantages = self.qModel.forward(state)
            chooseAct = T.argmax(advantages).item()
        ## One hot encoding of action chosen
        action = self.possibleActions[chooseAct]
        return action, chooseAct

    """
    The main training function for our agent. We use this function to update the
    model that our agent has by updating the Q Values at specific states. Using
    the updated values, we can update our model via gradient descent.

    @param memoryBuffer: the replay buffer containing our agent's memories
    """
    def updateModel(self, memoryBuffer):
        actions = []
        rewards = []
        dones = []
        ## Will hold batchSize samples of images from memory buffer (that have shape of inputSize)
        states = np.zeros(((self.batchSize,) + (self.inputSize[2], self.inputSize[0], self.inputSize[1])), dtype = np.float32)
        nextStates = np.zeros(((self.batchSize,) + (self.inputSize[2], self.inputSize[0], self.inputSize[1])), dtype = np.float32)
        ## Random sample of memories
        memories = memoryBuffer.sampleMemory()
        for i in range(len(memories)):
            actions.append(memories[i][1])
            rewards.append(memories[i][2])
            dones.append(memories[i][4])
            states[i, :, :, :] = (memories[i][0])
            nextStates[i, :, :, :] = (memories[i][3])
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        rewards = T.tensor(rewards, dtype = T.float).to(self.qModel.device)
        dones = T.tensor(dones).to(self.qModel.device)
        batchIndex = np.arange(self.batchSize)

        ## Determine values and advantages for the predictor and target network
        qVals, qAdvantages = self.qModel.forward(states)
        qValsNext, qAdvantagesNext = self.qModel.forward(nextStates)
        targetVals, targetAdvantages = self.targetModel.forward(nextStates)

        ## Determine the Q Values based on Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        qPredict = T.add(qVals, (qAdvantages - qAdvantages.mean(dim = 1, keepdim = True)))
        qPredictNext = T.add(qValsNext, (qAdvantagesNext - qAdvantagesNext.mean(dim = 1, keepdim = True)))
        targetPredictNext = T.add(targetVals, (targetAdvantages - targetAdvantages.mean(dim = 1, keepdim = True)))

        ## Get the max action for the next state using the main model
        maxActions = T.argmax(qPredictNext, dim = 1)

        qTarget = qPredict.clone()
        ## If the game is over, set the dones flag to 0. Otherwise it is a 1
        dones = T.gt(dones, 0)
        targetPredictNext[dones] = 0.0
        ## Get the reward value of the optimal action in the next state using the target model
        qEquation = targetPredictNext[batchIndex, maxActions]
        ## If it is a terminal state (i.e. dones = 0), then we only use the reward to update.
        ## Otherwise, use the full bellman iteration equation
        qTarget[batchIndex, actions] = rewards + self.gamma * qEquation
        self.updateEpsilon()

        loss = self.qModel.loss(qTarget, qPredict).to(self.qModel.device)
        ## Reset the gradients since we are updating batch-wise
        self.qModel.optimizer.zero_grad()
        loss.backward()
        self.qModel.optimizer.step()
        ## Update the target model after a certain amount of actions
        if self.stepCounter % self.updateTarget == 0:
            self.updateTargetModel()
        self.stepCounter += 1

    def updateEpsilon(self):
        if self.epsilon > self.epsilonMin:
            self.epsilon -= self.epsilonDecay
        else:
            self.epsilon = self.epsilonMin

    def updateTargetModel(self):
        print('--- Updating Target Model ---')
        self.targetModel.load_state_dict(self.qModel.state_dict())

    def saveModels(self):
        self.qModel.saveModelMain()
        self.targetModel.saveModelTarget()

    def loadModels(self):
        self.qModel.loadModelMain()
        self.targetModel.loadModelTarget()

class DQNAgentPlay:
    """
    Once our model is trained, we use the learned weights to play Sonic

    @param numActions: the number of actions available to the agent
    @param inputSize: the number of frames and new processed image size
    @param possibleActions: possible actions the agent can take
    """
    def __init__(self, numActions, inputSize, possibleActions, mainName, targetName, epsilon):
        self.learningRate = 1e-4
        self.qModel = Model(self.learningRate, inputSize, numActions, mainName, targetName)
        self.possibleActions = possibleActions
        self.qModel.loadModelMain()
        self.numActions = numActions
        self.epsilon = epsilon

    """
    Using an epsilon greedy approach, we will determine the action to take at a
    certain state.

    @param state: the current state our agent is in
    @return an action (number) and it's one hot encoded counterpart
    """
    def chooseAction(self, state):
        coinFlip = np.random.rand()
        explore = self.epsilon
        ## Epsilon greedy approach to choose action
        if explore > coinFlip:
            chooseAct = random.randrange(self.numActions)
        else:
            values, advantages = self.qModel.forward(state)
            chooseAct = T.argmax(advantages).item()
        ## One hot encoding of action chosen
        action = self.possibleActions[chooseAct]
        return action, chooseAct
