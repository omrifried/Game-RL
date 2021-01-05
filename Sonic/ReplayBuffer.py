import numpy as np
from collections import deque

class ReplayBuffer:
    """
    The ReplayBuffer stores previous memories that the agent had. The memories
    are in the form of a (state, action, reward, nextState, done) tuple that
    conveys actions that the agent took at specific states and what their results
    were. The agent uses these memories to update the information it stores for
    states and determine the best action in each state

    @param numActions: the number of actions available to the agent
    @param memorySize: how many memories to store
    @param possibleActions: possible actions that the agent can take
    """
    def __init__(self, numActions, memorySize, possibleActions):
        self.batchSize = 32
        self.possibleActions = possibleActions
        self.memorySize = memorySize
        self.buffer = deque(maxlen = memorySize)
        self.numActions = numActions
        self.stepCount = 0.0

    """
    Prepopulate the memory buffer with random steps so that we can sample from the buffer
    and begin training the model. We create random steps and store them in the buffer
    which then allows us to randomly sample memories and use them for learning

    @param: environment - the game environment (using gym or retro)
    @param: processor - the Preprocess object used for preprocessing and stacking
    @param: stack - the stack we use to stack 4 frames for movement understanding
    @param: pretrain - the amount of random actions to fill the buffer with
    """
    def prePopulate(self, environment, processor, pretrain, inputSize):
        stagnationCount = 0.0
        prevInfo = {}
        for i in range(pretrain):
            if i == 0:
                stackArray = np.zeros((inputSize[0], inputSize[1]))
                stackFrames = deque([stackArray for x in range(inputSize[2])], maxlen = inputSize[2])
                state = environment.reset()
                state = processor.processFrame(state)
                state = processor.stackFrame(stackFrames, state, True)
            ## Take a random action
            chooseAct = random.randrange(self.numActions)
            action = self.possibleActions[chooseAct]
            nextState, reward, done, info = environment.step(action)
            ## Check for stagnation
            if prevInfo == info:
                stagnationCount += 1
            else:
                stagnationCount = 0
            prevInfo = info
            if stagnationCount >= 25:
                reward -= 1
            nextState = processor.processFrame(nextState)
            ## Add the nextState the to the stack and get the stacked version of nextState
            nextState = processor.stackFrame(state, nextState, False)

            ## If we are done with the current training episode
            if done:
                experience = self.createMemory(state, chooseAct, reward, nextState, done)
                self.addMemory(experience)
                ## Reset the game and reinitialize stack
                stagnationCount = 0.0
                prevInfo = {}
                stackArray = np.zeros((inputSize[0], inputSize[1]))
                stackFrames = deque([stackArray for x in range(inputSize[2])], maxlen = inputSize[2])
                state = environment.reset()
                state = processor.processFrame(state)
                state = processor.stackFrame(stackFrames, state, True)
            else:
                ## Cycle through the game
                experience = self.createMemory(state, chooseAct, reward, nextState, done)
                self.addMemory(experience)
                state = nextState

    """
    Create memories that will be stored in the buffer. We group the state, acion, nextState,
    reward, and done flag together as a memory

    @param state: the current state the agent is in
    @param action: the action taken at the current state
    @param reward: the reward due to the action
    @param nextState: the nextState the agent lands in as a result of the chosen action
    @param done: a flag to determine if the game is over
    """
    def createMemory(self, state, action, reward, nextState, done):
        memory = ((state, action, reward, nextState, done))
        return memory

    """
    Add the memory to the buffer so that it can randomly be sampled for learning

    @param memory: the memory we are adding to the buffer
    """
    def addMemory(self, memory):
        self.stepCount += 1
        self.buffer.append(memory)

    """
    Randomly sample memories to avoid state correlation. State correlation will result in our model
    optimizing for one space (it will optimize for the given sequence of states) which can result in
    very poor performance

    @return batchSize number of random memories
    """
    def sampleMemory(self):
        bufferSize = len(self.buffer)
        ## Random sample of memories with batch size specified
        randMemories = np.random.choice(bufferSize, size = self.batchSize, replace = False)
        return [self.buffer[i] for i in randMemories]
