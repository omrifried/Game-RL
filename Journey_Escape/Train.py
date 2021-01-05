import numpy as np
import gym
import time
from collections import deque
from ReplayBuffer import ReplayBuffer
import sys
sys.path.append('../')
from Utils import Preprocess

TRAIN = False
VISUAL = True
DUEL = True
DOUBLE = True

## Determine which agent to use
if DUEL and DOUBLE:
    from Agents.DUEL_DDQN_Agent import DQNAgent, DQNAgentPlay
elif DOUBLE and not DUEL:
    from Agents.DDQN_Agent import DQNAgent, DQNAgentPlay
else:
    from Agents.DQN_Agent import DQNAgent, DQNAgentPlay

def main():
    ## Only necessary actions for Agent: Forwards, Right, Left, Backwards. Fast Right, Fast Left
    possibleActions = [1, 2, 3, 4, 9, 10]
    if TRAIN:
        env = gym.make('JourneyEscape-v0')
        numActions = len(possibleActions)
        imageX = 84
        imageY = 84
        numFrames = 4
        stepLimit = 15000
        inputSize = (imageX, imageY, numFrames)
        numGames = 1000
        bestScore = -np.inf
        memorySize = 10000
        preTrain = 1000
        epsDecay = 4.5e-6
        targetUpdate = 20000

        ## Initialize objects
        processor = Preprocess(imageX, imageY)
        dqnAgent = DQNAgent(numActions, inputSize, possibleActions, './JourneyMain.pth', './JourneyTarget.pth', epsDecay, targetUpdate)
        memory = ReplayBuffer(numActions, memorySize, possibleActions)
        ## Prepopulate memory with random actions
        memory.prePopulate(env, processor, preTrain, inputSize)

        for game in range(numGames):
            done = False
            state = env.reset()
            totalScore = 0.0
            numSteps = 0.0
            scores = deque(maxlen = 25)

            ## Preprocess the initial frame and stack frames
            stackArray = np.zeros((imageX, imageY))
            stackFrames = deque([stackArray for x in range(numFrames)], maxlen = numFrames)
            processedState = processor.processFrame(state)
            state = processor.stackFrame(stackFrames, processedState, True)

            while not done:
                if VISUAL:
                    env.render()
                action, actionIndex = dqnAgent.chooseAction(state)
                nextState, reward, done, info = env.step(action)
                totalScore += reward
                numSteps += 1
                nextState = processor.processFrame(nextState)
                nextState = processor.stackFrame(state, nextState, False)
                memories = memory.createMemory(state, actionIndex, reward, nextState, done)
                memory.addMemory(memories)
                memorySteps = memory.stepCount
                dqnAgent.updateModel(memory)

                ## Move the state to the next state
                state = nextState
                ## End the episode after certain number of steps
                if numSteps >= stepLimit:
                    done = True
            scores.append(totalScore)
            ## Average the score of the last 25 runs
            averageScore = np.mean(np.array(scores))
            if game % 50 == 0 and game > 0:
                dqnAgent.saveModels()
            currEpsilon = dqnAgent.epsilon
            print('Game: {}'.format(game), 'Average reward: {}'.format(averageScore), \
                  'Steps: {}'.format(numSteps), 'Epsilon: {}'.format(currEpsilon))
    else:
        ## Play the game via the learned weights:
        env = gym.make('JourneyEscape-v0')
        numActions = len(possibleActions)
        imageX = 84
        imageY = 84
        numFrames = 4
        inputSize = (imageX, imageY, numFrames)
        ## Amount of random actions to allow when watching trained agent
        epsilon = 0e-1
        dqnPlay = DQNAgentPlay(numActions, inputSize, possibleActions, './JourneyMain.pth', './JourneyTarget.pth', epsilon)
        numSteps = 0.0
        processor = Preprocess(imageX, imageY)
        for game in range(1):
            done = False
            state = env.reset()
            stackArray = np.zeros((imageX, imageY))
            stackFrames = deque([stackArray for x in range(numFrames)], maxlen = numFrames)
            processedState = processor.processFrame(state)
            state = processor.stackFrame(stackFrames, processedState, True)
            totalScore = 0
            while not done:
                if VISUAL:
                    env.render()
                    ## Slow down visual
                    time.sleep(0.03)
                action, actionIndex = dqnPlay.chooseAction(state)
                nextState, reward, done, info = env.step(action)
                totalScore += reward
                nextState = processor.processFrame(nextState)
                nextState = processor.stackFrame(state, nextState, False)
                state = nextState
                numSteps += 1
            print(totalScore)

if __name__ == "__main__":
    main()
