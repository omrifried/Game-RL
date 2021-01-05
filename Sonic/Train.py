import numpy as np
import retro
from collections import deque
from ReplayBuffer import ReplayBuffer
import sys
sys.path.append('../')
from Utils import Preprocess

TRAIN = False
VISUAL = True
DUEL = True
DOUBLE = True

## Determine which algorithm to use
if DUEL and DOUBLE:
    from Agents.DUEL_DDQN_Agent import DQNAgent, DQNAgentPlay
elif DOUBLE and not DUEL:
    from Agents.DDQN_Agent import DQNAgent, DQNAgentPlay
else:
    from Agents.DQN_Agent import DQNAgent, DQNAgentPlay

def main():
    ## Only necessary actions for Sonic
    possibleActions = {
    # ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"],
        ## Left
        0: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ## Right
        1: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ## Down
        2: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ## Down Left (Roll)
        3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        ## Down Right (Roll)
        4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        ## A (Jump)
        5: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ## Down A (Super roll)
        6: [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ## B (Jump)
        7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ## Down B (Super roll)
        8: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ## C (Jump)
        9: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ## Down C (Super roll)
        10: [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    }
    if TRAIN:
        env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1', scenario = 'contest')
        numActions = len(possibleActions)
        imageX = 84
        imageY = 84
        numFrames = 4
        stepLimit = 1000
        inputSize = (imageX, imageY, numFrames)
        numGames = 1000
        bestScore = -np.inf
        memorySize = 10000
        preTrain = 1000
        epsDecay = 2e-4
        targetUpdate = 1250

        ## Initialize objects
        processor = Preprocess(imageX, imageY)
        dqnAgent = DQNAgent(numActions, inputSize, possibleActions, './SonicMain.pth', './SonicTarget.pth', epsDecay, targetUpdate)
        memory = ReplayBuffer(numActions, memorySize, possibleActions)
        ## Prepopulate memory with random actions
        memory.prePopulate(env, processor, preTrain, inputSize)

        ## Play for a specified number of games
        for game in range(numGames):
            done = False
            state = env.reset()
            totalScore = 0.0
            numSteps = 0.0
            prevInfo = {}
            stagnationCount = 0.0
            maxPosition = 0.0
            stepCount = 0.0
            scores = deque(maxlen = 25)

            ## Process the initial frame and stack frames
            stackArray = np.zeros((imageX, imageY))
            stackFrames = deque([stackArray for x in range(numFrames)], maxlen = numFrames)
            processedState = processor.processFrame(state)
            state = processor.stackFrame(stackFrames, processedState, True)

            ## Continue taking actions until the game is done
            while not done:
                if VISUAL:
                    env.render()
                action, actionIndex = dqnAgent.chooseAction(state)
                nextState, reward, done, info = env.step(action)
                position = info['x']
                ## Check for stagnation
                if prevInfo == info:
                    stagnationCount += 1
                else:
                    stagnationCount = 0
                prevInfo = info
                if stagnationCount >= 25:
                    reward -= 1
                ## Determine if Sonic is moving forward
                if position > maxPosition:
                    stepCount = 0.0
                    maxPosition = position
                else:
                    stepCount += 1
                totalScore += reward
                numSteps += 1

                nextState = processor.processFrame(nextState)
                nextState = processor.stackFrame(state, nextState, False)
                memories = memory.createMemory(state, actionIndex, reward, nextState, done)
                memory.addMemory(memories)
                memorySteps = memory.stepCount
                dqnAgent.updateModel(memory)
                state = nextState
                ## End the episode after certain number of steps
                if stepCount >= stepLimit:
                    done = True

            scores.append(totalScore)
            ## Average the score of the last 25 runs
            averageScore = np.mean(np.array(scores))
            if game % 50 == 0 and game > 0:
                dqnAgent.saveModels()
            currEpsilon = dqnAgent.epsilon
            print('Game: {}'.format(game), 'Average reward: {}'.format(averageScore), \
                  'Fartest distance: {}'.format(maxPosition), 'Steps: {}'.format(numSteps), \
                  'Epsilon: {}'.format(currEpsilon))
    else:
        ## Play the game via the learned weights:
        env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1', scenario = 'contest')
        numActions = len(possibleActions)
        imageX = 84
        imageY = 84
        numFrames = 4
        inputSize = (imageX, imageY, numFrames)
        ## Amount of random actions to allow when watching trained agent
        epsilon = 1e-1
        dqnPlay = DQNAgentPlay(numActions, inputSize, possibleActions, './SonicMain.pth', './SonicTarget.pth', epsilon)
        numSteps = 0.0
        processor = Preprocess(imageX, imageY)
        for game in range(1):
            done = False
            state = env.reset()
            totalScore = 0
            stackArray = np.zeros((imageX, imageY))
            stackFrames = deque([stackArray for x in range(numFrames)], maxlen = numFrames)
            processedState = processor.processFrame(state)
            state = processor.stackFrame(stackFrames, processedState, True)
            while not done:
                if VISUAL:
                    env.render()
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
