import numpy as np
from collections import deque
from skimage.color import rgb2gray
from skimage import transform

class Preprocess:
    """
    Process the initial raw image. We want to convert the image to grayscale, as
    well as lower the number of pixels so that our algorithm uses a less dense
    version of the image for training. This will ultimately speed up training

    @param x: the number of pixels we are scaling down to on x
    @param y: the number of pixels we are scaling down to on y
    """
    def __init__(self, x, y):
        self.xImage = x
        self.yImage = y
        self.numFrames = 4

    """
    Convert the image to grayscale and normalize the pixel values so that it is
    easier for our neural network to learn

    @param frame: the raw initial image
    @return the new grayscale, normalized image matrix
    """
    def processFrame(self, frame):
        ## Convert to grayscale and normalize
        grayFrame = rgb2gray(frame)
        ## Downsize the image
        scaledFrame = transform.resize(grayFrame, [self.xImage, self.yImage])
        ## Convert to float and normalize
        scaledFrame = np.ascontiguousarray(scaledFrame, dtype = np.float32) / 255
        return scaledFrame

    """
    Stack numFrames frames in a row. Stacking frames allows the computer to determine
    motion as we have 4 consecutive images stacked on top of each other. These 4
    images allow the computer to determine motion in the image input

    @param stackFrames: a queue that will hold the 4 stacked images
    @param processedFrame: the grayscale, normalized image matrix
    @param new: a boolean to determine if we are in a new game
    @return the queue containing the 4 stacked images
    """
    def stackFrame(self, stackFrames, processedFrame, new):
        if new:
            ## Clear current stack
            stackArray = np.zeros((self.xImage, self.yImage), dtype = np.float32)
            stackedFrames = deque([stackArray for x in range(self.numFrames)], maxlen = self.numFrames)
            stackedFrames = stackFrames
            ## Populate stack with four identical images for a new game
            for i in range(self.numFrames):
                stackedFrames.append(processedFrame)
        else:
            ## Add new frame to stack and remove oldest
            stackFrames.append(processedFrame)
            stackedFrames = stackFrames
        return stackedFrames
