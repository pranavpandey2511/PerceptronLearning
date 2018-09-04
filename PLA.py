import pandas as pd
import numpy as np


def createPoints(N):
    r = np.random.uniform(-1,1,10)
    points = []

    x1, x2, y1, y2 = np.random.choice(r,4)  #Target Function

    for _ in range(N):
        inpx, inpy = np.random.choice(r,2)
        points.append([1,x1,y1,targetFunction(x1,y1,x2,y2, inpx, inpy)])

    return x1, x2, y1, y2, points

def targetFunction(x1,y1,x2,y2, inpx, inpy):
    dscr = ((y2-y1)*(inpx-x1)) - ((inpy-y1)*(x2-x1))
    if (dscr < 0):
        return -1
    else:
        return 1

def hypothesis(x, w):
    return ((x[0]*w[0])+(x[1]*w[1])+(x[2]*w[2]))


def trainPLA(iterationLimit, training_points):
    convCount = 0
    w =[0]*3
    pointsNum = 0
    #learned = False
    randPoints = training_points.copy()

    while(True):
        errorCount = 0
        #for pnt in randPoints:
        pnt = randPoints[np.random.randint(0,len(randPoints)-1)]
        if(np.sign(hypothesis(pnt, w)) != pnt[3]):
            errorCount += 1
            w[0] = w[0] + (pnt[3] * pnt[0])
            w[1] = w[1] + (pnt[3] * pnt[1])
            w[2] = w[2] + (pnt[3] * pnt[2])
        pointsNum += 1
        if (errorCount == 0 and pointsNum <= len(training_points)):
            randPoints.remove(pnt)
            convCount = convCount + 1
        if errorCount > 0:
            convCount = convCount + 1
        if(pointsNum > len(training_points)):
            if ((convCount == iterationLimit) or errorCount == 0):
                break

    return convCount, w

def findErrorProbability(x1,y1,x2,y2, weights, numberOfPointsToTest):
    numberOfErrors = 0
    for i in range(0, numberOfPointsToTest-1):
        #generate random test points
        x = np.random.uniform(-1,1)
        y = np.random.uniform(-1,1)

        #compare results from target function and hypothesis function
        if targetFunction(x1,y1,x2,y2,x,y) != np.sign(hypothesis([1,x,y], weights)):
            numberOfErrors += 1 # keep track of errors
    return numberOfErrors/float(numberOfPointsToTest)

def main():
    N = 100
    x1,x2,y1,y2,training_points = createPoints(N)
    count = []
    errorProb = []
    for _ in range(10):
        c, weight = trainPLA(1000, training_points)
        err = findErrorProbability(x1,x2,y1,y2, weight, N)
        count.append(c)
        errorProb.append(err)
    avgCount = np.mean(count)
    avgError = np.mean(errorProb)
    print("iterations:" + str(avgCount))
    print("Error Probability" + str(avgError))

if __name__ == '__main__':
    main()

