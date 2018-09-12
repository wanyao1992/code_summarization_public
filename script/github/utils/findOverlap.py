###########################################################################################################
# Author: Tjalling Haije
# Project: code-to-comment 
# For: Bsc AI, University of Amsterdam
# Date: June, 2016
###########################################################################################################

testFile = "../data/allCode/dev/10pt.random"
trainFile = "../data/allCode/train/90pt.random"

extensions = [".en",".code"]


for extension in extensions:

    with open(testFile + extension) as f:
        testF = f.readlines()

    with open(trainFile + extension) as f:
        trainF = f.readlines()

    overlap = 0
    for line in testF:
    	if line in trainF:
    		overlap += 1


    # overlap = len(set(testF) & set(trainF))
    testLines = len(testF)
    overlapPercentage = (overlap / (testLines * 1.0)) * 100

    print "Overlap of " , extension , ' is: ', overlap , '/', testLines , ' = ' , overlapPercentage , ' % '