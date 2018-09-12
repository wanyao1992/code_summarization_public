###########################################################################################################
# Author: Tjalling Haije
# Project: code-to-comment 
# For: Bsc AI, University of Amsterdam
# Date: June, 2016
###########################################################################################################

file = "allCode_sub4_3x512_bleu8.en"

testFile = "../data/edx_pylearn_scikit_salt/dev/10pt.random.en"
testFileSpaced = "../data/edx_pylearn_scikit_salt/dev/10pt.random.spaced.en"
trainFile = "../data/edx_pylearn_scikit_salt/train/90pt.random.en"
transFile = "../evaluation/bleu/bleu_test_data/" + file



with open(testFile) as f:
    testF = f.readlines()

with open(trainFile) as f:
    trainF = f.readlines()

with open(testFileSpaced) as f:
    testSpacedF = f.readlines()

with open(transFile) as f:
	transF = f.readlines()

equal = 0
counter = 0
inOverlap = 0

# loop through all test lines
for i in xrange(len(testF)):

	# check if the annotation is correct
	if transF[i] == testSpacedF[i]:
		equal += 1

		# check if the translation was in the overlap
		if testF[i] in trainF:
			inOverlap += 1
	counter += 1



testLines = len(testF)
precision = (equal / float(counter)) * 100
inOverlapPerc = (inOverlap / float(equal)) * 100

print "Precision:", precision
print "Correct annotations:", equal , " (" , precision , ")"
print "In overlap:", inOverlap , " (", inOverlapPerc , ")"
print "Unique annotations:", equal - inOverlap , " (", 100 - inOverlapPerc , ")"