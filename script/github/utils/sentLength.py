testFile = "../data/django/dev/10pt.random"
trainFile = "../data/django/train/90pt.random"

extensions = [".en",".code"]


for extension in extensions:

    with open(testFile + extension) as f:
        testF = f.readlines()

    sentLength = []
    for line in testF
    


    with open(trainFile + extension) as f:
        trainF = f.readlines()




    overlap = len(set(testF) & set(trainF))
    testLines = len(testF)
    overlapPercentage = (overlap / (testLines * 1.0)) * 100

    print "Overlap of " , extension , ' is: ', overlap , '/', testLines , ' = ' , overlapPercentage , ' % '



def spaceLine(dev_sent):
	# add spaces around punctuation
    dev_sent = re.sub('([.,\'\"\[\]\{\}\(\)])', r' \1 ', dev_sent)
    dev_sent = re.sub('\s{2,}', ' ', dev_sent)

    return len(dev_sent.split())