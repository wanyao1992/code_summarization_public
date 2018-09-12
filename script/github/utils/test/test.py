import os

reference = "10pt.random.spaced.en"
translation = "translated.en"

tempRefF = "tempRF.txt"
tempTransF = "tempTF.txt"

os.system("perl multi-bleu.perl " + reference + " < " + translation)

with open(reference) as f:
    refF = f.readlines()

with open(translation) as f:
    transF = f.readlines()

counter = 0
equal = 0
for i in xrange(len(refF)):
	
	if refF[i] == transF[i]:
		print refF[i] , " -- equal to -- " , transF[i] 

		# open(tempRF,'w') as trf:
		# 	trf.write(refF[i])
		# open(tempTransF,'w') as ttf:
		# 	ttf.write(transF[i])
		equal += 1



	counter += 1

precision = (equal/float(counter)) * 100
print "\nAccuracy is:" , equal , "/", counter , " = " , precision , "%"