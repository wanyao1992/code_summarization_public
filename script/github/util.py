import re

# tokenize a sentence by splitting at punctuation marks
def tokenize(sentence):
    _WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
    words = []

    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))

    return len([w for w in words if w])

# filter anoying repetitive characters in the comment
def cleanComment(comment):
    # filter consecutive dashes, at least 3 after eachother
    return re.sub(r'([-]{3,})', "", comment)

def cleanCode(code):
	# remove consequtive dashes, and remove the complete line if it was an comment
	for i in xrange(len(code)):

		# catch a mysterious error, and continue
		try:
			code[i] = re.sub(r'([-]{4,})', "", code[i])
		except:
			return code

		if code[i].replace("#","") == "":
			code.pop(i)

	return code
