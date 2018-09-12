###########################################################################################################
# Author: Tjalling Haije
# Project: code-to-comment 
# For: Bsc AI, University of Amsterdam
# Date: May, 2016
###########################################################################################################

import os
import sys
import subprocess
import re
import random


# this function adds spaces around all punctuation in the specified file, and writes it to the output file
def gen_random_dataset(): 

    input_code_file = "../data/allCodeCommentOnly/all.code"
    input_en_file = "../data/allCodeCommentOnly/all.en"
    output_dev_en_file = "../data/allCodeCommentOnly/dev/10pt.random.en"
    output_dev_code_file = "../data/allCodeCommentOnly/dev/10pt.random.code"
    output_train_code_file = "../data/allCodeCommentOnly/train/90pt.random.code"
    output_train_en_file = "../data/allCodeCommentOnly/train/90pt.random.en"


    # read the original files
    with open(input_code_file) as f:
        code_file = f.readlines()
    with open(input_en_file) as g:
        en_file = g.readlines()

    fileLength = len(code_file)
    # create a list with all lines shuffled
    indices = random.sample(range(0, fileLength), fileLength)

    # open the dev files
    with open(output_dev_en_file, 'w') as dev_en_file:
        with open(output_dev_code_file, 'w') as dev_code_file:
                
            # write 10% random lines to the dev files
            for x in xrange(0, int(round(0.1 * fileLength))):
                dev_en_file.write(en_file[indices[x]])
                dev_code_file.write(code_file[indices[x]])

    print ("Dev files created.")

    # open the train files
    with open(output_train_en_file, 'w') as train_en_file:
        with open(output_train_code_file, 'w') as train_code_file:

            # write 90% random lines to the train files
            for x in xrange(int(round(0.1 * fileLength)), fileLength):
                train_en_file.write(en_file[indices[x]])
                train_code_file.write(code_file[indices[x]])

    print ("Train files created.")
    print ("Done.")


# def main(_):
#     gen_random_dataset()
#     # os.system("python punctuation_police.py")




if __name__ == "__main__":
    gen_random_dataset()
