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


# this function adds spaces around all punctuation in the specified file, and writes it to the output file
def space_punctuation(): 

    dev_file = "../data/allCodeCommentOnly/dev/10pt.random.en"
    output_dv_file = "../data/allCodeCommentOnly/dev/10pt.random.spaced.en"

    # open files
    with open(dev_file) as dev_file:
        with open(output_dv_file, 'w') as output_dev_file:

            # get initial sentences
            dev_sent = dev_file.readline()

            # loop while we haven't reached the EOF
            while(dev_sent):

                # add spaces around punctuation
                dev_sent = re.sub('([.,`\'\"\[\]\{\}\(\)])', r' \1 ', dev_sent)
                dev_sent = re.sub('\s{2,}', ' ', dev_sent)

                # write the new lines to the files
                dev_sent = dev_sent.rstrip()
                output_dev_file.write(dev_sent + "\n")

                # read new sentences
                dev_sent = dev_file.readline()

    print ("Done. Police out.")



if __name__ == "__main__":
    space_punctuation()
