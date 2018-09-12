###########################################################################################################
# Author: Tjalling Haije
# Project: code-to-comment 
# For: Bsc AI, University of Amsterdam
# Date: May, 2016
###########################################################################################################

import tensorflow as tf
import os
import sys
import subprocess
import re
import copy


# The original Django dataset seemed to have been split wrong at certain comments
# Part of the previous comment ended up in the next comment, divided by three spaces
# this script loops over all sentences, and when it finds three spaces it adds them to 
# the previous comment
def fix_newlined_comments(): 
    with tf.Session() as sess:

        input_file = "../data/django/original/90pt.en"
        output_file = "../data/django/train/90pt.en"
        split_at = "   " # three spaces

        with tf.gfile.GFile(input_file, mode="r") as inp_file:
            with tf.gfile.GFile(output_file, mode="w") as outp_file:

                # get initial sentences
                input_sent = inp_file.readline()
                prev_sent = ""
                counter = 1

                # loop while we haven't reached the EOF
                while(input_sent):

                    # split line at triple space, and add the first part to the previous line
                    line = input_sent.split("   ")
                    if (len(line) > 1):

                        prev_sent = prev_sent.rstrip()
                        length = len(line)

                        # append all comment sections to the previous line
                        # except the last one
                        for x in xrange(0, len(line)-1):
                            prev_sent += " " + line[x]
                        prev_sent += "\n"

                        # our current line is the last item in the list
                        input_sent = line[len(line)-1]

                    # write to the output file and read the next line
                    if (prev_sent != ""):
                        outp_file.write(prev_sent)
                    prev_sent = input_sent
                    input_sent = inp_file.readline()    
                    counter += 1


                # because we write the previous line each time, we need to add the last line at the end
                outp_file.write(prev_sent)

        print "Done"

def main(_):
    fix_newlined_comments()


if __name__ == "__main__":
    tf.app.run()
