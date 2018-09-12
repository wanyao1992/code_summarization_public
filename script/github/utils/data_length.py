###########################################################################################################
# Author: Tjalling Haije
# Project: code-to-comment 
# For: Bsc AI, University of Amsterdam
# Date: May, 2016
###########################################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
from subprocess import call

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
buckets_code = [5,10,20,40,100,250,2000]
buck_code = [0,0,0,0,0,0,0]


buckets_en = [10,15,25,50,100,200,2000]
buck_en = [0,0,0,0,0,0,0]


# Calculate the amount of items in each bucket specified above
def calc_buckets(en_file, code_file):
    
    # open files
    with tf.gfile.GFile(en_file, mode="r") as en_f:
        with tf.gfile.GFile(code_file, mode="r") as code_f:

            # read first line
            en_sent = en_f.readline()
            code_sent = code_f.readline()
            print ("reading files")

            # loop while we are not at the EOF
            while (en_sent and code_sent):
            
                # check which bucket the sentence is in
                i = min ([i for i in xrange(len(buckets_code)) if buckets_code[i] > len(code_sent.split())])
                buck_code[i] += 1
                
                j = min ([j for j in xrange(len(buckets_en)) if buckets_en[j] > len(en_sent.split())])
                buck_en[j] += 1
                
                en_sent = en_f.readline()
                code_sent = code_f.readline()
                
            # print results
            print ("English buckets:")   
            for x in xrange(len(buckets_en)):
                print ("Bucket %d has %d items" % (buckets_en[x],buck_en[x]))
                
            print("----------------------------")
            
            print ("Code buckets:")   
            for y in xrange(len(buckets_en)):
                print ("Bucket %d has %d items" % (buckets_code[y],buck_code[y]))
                
            
def main(_):
    calc_buckets("../data/djangoCode/train/90pt.random.ids5000.en","../data/djangoCode/train/90pt.random.ids5000.code")
    
    
if __name__ == "__main__":
  tf.app.run()
