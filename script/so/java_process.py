import subprocess
import sys
import fileinput
import re
import numpy as np
from os.path import basename, splitext
import glob
import os
import argparse

data_dir = "/media/BACKUP/ghproj_d/code_summarization/so-java/"
original_path = "/media/BACKUP/DATA/javaAndroidDB/"
# original_path = data_dir + "original/"
processed_path = data_dir + "processed/"
train_path = data_dir + "train/"

parser = argparse.ArgumentParser(description='java_process.py')
parser.add_argument('-train_portion', type=float, default=0.6)
parser.add_argument('-dev_portion', type=float, default=0.2)

opt = parser.parse_args()

def get_file_list():
    command = """ls %s""" %(original_path)
    print(command)
    files = subprocess.check_output([command], shell=True)
    file_list = files.splitlines()
    return file_list

def clean_code(code):
    code = ''.join(code)
    code = code.replace('\n', ' DCNL ')
    code = code.strip()
    return code

def clean_comment(comment):
    comment = comment.lower()
    comment_splitted = comment.split('.')
    if len(comment_splitted) > 1 and len(comment_splitted[0].split())<2:
        comment = comment_splitted[1]
    else:
        comment = comment_splitted[0]

    comment = comment.replace('how to', '')
    comment = comment.replace('how do i', '')
    comment = comment.replace('how can i', '')
    comment = comment.replace('do ', ' ')
    comment = comment.replace('is it', 'it is')
    comment = comment.replace('is this', 'this is')
    comment = comment.replace("it's", 'it is')
    comment = comment.replace("this's", 'this is')
    comment = comment.replace("that's", 'that is')
    comment = comment.replace("there's", 'there is')
    comment = comment.replace("<br>", '')
    comment = comment.replace("\n", '')
    comment = comment.replace('?', '')
    comment = comment.replace('``', '')
    comment = comment.replace('`', '')
    comment = comment.replace('\"', '')
    comment = comment.replace('\'', '')
    comment = comment.replace('//', '')

    comment = comment.strip().strip('\n') + ' .'

    return comment

def generate_pairs(f_name):
    with open(processed_path + 'all.code', 'a') as code_file:
        with open(processed_path + 'all.comment', 'a') as comment_file:
            with open(original_path + f_name, 'r') as pair_file:
                pair_lines = pair_file.readlines()
                # for i in range(len(pair_lines)):

                comment = clean_comment(pair_lines[0])
                code = clean_code(pair_lines[2:])
                print(f_name, " comment: ", comment.strip('\n'))

                code_file.write(code + '\n')
                comment_file.write(comment + '\n')

def split_dataset(train_portion, dev_portion):
    test_portion = 1 - train_portion - dev_portion
    with open(processed_path + "all.code", 'r') as all_file:
        num = len(all_file.readlines())

    sidx = np.random.permutation(num)
    n_dev = int(np.round(num * dev_portion))
    dev, data = (sidx[:n_dev], sidx[n_dev:])
    print('Number of questions in dev set: %d.' % len(dev))

    pidx = np.random.permutation(len(data))
    n_train = int(np.round(num * train_portion))
    train, test = (data[pidx[:n_train]], data[pidx[n_train:]])
    print('Number of questions in train set: %d.' % len(train))
    print('Number of questions in test set: %d.' % len(test))

    idx = 0
    with open(processed_path + "all.code", 'r') as all_file:
        with open(train_path + "train%s%s%s.code" % (train_portion, dev_portion, test_portion), 'w') as train_file:
            with open(train_path + "dev%s%s%s.code" % (train_portion, dev_portion, test_portion), 'w') as dev_file:
                with open(train_path + "test%s%s%s.code" % (train_portion, dev_portion, test_portion), 'w') as test_file:
                    for a_line in all_file.readlines():
                        if idx in train:
                            train_file.write(a_line)
                        elif idx in dev:
                            dev_file.write(a_line)
                        elif idx in test:
                            test_file.write(a_line)
                        idx += 1

    idx = 0
    with open(processed_path + "all.comment", 'r') as all_file:
        with open(train_path + "train%s%s%s.comment" % (train_portion, dev_portion, test_portion), 'w') as train_file:
            with open(train_path + "dev%s%s%s.comment" % (train_portion, dev_portion, test_portion), 'w') as dev_file:
                with open(train_path + "test%s%s%s.comment" % (train_portion, dev_portion, test_portion), 'w') as test_file:
                    for a_line in all_file.readlines():
                        if idx in train:
                            train_file.write(a_line)
                        elif idx in dev:
                            dev_file.write(a_line)
                        elif idx in test:
                            test_file.write(a_line)
                        idx += 1

def build_vocab(filepath, dst_path, lowercase=True):
    vocab = set()
    with open(filepath) as f:
        for line in f:
            if lowercase:
                line = line.lower()
            vocab |= set(line.split()) # TODO: can use ast token
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

if __name__ == '__main__':
    print("Creating Code-Comment pairs..")
    if os.listdir(processed_path):
        for f in os.listdir(processed_path):
            os.remove(processed_path + f)

    file_list = get_file_list()

    print("=========================")
    print(file_list[0:10])
    print("len of files: ", len(file_list))

    for f_name in file_list:
        generate_pairs(f_name)

    # concatenate multiple files into one file

    split_dataset(opt.train_portion, opt.dev_portion)
    # build_vocab(os.path.join(processed_path, 'all.code'), os.path.join(train_path, 'code_vocab.txt'))
    # build_vocab(os.path.join(processed_path, 'all.code'), os.path.join(train_path, 'code_vocab_cased.txt'), lowercase=False)
    # build_vocab(os.path.join(processed_path, 'all.comment'), os.path.join(train_path, 'comment_vocab.txt'))
    # build_vocab(os.path.join(processed_path, 'all.comment'), os.path.join(train_path, 'comment_vocab_cased.txt'), lowercase=False)
