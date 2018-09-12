import subprocess
import sys
import fileinput
import re
import numpy as np
from os.path import basename, splitext
import util
import glob
import argparse

data_dir = "/media/BACKUP/ghproj_d/code_summarization/github-python/"
original_path = data_dir + "original/"
processed_path = data_dir + "processed/"
train_path = data_dir + "train/"
parser = argparse.ArgumentParser(description='python_process.py')
parser.add_argument('-train_portion', type=float, default=0.6)
parser.add_argument('-dev_portion', type=float, default=0.2)

opt = parser.parse_args()

def clean_code(declbody):
    # declbody = declbody.replace(' DCNL DCSP', '\n')
    # declbody = declbody.replace(' DCNL ', '\n')
    # declbody = declbody.replace(' DCSP ', '')
    return declbody

def clean_comment(description):
    description = description.replace(' DCNL DCSP', ' ')
    description = description.replace(' DCNL ', ' ')
    description = description.replace(' DCSP ', ' ')

    description = description.lower()

    description = description.replace("this's", 'this is')
    description = description.replace("that's", 'that is')
    description = description.replace("there's", 'there is')

    description = description.replace('\\', '')
    description = description.replace('``', '')
    description = description.replace('`', '')
    description = description.replace('\'', '')

    removes = re.findall("(?<=[(])[^()]+[^()]+(?=[)])", description)
    for r in removes:
        description = description.replace('('+r+')', '')

    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', description)
    for url in urls:
        description = description.replace(url, 'URL')

    description = description.split('.')[0]
    description = description.split(',')[0]
    description = description.split(':param')[0]
    description = description.split('@param')[0]
    description = description.split('>>>')[0]

    description = description.strip().strip('\n') + ' .'

    return description

def generate_pairs():
    with open(processed_path + 'all.code', 'w') as code_file:
        with open(processed_path + 'all.comment', 'w') as comment_file:
            with open(original_path + 'data_ps.declbodies', 'r') as declbodies_file:
                with open(original_path + 'data_ps.descriptions', 'r') as descriptions_file:
                    declbodies_lines = declbodies_file.readlines()
                    descriptions_lines = descriptions_file.readlines()
                    print(len(descriptions_lines))
                    print(len(declbodies_lines))
                    for i in range(len(declbodies_lines)):
                        code = clean_code(declbodies_lines[i])
                        comment = clean_comment(descriptions_lines[i])
                        if not comment.startswith('todo') and len(comment.split())>2 and comment[0].isalpha():
                            code_file.write(code)
                            comment_file.write(comment + '\n')

def split_dataset(train_portion, dev_portion):
    test_portion = 1 - train_portion - dev_portion
    num = 0
    with open(processed_path + "all.code", 'r') as training_file:
        num = len(training_file.readlines())

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

if __name__ == '__main__':
    print("Creating Code-Comment pairs..")
    generate_pairs()
    split_dataset(opt.train_portion, opt.dev_portion)
