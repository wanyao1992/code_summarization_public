import subprocess
import sys
import fileinput
import re
import numpy as np
from os.path import basename, splitext
import util
import glob
import os
import argparse

data_dir = "/media/BACKUP/ghproj_d/code_summarization/github-java/"
repos = ["apache-ant-1.8.4", "apache-maven-3.0.4",  "MinorThird", "apache-cassandra-1.2.0", "batik-1.7", "xalan-j-2.7.1", "apache-log4j-1.2.17", "lucene-3.6.2", "xerces-2.11.0"] #
original_path = data_dir + "original/"
processed_path = data_dir + "processed/"
train_path = data_dir + "train/"

parser = argparse.ArgumentParser(description='java_process.py')
parser.add_argument('-train_portion', type=float, default=0.6)
parser.add_argument('-dev_portion', type=float, default=0.2)

opt = parser.parse_args()

def get_file_list(repo):
    command = """ls -R %s|grep .java | awk '
    /:$/&&f{s=$0;f=0}
    /:$/&&!f{sub(/:$/,"");s=$0;f=1;next}
    NF&&f{ print s"/"$0 }'""" %(original_path+repo)
    print(command)
    files = subprocess.check_output([command], shell=True)
    file_list = files.splitlines()
    return file_list

def clean_comment(comment):
    # cleaned_comment = []
    # for c in comment:
    #     if c in ['<p>', '</p>', '<code>', '</code>']:
    #         cleaned_comment.append('')
    #     else:
    #         cleaned_comment.append(c)
    # return ''.join(cleaned_comment)
    comment = comment.lower()
    for c in ['<p>', '</p>', '<code>', '</code>', '<i>', '</i>', '<b>', '</b>', '<br>', '</br>']:
        comment = comment.replace(c, '')

    comment = comment.replace("this's", 'this is')
    comment = comment.replace("that's", 'that is')
    comment = comment.replace("there's", 'there is')

    comment = comment.replace('\\', '')
    comment = comment.replace('``', '')
    comment = comment.replace('`', '')
    comment = comment.replace('\'', '')

    removes = re.findall("(?<=[(])[^()]+[^()]+(?=[)])", comment)
    for r in removes:
        comment = comment.replace('('+r+')', '')
    # print "c: ", comment
    comment = comment.split('.')[0]
    comment = comment.split('*')[0]
    comment = comment.split('@')[0]
    # print "comment: ", comment
    comment = comment.strip().strip('\n') + ' .'
    return comment

def generate_pairs(f_name, repo):
    print(f_name)

    # file_lines = java_file.readlines()
    comment_linenums = []
    with open(f_name, 'r') as java_file:
        file_lines = java_file.readlines()
        line_num = 0
        for line in file_lines:
            if line.split('\t')[1] == 'TokenNameCOMMENT_JAVADOC':
                comment_linenums.append(line_num)
            line_num += 1

    if len(comment_linenums) > 2:
        comment_linenums = comment_linenums[2:-1]

    with open(processed_path + repo + '.code', 'a') as code_file:
        with open(processed_path + repo + '.comment', 'a') as comment_file:
            for i in range(len(comment_linenums)):
                comment = file_lines[comment_linenums[i]].strip('\n').split('\t')[-1].strip()
                try:
                    code = [l.split('\t')[0] for l in file_lines[comment_linenums[i]+1:comment_linenums[i+1]] if not l.split('\t')[1]== 'TokenNameCOMMENT_LINE']
                except:
                    code = [l.split('\t')[0] for l in file_lines[comment_linenums[i]+1:-1] if not l.split('\t')[1]== 'TokenNameCOMMENT_LINE']
                code = ' '.join(code).replace('\n', ' ')
                comment = clean_comment(comment)
                if not comment.startswith('*') and len(comment.split()) > 2 and comment[0].isalpha():
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
    for repo in repos:
        print("repo:" , repo)

        file_list = get_file_list(repo)

        print("=========================")
        print(file_list)

        for f_name in file_list:
            if f_name.endswith('.java'):
                generate_pairs(f_name, repo)

    # concatenate multiple files into one file
    command1 = "cat %s*code > %sall.code" % (processed_path, processed_path)
    command2 = "cat %s*comment > %sall.comment" % (processed_path, processed_path)
    subprocess.check_output([command1], shell=True)
    subprocess.check_output([command2], shell=True)

    split_dataset(opt.train_portion, opt.dev_portion)
    # build_vocab(os.path.join(processed_path, 'all.code'), os.path.join(train_path, 'code_vocab.txt'))
    # build_vocab(os.path.join(processed_path, 'all.code'), os.path.join(train_path, 'code_vocab_cased.txt'), lowercase=False)
    # build_vocab(os.path.join(processed_path, 'all.comment'), os.path.join(train_path, 'comment_vocab.txt'))
    # build_vocab(os.path.join(processed_path, 'all.comment'), os.path.join(train_path, 'comment_vocab_cased.txt'), lowercase=False)
