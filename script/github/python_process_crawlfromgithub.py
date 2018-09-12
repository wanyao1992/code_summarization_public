
# Find filepaths to files containing a string, such as: "# "
# grep -r -l --include \*.py "# "
# grep -r -l --include \*.py '"""'

import subprocess
import sys
# import getDocStrings
import fileinput
import re
import numpy as np
from os.path import basename, splitext
import util

dataDir = "/media/BACKUP/ghproj_d/code_summarization/github-python/"

repos = ["django"] # , "pandas", "pylearn2", "salt", "scikit-learn", "edx-platform",
originalPath = dataDir + "original/"
processedPath = dataDir + "processed/"
trainPath = dataDir + "train/"
# processedPath = dataDir + "processed/"
# readableFile = dataDir + "processed/"

# commentCodeExt = ".commentCode"
# commentExt = ".comment"
# docstringCodeExt = ".dsCode"
# docstringExt = ".ds"
commentList = ["# ", "#!"]
commentExceptions = ["todo","to do"]
# the largest bucket, no need to get code-comment pairs larger than this
maxBucket = [40,50]

def generate_pairs(filename, repo, module='<string>'):
    """ Loop through the source code and filter comments and
    their correspondig code. """
    # source = open(source)
    # sys.exit()
    # open the file
    # if hasattr(source, 'read'):
    #     filename = getattr(source, 'name', module)
    #     module = splitext(basename(filename))[0]
    #     source = source.read()

    lines = open(filename, 'r').read().splitlines()
    normalComments = 0
    inlineComments = 0
    rejectedComments = 0

    i = -1
    count = 0
    # print "source: "
    # print source

    # check each line for comments
    while i < len(lines):
        line = lines[i]

        # check if the line starts with an comment, if so
        # get the comment and code, and skip to the correct line after
        # the comment
        if line.strip()[:2] in commentList:
            (i, success) = filterComment(lines, i, repo, maxBucket)

            if count != 0 and i == count:
                sys.exit(0)

            count = i

            # only increment the count if there was no error
            if success:
                normalComments += 1
            else:
                rejectedComments += 1
            continue

        # check if we have an inline comment
        # if "# " in line.strip():
        #     parts = line.split("# ")


        #     if len(parts) != 2:
        #         pass
        #         # print ">Something is not right, skipping comment"
        #     else:
        #         code = parts[0].strip()
        #         comment = parts[1].strip().replace("#")

        #         if comment != "" and code != "":
        #             inlineComments += 1

        #             with open(commentFile, "a") as commentF:
        #                 commentF.write(comment + "\n!@#$%!@#$%!@#$%!@#$%!@#$%")

        #             with open(codeFile, "a") as codeF:
        #                 codeF.write(code.strip() + "\n!@#$%!@#$%!@#$%!@#$%!@#$%")
        #                 # codeF.write(" ".join([x.strip() for x in code]) + "\n")

        #             # print ">Comment:" , comment
        #             # print ">Code: \n" , code

        # increment by one
        i += 1

    # print "Total comments found: " , normalComments + inlineComments + rejectedComments
    # print "Normal comments: ", normalComments
    # print "Inline comments: ", inlineComments
    # print "Rejected comments: ", rejectedComments
    return (normalComments, inlineComments, rejectedComments)

def filterComment(lines, startLine, repo, maxBucket):
    """ Find the comment at line i in the list source. When found check for
    a multiline comment and get the corresponding code """

    comment = ""
    indentation = -1
    currIndent = -1
    code = []
    globalI = len(lines) + 10

    # loop through all the lines in the source, get the comment
    # and the corresponding code
    with open(processedPath + repo +  ".comment", "a") as commentF:
        with open(processedPath + repo +  ".commentCode", "a") as codeF:
            for i in xrange(startLine, len(lines)):

                globalI = i
                line = lines[i]

                # comments need to be directly above code
                if line.strip() == "" and comment == "":
                    return (i,False)

                # Continue if we have an divider row
                if line.replace("#", "").strip() == "" and line.strip() != "":
                    continue

                # check if it is an comment, and if so add it to the comment
                if line.strip()[:2] in commentList:
                    comment += line.strip().replace("#", "") + " "
                    continue

                # lines with docstrings are skipped
                if '"""' in line or "'''" in line:
                    return (i,False)

                # if we get here, it means we are not in the comment anymore
                # First get the indentation level of the current line of code
                currIndent = len(line) - len(line.lstrip())

                # If it is the first line of code, set our indentation level
                if indentation == -1:
                    indentation = currIndent

                # if we hit an empty line and have no code yet, return with an error
                if line.strip() == "" and code == []:
                    return (i,False)

                # if we hit an empty line or go to an parent piece in the code
                # return the gathered code
                if line.strip() == "" or indentation > currIndent or (any(c in line for c in commentList)):
                    code = util.cleanCode(code)

                    # no need to save code-comment pairs larger than maxBucket size
                    if util.tokenize("".join(code)) < maxBucket[0] and util.tokenize(comment) < maxBucket[1] \
                    and not (any(exc in comment.lower() for exc in commentExceptions)):
                        # write to file
                        for j in xrange(len(code)):
                            codeF.write(code[j] + "\n")
                        codeF.write("!@#$%!@#$%!@#$%!@#$%!@#$%")
                        commentF.write(util.cleanComment(comment) + "\n!@#$%!@#$%!@#$%!@#$%!@#$%")

                        return (i,True)
                    else:
                        return (i,False)

                # add the line to our code if all is well (without any inline comments if any)
                if line.strip() != "":
                    code.append(line)

            code = util.cleanCode(code)

            # if we are here check if we have a comment / code not empty and smaller than maxBucket size
            if comment.strip() != "" and code != [] and \
            util.tokenize("".join(code)) < maxBucket[0] and util.tokenize(comment) < maxBucket[1] \
            and not (any(exc in comment.lower() for exc in commentExceptions)):
                # write to file
                for j in xrange(len(code)):
                    codeF.write(code[j] + "\n")
                codeF.write("!@#$%!@#$%!@#$%!@#$%!@#$%")
                commentF.write(util.cleanComment(comment) + "\n!@#$%!@#$%!@#$%!@#$%!@#$%")

                return (globalI+1,True)
            else:
                return (globalI+1,False)
#

# retrieve a file list of files with comments and docstrings in the repo
def getFileList(repo):
    try:
        # get lists of all files with comments in the repo
        comments = subprocess.check_output(["grep -r -l --include \*.py '# ' " + repo], shell=True)
        # print "comments: "
        # print comments
        # sys.exit()
        files_w_comments = comments.splitlines()
        print("Found %d files with comments" % len(files_w_comments))

        # # get a list of all files with doc strings
        # doc_strings = subprocess.check_output(["grep -r -l --include \*.py '\"\"\"' " + repo], shell=True)
        # files_w_doc_strings = doc_strings.splitlines()
        # print "Found %d files with doc strings" % len(files_w_doc_strings)
    except:
        print("Unexpected error, most likely no doc strings or comments found. Does the repo exist? \n The error:", sys.exc_info()[0])
        sys.exit(0)

    return files_w_comments


# get the block comment - code pairs
# def getCommentPairs(files_w_comments, repo):
#     # set file names and empty files
#     # codeFile = processedPath + repo +  commentCodeExt
#     # commentFile = processedPath + repo + commentExt
#     # open(codeFile, 'w').close()
#     # open(commentFile, 'w').close()
#     codeFile = open(processedPath + repo +  commentCodeExt, 'w')
#     commentFile = open(processedPath + repo + commentExt, 'w')
#     # counter = 0
#
#     # loop through all files with block comments
#     print "\nBlock comments:"
#     # normalComments = 0
#     # inlineComments = 0
#     # rejectedComments = 0
#     for f in files_w_comments:
#         # print "File " , counter, ":", file
#         # counter += 1
#
#         with open(f) as fp:
#             getComments.generate_pairs(fp, codeFile, commentFile, maxBucket)
#             # (x, y, z) = getComments.generate_pairs(fp, codeFile, commentFile, maxBucket)
#             # normalComments += x
#             # inlineComments += y
#             # rejectedComments += z
#
#     # print "Total comments found: " , normalComments + inlineComments + rejectedComments
#     # print "Normal comments: ", normalComments
#     # print "Inline comments: ", inlineComments
#     # print "Rejected comments: ", rejectedComments
#

# # Get the docstring-code pairs
# def getDocStringPairs(files_w_doc_strings, repo):
#     counter = 0
#
#     # set file names and empty files
#     codeFile = processedPath + repo +  docstringCodeExt
#     commentFile = processedPath + repo + docstringExt
#     open(codeFile, 'w').close()
#     open(commentFile, 'w').close()
#
#     # loop through all files with docstrings
#     print "\nDocstrings:"
#     normalDocStrings = 0
#     rejectedDocStrings = 0
#     for file in files_w_doc_strings:
#
#         # print "File " , counter, ":", file
#         counter += 1
#
#         with open(file) as fp:
#             (x,y) = getDocStrings.generate_pairs(fp, codeFile, commentFile, maxBucket)
#             normalDocStrings += x
#             rejectedDocStrings += y
#
#     print "Total docstrings found: " , normalDocStrings + rejectedDocStrings
#     print "Normal docstrings: ", normalDocStrings
#     print "Rejected docstrings: ", rejectedDocStrings


# loop through the repo list and extract all comment-code pairs
# def createCCPair():
#     for repo in repos:
#         # print "\n"
#         # print "-" * 50
#         print "repo:" , repo
#         # print "-" * 50
#
#         # get file list
#         comments_files = getFileList(originalPath + repo)
#         # print "comments_files"
#         # print comments_files
#         # print "files_w_doc_strings"
#         # print len(files_w_doc_strings)
#
#         # extract code-comment pairs
#         # getCommentPairs(files_w_comments, repo)
#         # getDocStringPairs(files_w_doc_strings, repo)
#         for f_name in comments_files:
#             # print "File " , counter, ":", file
#             # counter += 1
#
#             # with open(f) as fp:
#             generate_pairs(f_name, repo)


# convert the raw newline seperated data into a readable format
# def createReadableFormat(file, codeF, commentF, counter):
#     with open(file, "a") as file:
#         for repo in repos:
#
#             codeFile = processedPath + repo +  codeF
#             commentFile = processedPath + repo + commentF
#
#             # read the lines and do some string / list conversion stuff
#             codeLines =  open(codeFile, "r").readlines()
#             codeLines = "".join(codeLines)
#             codeLines = codeLines.split("!@#$%!@#$%!@#$%!@#$%!@#$%")
#             commentLines = open(commentFile, "r").readlines()
#             commentLines = "".join(commentLines)
#             commentLines = commentLines.split("!@#$%!@#$%!@#$%!@#$%!@#$%")
#
#
#             # loop through the lines
#             for i in xrange(len(codeLines)):
#
#                 if "Parameters ----------" in commentLines[i]:
#                     commentLines[i] = commentLines[i].split("Parameters ----------")[0].strip()
#
#                 if codeLines[i].strip() != '' and commentLines[i].strip() != '':
#                     file.write("Pair : " + str(counter) + "\n")
#                     file.write("Comment:" + commentLines[i].strip() + "\n")
#                     file.write("Code:\n" + codeLines[i].rstrip() + "\n\n")
#                     counter += 1
#
#     return counter


# # convert the raw newline seperated data into training files
# def createTrainingFile(eFile, cFile, codeFileExtension, commentFileExtension, counter, repo):
#     with open(eFile, "a") as enFile:
#         with open(cFile, "a") as codeFile:
#
#             # get the processed files in raw format
#             codeFileName = processedPath + repo +  codeFileExtension
#             commentFileName = processedPath + repo + commentFileExtension
#
#             # read the lines and remove annoying spaces / enters and stuff
#             codeLines =  open(codeFileName, "r").readlines()
#             codeLines = "".join(codeLines)
#             codeLines = " ".join(codeLines.split())
#             codelines = "".join(codeLines)
#             codeLines = codeLines.split("!@#$%!@#$%!@#$%!@#$%!@#$%")
#             commentLines = open(commentFileName, "r").readlines()
#             commentLines = "".join(commentLines)
#             commentLines = commentLines.split("!@#$%!@#$%!@#$%!@#$%!@#$%")
#
#             # loop through the lines
#             for i in xrange(len(codeLines)):
#
#                 # any(x in a for x in b)
#
#                 if "Parameters ----------" in commentLines[i]:
#                     commentLines[i] = commentLines[i].split("Parameters ----------")[0].strip()
#
#                 if codeLines[i].strip() != '' and commentLines[i].strip() != '':
#                     codeFile.write(codeLines[i].strip().replace("\n","") + "\n")
#                     enFile.write(commentLines[i].strip().replace("\n","") + "\n")
#                     counter += 1
#
#     return counter

def createTrainingFiles():
    for repo in repos:
        # eFile = processedPath + repo + ".en"
        # cFile = processedPath + repo + ".code"
        #
        # # empty files
        # open(eFile, 'w').close()
        # open(cFile, 'w').close()

        # counter = 0
        # # convert the docstring-code pairs and comment-code pairs into two large files
        # counter = createTrainingFile(enFile, codeFile, commentCodeExt, commentExt, 1, repo)
        # # createTrainingFile(enFile, codeFile, docstringCodeExt, docstringExt, counter, repo)
        with open(processedPath + repo + ".en", "w") as enFile:
            with open(processedPath + repo + ".code", "w") as codeFile:

                # get the processed files in raw format
                # codeFileName = processedPath + repo +  ".commentCode"
                # commentFileName = processedPath + repo + "comment"

                # read the lines and remove annoying spaces / enters and stuff
                codeLines =  open(processedPath + repo +  ".commentCode", "r").readlines()
                codeLines = "".join(codeLines)
                codeLines = " ".join(codeLines.split())
                codelines = "".join(codeLines)
                codeLines = codeLines.split("!@#$%!@#$%!@#$%!@#$%!@#$%")
                commentLines = open(processedPath + repo + ".comment", "r").readlines()
                commentLines = "".join(commentLines)
                commentLines = commentLines.split("!@#$%!@#$%!@#$%!@#$%!@#$%")

                # loop through the lines
                for i in xrange(len(codeLines)):
                    # any(x in a for x in b)

                    if "Parameters ----------" in commentLines[i]:
                        commentLines[i] = commentLines[i].split("Parameters ----------")[0].strip()

                    if codeLines[i].strip() != '' and commentLines[i].strip() != '':
                        codeFile.write(codeLines[i].strip().replace("\n","") + "\n")
                        enFile.write(commentLines[i].strip().replace("\n","") + "\n")


def concatenateTrainingFiles():
    enFileAll = processedPath + "all.en"
    codeFileAll = processedPath + "all.code"

    # Conctatenate all seperate trainingsfile into a single file
    with open(enFileAll, 'w') as enFileAll:
        with open(codeFileAll, 'w') as codeFileAll:
            for repo in repos:
                # get seperate training files of this repo
                enFile = processedPath + repo + ".en"
                codeFile = processedPath + repo + ".code"

                # write the comments to the comment file
                with open(enFile) as enFile:
                    for line in enFile:
                        enFileAll.write(line)

                # write the code to the code file
                with open(codeFile) as codeFile:
                    for line in codeFile:
                        codeFileAll.write(line)


def split_dataset(train_portion, dev_portion, test_portion):
    num = 0
    with open(processedPath + "all.code", 'r') as training_file:
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
    with open(processedPath + "all.code", 'r') as all_file:
        with open(trainPath + "train%s.code" % (train_portion), 'w') as train_file:
            with open(trainPath + "dev%s.code" % (dev_portion), 'w') as dev_file:
                with open(trainPath + "test%s.code" % (test_portion), 'w') as test_file:
                    for a_line in all_file.readlines():
                        if idx in train:
                            train_file.write(a_line)
                        elif idx in dev:
                            dev_file.write(a_line)
                        elif idx in test:
                            test_file.write(a_line)
                        idx += 1

    idx = 0
    with open(processedPath + "all.en", 'r') as all_file:
        with open(trainPath + "train%s.en" % (train_portion), 'w') as train_file:
            with open(trainPath + "dev%s.en" % (dev_portion), 'w') as dev_file:
                with open(trainPath + "test%s.en" % (test_portion), 'w') as test_file:
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
    for repo in repos:
        print("repo:" , repo)
        comments_files = getFileList(originalPath + repo)

        for f_name in comments_files:
            generate_pairs(f_name, repo)

    # print "Converting into readable format.."
    # # empty file
    # file = readableFile + "readable.txt"
    # open(file, 'w').close()
    # counter = createReadableFormat(file, commentCodeExt, commentExt, 1)
    # print "counter: ", counter
    # createReadableFormat(file, docstringCodeExt, docstringExt, counter)

    print("Converting into seperate training files..")
    createTrainingFiles()

    print("Converting into single training file..")
    concatenateTrainingFiles()
    #
    # train_portion = 0.6
    # dev_portion = 0.2
    # test_portion = 0.2
    # split_dataset(train_portion, dev_portion, test_portion)
