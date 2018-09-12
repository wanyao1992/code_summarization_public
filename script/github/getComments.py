# -*- coding: utf-8 -*-
""" Open a file, find all hashtag comments in the file and get the corresponding code"""

from os.path import basename, splitext
import sys
import util

commentList = ["# ", "#!"]
commentExceptions = ["todo","to do"]

# no need to get code-commet pairs larger than the max bucket
maxBucket = [40, 50]


def generate_pairs(source, codeFile, commentFile, maxBucket, module='<string>'):
    """ Loop through the source code and filter comments and
    their correspondig code. """

    # open the file
    if hasattr(source, 'read'):
        filename = getattr(source, 'name', module)
        module = splitext(basename(filename))[0]
        source = source.read()

    source = source.splitlines()
    normalComments = 0
    inlineComments = 0
    rejectedComments = 0

    i = -1
    count = 0
    # print "source: "
    # print source

    # check each line for comments
    while i < len(source):
        line = source[i]

        # check if the line starts with an comment, if so
        # get the comment and code, and skip to the correct line after
        # the comment
        if line.strip()[:2] in commentList:
            (i, success) = filterComment(source, i, codeFile, commentFile, maxBucket)

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

def filterComment(source, startLine, codeFile, commentFile, maxBucket):
    """ Find the comment at line i in the list source. When found check for
    a multiline comment and get the corresponding code """

    comment = ""
    indentation = -1
    currIndent = -1
    code = []
    globalI = len(source) + 10

    # loop through all the lines in the source, get the comment
    # and the corresponding code
    with open(commentFile, "a") as commentF:
        with open(codeFile, "a") as codeF:
            for i in xrange(startLine, len(source)):

                globalI = i
                line = source[i]

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
# if __name__ == '__main__':
#     import sys
#
#     with open(sys.argv[1]) as fp:
#         make_pairs(fp)
