import re
import grammar
import tokenize

def getType(token):

    # print "Token is: '" , token , "'"
    token = token.strip()

     # skip empty spaces
    if token == "":
        return "SPACE"
    elif grammar.is_keyword(token):
        return "KEYWORD"
    elif grammar.is_string_literal(token):
        return "STR_LIT"
    elif grammar.is_operator(token):
        return "OPERATOR"
    elif grammar.is_num_literal(token):
        return "NUM_LIT"
    elif grammar.is_identifier(token):
        return "IDENTIFIER"
    elif grammar.is_delimeter(token):
        return "DELIMETER"
    # elif grammar.is_indent(token):
    #     pass
    elif grammar.is_newline(token):
        return "NEWLINE"
    else:
        return "UNK_TYPE"

