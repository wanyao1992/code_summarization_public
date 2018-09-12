import keyword
import re

newline = ["\n"]
tab = ["\t"]

operators = ['+','-','*','**','/','//','%','<<','>>','&','|','^','~','<','>',
'<=','>=','==','!=','<>']

delimeters = ['(',')','[',']','{','}','@',',',
':','.','`','=',';','+=','-=','*=','/=','//=','%=',
'&=','|=','^=','>>=','<<=','**=']


def is_operator(a):
    if a in operators:
        return True
    return False


def is_delimeter(a):
    if a in delimeters:
        return True
    return False



def is_keyword(a):
    if keyword.iskeyword(a):
        return True
    return False

def is_string_literal(a):
    # check normal quotes
    if re.match("""["][0-9A-Za-z#$%=@!{},`~&*()<>?.:;_|^/+\t\r\n\\\[\]'-]*["]""", a):
        return True

    # check single quotes
    if re.match("""['][0-9A-Za-z#$%=@!{},`~&*()<>?.:;_|^/+\t\r\n\\\[\]'-]*[']""", a):
        return True

    return False

def is_num_literal(a):
    if is_numeric(a):
        return True

    return False

def is_newline(a):
    if a in newline:
        # print "(NEWLINE %s)" % a
        return True
    return False
    
def is_tab(a):
    if a in tab:
        # print "(TAB %s)" % a
        return True
    return False

def is_identifier(a):
    if re.match("^[_a-zA-Z]\w*", a):
        # print "(IDENTIFIER %s)" % a
        return True
    return False


def is_numeric(lit):
    'Return value of numeric literal string or ValueError exception'
 
    # Handle '0'
    if lit == '0': return 0
    # Hex/Binary
    litneg = lit[1:] if lit[0] == '-' else lit
    if litneg[0] == '0':
        if litneg[1] in 'xX':
            return int(lit,16)
        elif litneg[1] in 'bB':
            return int(lit,2)
        else:
            try:
                return int(lit,8)
            except ValueError:
                pass
 
    # Int/Float/Complex
    try:
        return int(lit)
    except ValueError:
        pass

    try:
        return float(lit)
    except ValueError:
        pass

    try:
        return complex(lit)
    except ValueError:
        pass
