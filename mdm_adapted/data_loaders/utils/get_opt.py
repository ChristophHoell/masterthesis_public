import os
from argparse import Namespace
import re

def get_opt(path, device):
    opt = Namespace()
    opt_dict = vars(opt)
   
    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')

    print("Reading", path)

    with open(path) as f:
        for line in f:
            if line.strip() not in skip:
                key, value = line.strip().split(": ")
                if value in ("True", "False"):
                    opt_dict[key] = value == "True"
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)
        
    opt_dict["which_epoch"] = "latest"

    return opt

def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # Strip sign
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')          # Match to Regex
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # Strip Sign
    if str(numStr).isdigit():
        flag = True
    return flag
