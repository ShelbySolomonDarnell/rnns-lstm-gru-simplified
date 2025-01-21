#------------------------------------- Basic Imports --------------------------
import sys
import os
import time
import torch
import inspect
import configparser
from collections.abc import Sequence
#----------------------------------- Logging Imports --------------------------
import logging
#----------------------------------- Logging config  --------------------------
logging.basicConfig(filename='logs/werernns.log', encoding='utf-8', level=logging.DEBUG)
tellem = logging.getLogger(__name__)

cfg = configparser.ConfigParser()
cfg.read('settings.cfg')
print('[common.py] Shakespeare dataset location {0}'.format(cfg.get('DATASETS', 'shakespeare')))

'''
This function takes two tensor size objects and the dimension on which they are 
to be concantenated. The size objects are iterated over and listed as match or 
not based on dimension and equality.
'''
def compare_tensor_sizes(tsA, tsB, dim):
    f_name = inspect.stack()[0][3]
    res = 1 
    resTxt = '' 
    ndx = 0
    if tsA == tsB:
        res = 1
        resTxt = 'Sizes are equal {0}'.format(tsA)
    elif len(tsA)==len(tsB):
        for a, b in zip(tsA,tsB):
            if a != b and dim==ndx:
                resTxt += '\n\t{0} ----- {1} dims not equal, but its not necessary '.format(a,b)
            elif a != b and dim != ndx:
                res = 0
                resTxt += '\n\t{0} ----- {1} dims not equal, this is no good'.format(a,b)
            else:
                resTxt += '\n\t{0} ----- {1}'.format(a, b)
            ndx += 1
    else:
        resTxt = 'Sizes are not equal\n\t{0}\n\t{1}'.format(tsA,tsB)
    return res, resTxt
            
def check_tensors(tensors, dim):
    f_name = inspect.stack()[0][3]
    result = 1
    resp = ''
    ndx = 0
    tsize = None
    for tensor in tensors:
        if ndx==0:
            tsize = tensor.shape
        else:
            result, resp = compare_tensor_sizes(tsize, tensor.shape, dim)
        ndx += 1
    return result, resp

def torch_cat_with_check(tensors, dim=0):
    f_name = inspect.stack()[0][3]
    if not isinstance(tensors, Sequence):
        errTxt = '[{0}] I require a sequence'.format(f_name)
        tellem.error(errTxt)
        raise TypeError(errTxt)
    else:
        if len(tensors) < 2:
            tellem.error("[{0}] The list must have more than one tensor.".format(f_name))
            raise Exception("[{0}] The list must have more than one tensor.".format(f_name))
        else:
            tensors_match, errTxt = check_tensors(tensors, dim)
            if tensors_match == 0:
                tellem.error(errTxt)
                raise Exception(errTxt)
    return torch.cat(tensors, dim)
