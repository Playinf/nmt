# prinparams.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys
import cPickle

def loadpkl(name):
    fd = open(name, 'r')
    opt = cPickle.load(fd)
    params = cPickle.load(fd)

    return params

def nparams(params):
    nparam = 0
    for param in params:
        nparam += param.size

    return nparam

if __name__ == '__main__':
    params = loadpkl(sys.argv[1])
    print nparams(params)
