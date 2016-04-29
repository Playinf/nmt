# utils.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import cPickle

# load vocabulary from file
def loadvocab(file):
    fd = open(file, 'r')
    vocab = cPickle.load(fd)
    fd.close()
    return vocab

def invertvoc(vocab):
    v = {}
    for k, idx in vocab.iteritems():
        v[idx] = k

    return v

def uniform(params, lower, upper, precision = 'float32'):

    for p in params:
        s = p.get_value().shape
        v = numpy.random.uniform(lower, upper, s).astype(precision)
        p.set_value(v)

def parameters(params):
    n = 0

    for item in params:
        v = item.get_value()
        n += v.size

    return n
