# plain.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import random

def tokenize(data):
    return data.split()

def numberize(data, voc, unk = 'UNK'):
    newdata = []
    unkid = voc[unk]

    for d in data:
        idlist = [voc[w] if w in voc else unkid for w in d]
        newdata.append(idlist)

    return newdata

def shuffle(data):
    rnum = random.random()

    for d in data:
        random.shuffle(d, lambda : rnum)

def normalize(bat):
    blen = [len(item) for item in bat]

    n = len(bat)
    maxlen = numpy.max(blen)

    b = numpy.zeros((maxlen, n), 'int32')
    m = numpy.zeros((maxlen, n), 'float32')

    for idx, item in enumerate(bat):
        b[:blen[idx], idx] = item
        m[:blen[idx], idx] = 1.0

    return b, m

def processdata(data, voc, eos = '<eos>'):
    data = [tokenize(item) + [eos] for item in data]
    data = numberize(data, voc)
    data, mask = normalize(data)

    return data, mask
