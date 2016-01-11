#!/usr/bin/python
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys
import time
import cPickle
import argparse

from sampler import sampler
from rnnsearch import rnnsearch
from utils import tokenize, numberize, normalize

# load model from file
def loadmodel(name):
    fd = open(name, 'r')
    option = cPickle.load(fd)
    params = cPickle.load(fd)
    model = rnnsearch(**option)

    for val, param in zip(params, model.parameter):
        param.set_value(val)

    fd.close()

    return model

def processdata(data, voc):
    data = [tokenize(item) + ['<eos>'] for item in data]
    data = numberize(data, voc)
    data = normalize(data)

    return data[0]

def parseargs(args = None):
    desc = 'translate using exsiting nmt model'
    parser = argparse.ArgumentParser(description = desc)

    # input model
    desc = 'trained model'
    parser.add_argument('--model', required = True, help = desc)
    # beam size
    desc = 'beam size'
    parser.add_argument('--beam-size', default = 10, type = int, help = desc)
    # max length
    desc = 'max translation length'
    parser.add_argument('--maxlen', type = int, help = desc)
    # min length
    desc = 'min translation length'
    parser.add_argument('--minlen', type = int, help = desc)

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parseargs()

    model = loadmodel(args.model)

    option = {}
    option['size'] = args.beam_size
    option['maxlen'] = args.maxlen
    option['minlen'] = args.minlen
    mdecoder = sampler(model, **option)

    option = model.option

    svocabs, tvocabs = option['vocabulary']
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    count = 0

    while True:
        line = sys.stdin.readline()

        if line == '':
            break

        data = [line]
        sentence = processdata(data, svocab)
        t1 = time.time()
        tlist = mdecoder.decode(sentence)
        t2 = time.time()

        if len(tlist) == 0:
            sys.stdout.write('\n')
            score = -10000.0
        else:
            best, score = tlist[0]
            sys.stdout.write(' '.join(best[:-1]))
            sys.stdout.write('\n')

        count = count + 1
        sys.stderr.write(str(count) + ' ')
        sys.stderr.write(str(score) + ' ' + str(t2 - t1) + '\n')
