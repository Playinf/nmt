# translate.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys
import time
import cPickle
import argparse

from data import processdata
from model.rnnsearch import rnnsearch, beamsearch

def loadmodel(name):
    fd = open(name, 'r')
    option = cPickle.load(fd)
    params = cPickle.load(fd)
    model = rnnsearch(**option)

    for val, param in zip(params, model.parameter):
        param.set_value(val)

    fd.close()

    return model

def parseargs(args = None):
    desc = 'translate using exsiting nmt model'
    parser = argparse.ArgumentParser(description = desc)

    # input model
    desc = 'trained model'
    parser.add_argument('--model', required = True, help = desc)
    # beam size
    desc = 'beam size'
    parser.add_argument('--beam-size', default = 10, type = int, help = desc)
    # normalize
    desc = 'normalize'
    parser.add_argument('--normalize', action = 'store_true', help = desc)
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

    svocabs, tvocabs = model.option['vocabulary']
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    count = 0

    option = {}
    option['maxlen'] = args.maxlen
    option['minlen'] = args.minlen
    option['beamsize'] = args.beam_size
    option['normalize'] = args.normalize

    while True:
        line = sys.stdin.readline()

        if line == '':
            break

        data = [line]
        seq, mask = processdata(data, svocab)
        t1 = time.time()
        tlist = beamsearch(model, seq, **option)
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
