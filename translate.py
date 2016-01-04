#!/usr/bin/python
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys
import time
import cPickle
import argparse

from rnnsearch import rnnsearch, decoder
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
    # threshold
    desc = 'beam search threshold'
    parser.add_argument('--threshold', type = float, help = desc)
    # max length
    desc = 'max translation length'
    parser.add_argument('--maxlen', type = int, help = desc)
    # min length
    desc = 'min translation length'
    parser.add_argument('--minlen', type = int, help = desc)

    return parser.parse_args(args)

if __name__ == '__main__':
    cmd = '--model search_model.converted.pkl --beam-size 10'.split()
    args = parseargs(cmd)

    if args.threshold == None:
        args.threshold = -1.0

    model = loadmodel(args.model)

    option = {}
    option['size'] = args.beam_size
    option['threshold'] = args.threshold
    option['maxlen'] = args.maxlen
    option['minlen'] = args.minlen
    mdecoder = decoder(model, **option)

    option = model.option

    svocabs, tvocabs = option['vocabulary']
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    count = 0

    #fd = open('/home/playinf/Workspace/data/dev-test/sjs/u8_nist02_src.token.plain')
    fd = open('debug.txt')

    while True:
        #line = sys.stdin.readline()
        line = fd.readline()

        if line == '':
            break

        data = [line]
        sentence = processdata(data, svocab)
        t1 = time.time()
        hlist = mdecoder.decode(sentence)
        t2 = time.time()

        if len(hlist) == 0:
            sys.stdout.write('\n')
            score = -10000.0
        else:
            best = hlist[0]
            translation = best.translation[:-1]
            sys.stdout.write(' '.join(translation))
            sys.stdout.write('\n')
            score = best.score

        count = count + 1
        sys.stderr.write(str(count) + ' ')
        sys.stderr.write(str(score) + ' ' + str(t2 - t1) + '\n')
