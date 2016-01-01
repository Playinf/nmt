#!/usr/bin/python
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys
import time
import cPickle
import argparse

from nmt import nmtmodel, decoder
from utils import tokenize, numberize, normalize

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

# load model from file
def loadmodel(name):
    fd = open(name, 'r')
    option = cPickle.load(fd)
    params = cPickle.load(fd)
    model = nmtmodel(**option)

    for val, param in zip(params, model.parameter):
        param.set_value(val)

    fd.close()

    return model

def processdata(data, voc):
    data = [tokenize(item) + ['</s>'] for item in data]
    data = numberize(data, voc)
    data, mask = normalize(data)

    return data, mask

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

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parseargs()

    if args.threshold == None:
        args.threshold = -1.0

    model = loadmodel(args.model)
    mdecoder = decoder(model, args.beam_size, args.threshold)
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
        xdata, xmask = processdata(data, svocab)
        t1 = time.time()
        hlist = mdecoder.decode(xdata, xmask)
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
