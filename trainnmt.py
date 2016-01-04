#!/usr/bin/python
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import time
import numpy
import cPickle
import argparse

from trainer import trainer
from rnnsearch import rnnsearch, decoder
from utils import batchstream, tokenize, shuffle, numberize, normalize

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

# write model into file
def serialize(name, model):
    fd = open(name, 'w')
    option = model.option
    params = model.parameter
    pval = [item.get_value() for item in params]
    cPickle.dump(option, fd)
    cPickle.dump(pval, fd)
    fd.close()

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

def uniform(params, lower, upper):
    precision = 'float32'

    for p in params:
        s = p.get_value().shape
        v = numpy.random.uniform(lower, upper, s).astype(precision)
        p.set_value(v)

def processdata(data, voc):
    xdata, ydata = data
    xvocab, yvocab = voc
    xdata = [tokenize(item) + ['<eos>'] for item in xdata]
    ydata = [tokenize(item) + ['<eos>'] for item in ydata]
    xdata = numberize(xdata, xvocab)
    ydata = numberize(ydata, yvocab)
    xdata, xmask = normalize(xdata)
    ydata, ymask = normalize(ydata)

    return xdata, xmask, ydata, ymask

def parseargs(args = None):
    desc = 'training rnnsearch'
    parser = argparse.ArgumentParser(description = desc)

    # training corpus
    desc = 'source and target corpus'
    parser.add_argument('--corpus', nargs = 2, help = desc)
    # training vocabulary
    desc = 'source and target vocabulary'
    parser.add_argument('--vocabulary', nargs = 2, help = desc)
    # output model
    desc = 'saved model'
    parser.add_argument('--model', required = True, help = desc)

    # embedding size
    desc = 'source and target embedding size'
    parser.add_argument('--embdim', nargs = 2, type = int, help = desc)
    # hidden size
    desc = 'source, target and alignment hidden size'
    parser.add_argument('--hidden', nargs = 3, type = int, help = desc)
    # maxout dim
    desc = 'maxout hidden dimension'
    parser.add_argument('--maxhid', default = 500, type = int, help = desc)
    # maxout number
    desc = 'maxout number'
    parser.add_argument('--maxpart', default = 2, type = int, help = desc)
    # deepout dim
    desc = 'deepout hidden dimension'
    parser.add_argument('--deephid', default = 620, type = int, help = desc)

    # epoch
    desc = 'maximum training epoch'
    parser.add_argument('--maxepoch', default = 10, type = int, help = desc)
    # learning rate
    desc = 'learning rate'
    parser.add_argument('--alpha', default = 1e-4, type = float, help = desc)
    # momentum
    desc = 'momentum'
    parser.add_argument('--momentum', default = 0.0, type = float, help = desc)
    # batch
    desc = 'batch size'
    parser.add_argument('--batch', type = int, default = 128, help = desc)
    # training algorhtm
    desc = 'optimizer'
    parser.add_argument('--optimizer', type = str, help = desc)
    # gradient renormalization
    desc = 'gradient renormalization'
    parser.add_argument('--norm', type = int, default = 1.0, help = desc)

    return parser.parse_args(args)

def getoption():
    option = {}

    option['embdim'] = [620, 620]
    option['hidden'] = [1000, 1000, 1000]
    option['maxpart'] = 2
    option['maxhid'] = 500
    option['deephid'] = 620

    option['cost'] = 0
    option['count'] = 0
    option['epoch'] = 0
    option['maxepoch'] = 10
    option['alpha'] = 1e-4
    option['batch'] = 128
    option['optimizer'] = 'rmsprop'
    option['variant'] = 'graves'

    return option

def override(option, args):
    if args.corpus == None:
        raise RuntimeError('error: no training corpus specified')
    if args.vocabulary == None:
        raise RuntimeError('error: no training vocabulary specified')
    scorpus = args.corpus[0]
    tcorpus = args.corpus[1]
    svocab = loadvocab(args.vocabulary[0])
    tvocab = loadvocab(args.vocabulary[1])
    isvocab = invertvoc(svocab)
    itvocab = invertvoc(tvocab)

    option['source_eos_id'] = len(isvocab)
    option['target_eos_id'] = len(itvocab)

    svocab['<eos>'] = option['source_eos_id']
    tvocab['<eos>'] = option['target_eos_id']
    isvocab[option['source_eos_id']] = '<eos>'
    itvocab[option['target_eos_id']] = '<eos>'

    if args.embdim != None:
        option['embdim'] = args.embdim

    if args.hidden != None:
        option['hidden'] = args.hidden

    if args.optimizer != None:
        option['optimizer'] = args.optimizer

    option['maxhid'] = args.maxhid
    option['maxpart'] = args.maxpart
    option['deephid'] = args.deephid

    option['corpus'] = [scorpus, tcorpus]
    option['vocabulary'] = [[svocab, isvocab], [tvocab, itvocab]]
    option['alpha'] = args.alpha
    option['maxepoch'] = args.maxepoch
    option['momentum'] = args.momentum
    option['batch'] = args.batch
    option['norm'] = args.norm

def skipstream(stream, count):
    for i in range(count):
        stream.next()

def getfilename(name):
    s = name.split('.')
    return s[0]

if __name__ == '__main__':
    args = parseargs()
    option = getoption()
    init = True

    if os.path.exists(args.model):
        model = loadmodel(args.model)
        option = model.option
        init = False
    else:
        override(option, args)
        init = True

    svocabs, tvocabs = option['vocabulary']
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    pathname, basename = os.path.split(args.model)
    modelname = getfilename(basename)
    stream = batchstream(option['corpus'], option['batch'])

    skipstream(stream, option['count'])
    epoch = option['epoch']
    maxepoch = option['maxepoch']

    if init:
        model = rnnsearch(**option)
        uniform(model.parameter, -0.01, 0.01)

    mdecoder = decoder(model, size = 10, threshold = -1.0)
    toption = {}
    toption['algorithm'] = option['optimizer']
    toption['variant'] = option['variant']
    toption['constraint'] = ('norm', option['norm'])
    toption['norm'] = True
    modeltrainer = trainer(model, **toption)
    alpha = option['alpha']

    for i in range(epoch, maxepoch):
        totcost = 0.0
        for data in stream:
            shuffle(data)

            xdata, xmask, ydata, ymask = processdata(data, [svocab, tvocab])
            t1 = time.time()
            cost, norm = modeltrainer.train(xdata, xmask, ydata, ymask)
            modeltrainer.update(alpha = alpha)
            t2 = time.time()

            option['count'] += 1
            count = option['count']

            cost = cost / ymask.sum()
            totcost += cost
            print i + 1, count, cost, norm, t2 - t1

            # save model
            if option['count'] % 1000 == 0:
                filename = os.path.join(pathname, modelname + '.autosave.pkl')
                serialize(filename, model)

            option['cost'] = totcost

            if count % 50 == 0:
                ind = numpy.random.randint(0, option['batch'])
                sdata = data[0][ind]
                tdata = data[1][ind]
                xdata = xdata[:, ind:ind + 1]
                hls = mdecoder.decode(xdata)
                if len(hls) > 0:
                    best = hls[0]
                    print sdata
                    print tdata
                    print ' '.join(best.translation[:-1])
                else:
                    print sdata
                    print tdata
                    print 'warning: no translation'

        print '--------------------------------------------------'
        print 'averaged cost: ', totcost / option['count']
        print '--------------------------------------------------'

        # early stopping
        #if i >= 4:
        #    alpha = alpha / 2

        stream.reset()
        option['epoch'] = i + 1
        option['count'] = 0
        option['alpha'] = alpha
        model.option = option

        filename = modelname + '.iter-' + str(option['epoch']) + '.pkl'
        filename = os.path.join(pathname, filename)
        serialize(filename, model)
