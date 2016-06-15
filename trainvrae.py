#!/usr/bin/python
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import time
import numpy
import cPickle
import argparse

from optimizer import optimizer
from data import batchstream, processdata
from model.vrnnautoenc import vrnnautoenc, beamsearch

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
    model = vrnnautoenc(**option)

    for val, param in zip(params, model.parameter):
        param.set_value(val)

    fd.close()

    return model

def parseargs(args = None):
    desc = 'training rnnsearch'
    parser = argparse.ArgumentParser(description = desc)

    # training corpus
    desc = 'source and target corpus'
    parser.add_argument('--corpus', required = True, help = desc)
    # training vocabulary
    desc = 'source and target vocabulary'
    parser.add_argument('--vocabulary', required = True, help = desc)
    # output model
    desc = 'saved model'
    parser.add_argument('--model', required = True, help = desc)

    # embedding size
    desc = 'source and target embedding size'
    parser.add_argument('--embdim', nargs = 2, type = int, help = desc)
    # hidden size
    desc = 'source, target and alignment hidden size'
    parser.add_argument('--hidden', nargs = 3, type = int, help = desc)
    # latent size
    desc = 'latent varaible dimension'
    parser.add_argument('--latent', default = 2000, type = int, help = desc)
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
    parser.add_argument('--momentum', default = 0, type = float, help = desc)
    # batch
    desc = 'batch size'
    parser.add_argument('--batch', type = int, default = 128, help = desc)
    # training algorhtm
    desc = 'optimizer'
    parser.add_argument('--optimizer', type = str, help = desc)
    # gradient renormalization
    desc = 'gradient renormalization'
    parser.add_argument('--norm', type = float, default = 1.0, help = desc)
    # early stopping
    desc = 'early stopping iteration'
    parser.add_argument('--stop', type = int, default = 0, help = desc)
    # decay factor
    desc = 'decay factor'
    parser.add_argument('--decay', type = float, default = 0.5, help = desc)
    # drop rate
    desc = 'word drop rate'
    parser.add_argument('--drop', type = float, default = 0.6, help = desc)

    # save frequency
    desc = 'save frequency'
    parser.add_argument('--freq', type = int, default = 1000, help = desc)
    # sample frequency
    desc = 'sample frequency'
    parser.add_argument('--sfreq', type = int, default = 50, help = desc)

    return parser.parse_args(args)

def getoption():
    option = {}

    option['embdim'] = [620, 620]
    option['hidden'] = [1000, 1000, 1000]
    option['latent'] = 2000
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
    corpus = args.corpus
    vocab = loadvocab(args.vocabulary)
    ivocab = invertvoc(vocab)

    option['eos_id'] = len(ivocab)

    option['eos'] = '<eos>'
    vocab['<eos>'] = option['eos_id']
    ivocab[option['eos_id']] = '<eos>'

    if args.embdim != None:
        option['embdim'] = args.embdim

    if args.hidden != None:
        option['hidden'] = args.hidden

    if args.latent != None:
        option['latent'] = args.latent

    if args.optimizer != None:
        option['optimizer'] = args.optimizer

    option['maxhid'] = args.maxhid
    option['maxpart'] = args.maxpart
    option['deephid'] = args.deephid

    option['corpus'] = corpus
    option['vocabulary'] = [vocab, ivocab]
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

def dropword(seq, mask, voc, prob):
    sample = numpy.random.binomial(1, prob, seq.shape)
    sample = sample * mask
    sample = sample.astype('bool')
    dseq = seq.copy()
    dseq[sample] = voc['UNK']

    return dseq

if __name__ == '__main__':
    args = parseargs()
    option = getoption()
    init = True

    if os.path.exists(args.model):
        model = loadmodel(args.model)
        option = model.option
        init = False
    else:
        init = True

    override(option, args)
    vocab, ivocab = option['vocabulary']

    pathname, basename = os.path.split(args.model)
    modelname = getfilename(basename)
    batch = option['batch']
    stream = batchstream(option['corpus'], batch)

    skipstream(stream, option['count'])
    epoch = option['epoch']
    maxepoch = option['maxepoch']
    option['model'] = 'rnnsearch'

    if init:
        model = vrnnautoenc(**option)
        uniform(model.parameter, -0.08, 0.08)

    toption = {}
    toption['algorithm'] = option['optimizer']
    toption['variant'] = option['variant']
    toption['constraint'] = ('norm', option['norm'])
    toption['norm'] = True
    toption['initialize'] = option['shared'] if 'shared' in option else False
    trainer = optimizer(model, **toption)
    alpha = option['alpha']

    print parameters(model.parameter)

    best_score = 0.0
    factor = 0.0

    for i in range(epoch, maxepoch):
        totcost = 0.0
        for data in stream:
            seq, mask = processdata(data[0], vocab)
            dseq = dropword(seq, mask, vocab, args.drop)
            t1 = time.time()
            cost, kl, norm = trainer.optimize(seq, dseq, mask, factor)
            trainer.update(alpha = alpha)
            '''
            if not numpy.isnan(norm) or norm < 1000:
                trainer.update(alpha = alpha)
            elif numpy.isnan(norm):
                params = [numpy.array(p.get_value()) for p in model.parameter]
                params = [item * 0.9 for item in params]

                for p1, p2 in zip(params, model.parameter):
                    p2.set_value(p1)
            else:
                print 'not updating parameter', norm
            '''

            t2 = time.time()

            option['count'] += 1
            count = option['count']

            print i + 1, count, cost, factor, kl, norm, t2 - t1

            if numpy.isnan(cost) or numpy.isinf(cost):
                cost = 0.0

            totcost += cost
            option['cost'] = totcost

            factor = factor + 0.001

            # save model
            if count % args.freq == 0:
                svars = [p.get_value() for p in trainer.parameter]
                model.option = option
                model.option['shared'] = svars
                filename = os.path.join(pathname, modelname + '.autosave.pkl')
                serialize(filename, model)

            if count % args.sfreq == 0:
                #factor = factor + (factor + 0.005) ** 2
                ind = numpy.random.randint(0, batch)
                sentence = data[0][ind]
                seq = seq[:, ind:ind + 1]
                hls = beamsearch(model, seq)
                if len(hls) > 0:
                    best, score = hls[0]
                    print sentence
                    print ' '.join(best[:-1])
                else:
                    print sentence
                    print 'warning: no translation'

            if factor > 1.0:
                factor = 1.0

        print '--------------------------------------------------'
        print 'averaged lowerbound: ', totcost / option['count']
        print '--------------------------------------------------'

        # early stopping
        if i >= args.stop:
            alpha = alpha * args.decay

        stream.reset()
        option['epoch'] = i + 1
        option['count'] = 0
        option['alpha'] = alpha
        model.option = option

        filename = modelname + '.iter-' + str(option['epoch']) + '.pkl'
        filename = os.path.join(pathname, filename)
        serialize(filename, model)

    stream.close()
