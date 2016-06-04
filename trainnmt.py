#!/usr/bin/python
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import math
import time
import numpy
import cPickle
import argparse

from optimizer import optimizer
from data import batchstream, processdata
from metric.bleu import bleu
from model.rnnsearch import rnnsearch, beamsearch

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
    model = rnnsearch(**option)

    for val, param in zip(params, model.parameter):
        param.set_value(val)

    fd.close()

    return model

def loadreferences(names, case = True):
    references = []
    stream = batchstream(names)

    for data in stream:
        newdata= []
        for batch in data:
            line = batch[0]
            words = line.strip().split()
            if case:
                lower = [word.lower() for word in words]
                newdata.append(lower)
            else:
                newdata.append(words)

        references.append(newdata)

    stream.close()

    return references

def validate(scorpus, tcorpus, model, batch):

    if not scorpus or not tcorpus:
        return None

    stream = batchstream([scorpus, tcorpus], batch)
    svocabs, tvocabs = model.vocabulary
    totcost = 0.0
    count = 0

    for data in stream:
        xdata, xmask = processdata(data[0], svocabs[0])
        ydata, ymask = processdata(data[1], tvocabs[0])
        cost = model.compute(xdata, xmask, ydata, ymask)
        cost = cost[0]
        cost = cost * ymask.shape[1] / ymask.sum()
        totcost += cost / math.log(2)
        count = count + 1

    stream.close()

    bpc = totcost / count

    return bpc

def translate(model, corpus):
    fd = open(corpus, 'r')
    svocab = model.option['vocabulary'][0][0]
    trans = []

    for line in fd:
        line = line.strip()
        data, mask = processdata([line], svocab)
        hls = beamsearch(model, data)
        if len(hls) > 0:
            best, score = hls[0]
            trans.append(best[:-1])
        else:
            trans.append([])

    fd.close()

    return trans

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

    # compute bit per cost
    desc = 'compute bit per cost on validate dataset'
    parser.add_argument('--bpc', action = 'store_true', help = desc)
    # validate data
    desc = 'validate dataset'
    parser.add_argument('--validate', type = str, default = None, help = desc)
    # reference
    desc = 'reference data'
    parser.add_argument('--ref', type = str, nargs = '+', help = desc)

    # save frequency
    desc = 'save frequency'
    parser.add_argument('--freq', type = int, default = 1000, help = desc)
    # sample frequency
    desc = 'sample frequency'
    parser.add_argument('--sfreq', type = int, default = 50, help = desc)
    # validate frequency
    desc = 'validate frequency'
    parser.add_argument('--vfreq', type = int, default = 1000, help = desc)

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

    option['eos'] = '<eos>'
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
        init = True

    if args.ref:
        references = loadreferences(args.ref)
    else:
        references = None

    override(option, args)
    svocabs, tvocabs = option['vocabulary']
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    pathname, basename = os.path.split(args.model)
    modelname = getfilename(basename)
    batch = option['batch']
    stream = batchstream(option['corpus'], batch)

    skipstream(stream, option['count'])
    epoch = option['epoch']
    maxepoch = option['maxepoch']
    option['model'] = 'rnnsearch'

    if init:
        model = rnnsearch(**option)
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

    for i in range(epoch, maxepoch):
        totcost = 0.0
        for data in stream:
            xdata, xmask = processdata(data[0], svocab)
            ydata, ymask = processdata(data[1], tvocab)
            t1 = time.time()
            cost, norm = trainer.optimize(xdata, xmask, ydata, ymask)

            if not numpy.isnan(norm) or norm < 1000:
                trainer.update(alpha = alpha)
            elif numpy.isnan(norm):
                print 'warning: nan occured, restore parameters'
                model = loadmodel('nmt.autosave.pkl')
            else:
                print 'not updating parameter', norm

            t2 = time.time()

            option['count'] += 1
            count = option['count']

            cost = cost * ymask.shape[1] / ymask.sum()
            totcost += cost / math.log(2)
            print i + 1, count, cost, norm, t2 - t1

            option['cost'] = totcost

            # save model
            if count % args.freq == 0:
                svars = [p.get_value() for p in trainer.parameter]
                model.option = option
                model.option['shared'] = svars
                filename = os.path.join(pathname, modelname + '.autosave.pkl')
                serialize(filename, model)

            if count % args.vfreq == 0:
                if args.bpc:
                    for ref in args.ref:
                        bpc = validate(args.validate, ref, model, batch)
                    if bpc:
                        print count, 'bpc:', bpc

                if args.validate and references:
                    trans = translate(model, args.validate)
                    bleu_score = bleu(trans, references)
                    print 'bleu:', bleu_score
                    if bleu_score > best_score:
                        best_score = bleu_score
                        svars = [p.get_value() for p in trainer.parameter]
                        model.option = option
                        model.option['shared'] = svars
                        bestname = modelname + '.best.pkl'
                        filename = os.path.join(pathname, bestname)
                        serialize(filename, model)

            if count % args.sfreq == 0:
                ind = numpy.random.randint(0, batch)
                sdata = data[0][ind]
                tdata = data[1][ind]
                xdata = xdata[:, ind:ind + 1]
                hls = beamsearch(model, xdata)
                if len(hls) > 0:
                    best, score = hls[0]
                    print sdata
                    print tdata
                    print ' '.join(best[:-1])
                else:
                    print sdata
                    print tdata
                    print 'warning: no translation'


        print '--------------------------------------------------'

        if args.bpc:
            for ref in args.ref:
                bpc = validate(args.validate, ref, model, batch)
                if bpc:
                    print 'bpc:', bpc

        if args.validate and references:
            trans = translate(model, args.validate)
            bleu_score = bleu(trans, references)
            print i + 1, 'bleu:', bleu_score
            if bleu_score > best_score:
                best_score = bleu_score
                svars = [p.get_value() for p in trainer.parameter]
                model.option = option
                model.option['shared'] = svars
                bestname = modelname + '.best.pkl'
                filename = os.path.join(pathname, bestname)
                serialize(filename, model)

        print 'averaged cost: ', totcost / option['count']
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
        svars = [p.get_value() for p in trainer.parameter]
        model.option = option
        model.option['shared'] = svars
        serialize(filename, model)

    stream.close()
