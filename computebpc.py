# computebpc.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import math
import cPickle
import argparse

from rnnsearch import rnnsearch
from utils import tokenize, numberize, normalize, batchstream

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
    data, mask = normalize(data)

    return data, mask

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

def parseargs(args = None):
    desc = 'compute bit bper sequence'
    parser = argparse.ArgumentParser(description = desc)

    # training corpus
    desc = 'source and target corpus'
    parser.add_argument('--corpus', nargs = 2, required = True, help = desc)
    # trainded model
    desc = 'trained model'
    parser.add_argument('--model', required = True, help = desc)
    # batch
    desc = 'batch size'
    parser.add_argument('--batch', default = 128, help = desc)

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parseargs()
    model = loadmodel(args.model)
    bpc = validate(args.corpus[0], args.corpus[1], model, args.batch)
    print 'bpc:', bpc
