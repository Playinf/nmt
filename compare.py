import numpy
import theano

from trainnmt import loadmodel
from rnnsearch import rnnsearch

def readdata(name):
    fd = open(name, 'r')
    src = fd.readline().strip().split()
    tgt = fd.readline().strip().split()
    
    return src, tgt
    
def preparedata(data, voc):
    data = [voc[s] if s in voc else voc['UNK'] for s in data]
    ndata = numpy.zeros((len(data), 1)).astype('int32')
    
    for i, id in enumerate(data):
        ndata[i] = id
        
    return ndata
    
def randmodel():
    opt = {}
    opt['embdim'] = [62, 62]
    opt['hidden'] = [100, 100, 100]
    opt['maxhid'] = 50
    opt['deephid'] = 62
    opt['maxpart'] = 2
    svoc = {}
    isvoc = {}

    for i in range(160):
        svoc[i] = i
        isvoc[i] = i
    
    opt['vocabulary'] = [[svoc, isvoc], [svoc, isvoc]]
    
    model = rnnsearch(**opt)
    
    return model

def reference():
    model = loadmodel('nmt.pkl')
    compute_cost = theano.function(model.input, model.output)
    compute_grad = theano.function(model.input, model.gradient)
    src, tgt = readdata('sentence')
    src = preparedata(src, model.vocabulary[0][0])
    tgt = preparedata(tgt, model.vocabulary[1][0])
    xmask = numpy.ones_like(src).astype('float32')  
    ymask = numpy.ones_like(tgt).astype('float32')
    
    loss = compute_cost(src, xmask, tgt, ymask)
    grad = compute_grad(src, xmask, tgt, ymask)
    
if __name__ == '__main__':
    model = randmodel()
    compute_cost = theano.function(model.input, model.output)
    compute_grad = theano.function(model.input, model.gradient)
    src = numpy.random.randint(0, 160, (3, 1)).astype('int32')
    tgt = numpy.random.randint(0, 160, (3, 1)).astype('int32')
    xmask = numpy.ones_like(src).astype('float32')  
    ymask = numpy.ones_like(tgt).astype('float32')
    
    loss = compute_cost(src, xmask, tgt, ymask)
    grad = compute_grad(src, xmask, tgt, ymask)
    
