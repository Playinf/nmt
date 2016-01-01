# nn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano
import itertools

class nnunit:

    def __init__(self, isize, osize, func = theano.tensor.nnet.sigmoid):
        ftype = theano.config.floatX

        w = numpy.random.uniform(-0.05, 0.05, (osize, isize))
        b = numpy.zeros(osize)

        w = theano.shared(w.astype(ftype))
        b = theano.shared(b.astype(ftype))

        self.flag = 0
        self.module = []
        self.function = func
        self.parameter = [w, b]

    def __call__(self, x):
        weight = self.parameter[0]
        bias = self.parameter[1]
        function = self.function
        return function(theano.dot(x, weight.transpose()) + bias)

class gru:
    def __init__(self, isize, hsize, osize):
        gates = nnunit(isize + hsize, 2 * osize)
        transform = nnunit(isize + hsize, osize, theano.tensor.tanh)

        module = []
        module.append(gates)
        module.append(transform)

        params = [item.parameter for item in module]

        self.module = module
        self.parameter = list(itertools.chain(*params))

    def __call__(self, x, h):
        gates = self.module[0]
        transform = self.module[1]

        t = gates(theano.tensor.concatenate([x, h], 1))
        n = t.shape[1] / 2

        r = t[:, :n]
        z = t[:, -n:]
        c = transform(theano.tensor.concatenate([x, r * h], 1))
        h = z * h + (1 - z) * c

        return h
