# nn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano
import itertools

# representing embedding layer
class embedding:

    def __init__(self, embnum, embdim):
        emb = numpy.random.uniform(-0.08, 0.08, (embnum, embdim))
        emb = theano.shared(emb.astype(theano.config.floatX))

        self.parameter = [emb]

    def __call__(self):
        return self.parameter[0]

class embedder:

    def __init__(self, embdim = None):
        params = []

        if embdim != None:
            bias = numpy.zeros((embdim,))
            bias = theano.shared(bias.astype(theano.config.floatX))
            params.append(bias)

        self.parameter = params

    def __call__(self, emb, indices):
        # assumes:
        # indexs: 1d, (batch,) => (batch, embdim)
        #         2d, (seq, batch) => (seq, batch, embdim)

        if indices.ndim == 1:
            values = emb[indices]
            if len(self.parameter) == 0:
                return values
            else:
                bias = self.parameter[0]
                return values + bias
        elif indices.ndim == 2:
            values = emb[indices.flatten()]
            values = values.reshape((indices.shape[0], indices.shape[1], -1))

            if len(self.parameter) == 0:
                return values
            else:
                bias = self.parameter[0]
                return values + bias
        else:
            raise RuntimeError('indexs must be a 1d or 2d integer array')

# linear with bias or without bias
class linear:

    def __init__(self, *size, **option):
        input_sizes = size[:-1]
        output_size = size[-1]
        params = []

        for input_size in input_sizes:
            w = numpy.random.uniform(-0.08, 0.08, (input_size, output_size))
            w = theano.shared(w.astype(theano.config.floatX))
            params.append(w)

        if 'bias' not in option:
            option['bias'] = True

        if option['bias']:
            b = numpy.zeros((output_size,))
            b = theano.shared(b.astype(theano.config.floatX))
            params.append(b)

        self.size = size
        self.option = option
        self.parameter = params

    def __call__(self, *inputs):
        if self.option['bias']:
            params = self.parameter[:-1]
            bias = self.parameter[-1]
            y = bias

            for x, p in zip(inputs, params):
                y += theano.dot(x, p)

            return y
        else:
            y = theano.dot(inputs[0], self.parameter[0])
            params = self.parameter[1:]
            inputs = inputs[1:]

            for x, p in zip(inputs, params):
                y += theano.dot(x, p)

            return y

# gated recurrent unit
class gru:

    def __init__(self, *size):
        module = []

        module.append(linear(*size))
        module.append(linear(*size, bias = False))
        module.append(linear(*size, bias = False))

        self.module = module
        self.parameter = list(itertools.chain(*[m.parameter for m in module]))

    def __call__(self, *inputs):
        x = list(inputs[:-1])
        h = inputs[-1]
        input_transform = self.module[0]
        reset_gate = self.module[1]
        update_gate = self.module[2]

        r = theano.tensor.nnet.sigmoid(reset_gate(*inputs))
        z = theano.tensor.nnet.sigmoid(update_gate(*inputs))
        t = theano.tensor.tanh(input_transform(*(x + [r * h])))
        h = (1.0 - z) * h + z * t

        return h

class maxout:

    def __init__(self, *size, **option):
        module = []

        if 'maxpart' not in option:
            option['maxpart'] = 2

        module.append(linear(*size, **option))

        self.module = module
        self.option = option
        self.parameter = list(*itertools.chain([m.parameter for m in module]))

    def __call__(self, *inputs):
        k = self.option['maxpart']
        transform = self.module[0]

        z = transform(*inputs)

        z = z.reshape((z.shape[0], z.shape[1] / k, k))
        y = theano.tensor.max(z, 2)

        return y
