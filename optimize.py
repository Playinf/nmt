# optimize.py

import numpy
import theano

def zerosnan(params):
    for param in params:
        param[numpy.isnan(param)] = 0

def gradnorm(grad):
    return numpy.sqrt(sum(numpy.sum(g ** 2) for g in grad))

def scale(params, alpha):
    for param in params:
        param *= alpha

def rmsprop(params, grads, alpha, rho, epsilon, states):
    n = len(params)
    meangrad = states['meangrad']
    meannorm = states['meannorm']

    for i in xrange(n):
        param = params[i]
        grad = grads[i]
        meangrad[i] = rho * meangrad[i] + (1 - rho) * grad
        meannorm[i] = rho * meannorm[i] + (1 - rho) * (grad ** 2)
        denorm = states.meannorm[i] - states.meangrad[i] ** 2 + epsilon
        param = param - alpha * grad / numpy.sqrt(denorm)

def setparams(params1, params2):
    for p1, p2 in zip(params1, params2):
        p1.set_value(p2)

class trainer:

    def __init__(self, model, **option):
        inputs = model.input
        outvars = model.output + model.gradient
        states = {}
        params = []

        states['meangrad'] = []
        states['meannorm'] = []

        for param in model.parameter:
            val = param.get_value()
            params.append(val)
            states['meangrad'].append(numpy.zeros_like(val))
            states['meannorm'].append(numpy.zeros_like(val))

        self.state = states
        self.gradient = None
        self.update = rmsprop
        self.computegrad = theano.function(inputs, outvars)
        self.parameter = params
        self.maxnorm = option['maxnorm']
        self.model = model

    def train(self, *inputs):
        outs = self.computegrad(*inputs)
        self.gradient = outs[1:]

        return outs[0]

    def update(self, alpha, rho = 0.99, epsilon = 1e-8):
        norm = gradnorm(self.gradient)

        if numpy.isnan(norm):
            zerosnan(self.gradient)
            norm = gradnorm(self.gradient)

        if norm > self.maxnorm:
            factor = self.maxnorm / norm
            scale(self.gradient, factor)

        state = self.state
        rmsprop(self.parameter, self.gradient, alpha, rho, epsilon, state)
        setparams(self.model.parameter, self.parameter)

        return norm
