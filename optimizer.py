# optimizer.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano
from collections import OrderedDict

def grad_norm(grad):
    norm = theano.tensor.sqrt(sum(theano.tensor.sum(g ** 2) for g in grad))
    return norm

def grad_clip(grad, lower, upper):
    return [theano.tensor.clip(x, lower, upper) for x in grad]

def grad_renormalize(grad, threshold, epsilon = 1e-7):
    norm = grad_norm(grad)
    dtype = numpy.dtype(theano.config.floatX).type
    target_norm = theano.tensor.clip(norm, 0, dtype(threshold))
    multiplier = target_norm / (dtype(epsilon) + norm)
    grad = [step * multiplier for step in grad]
    return grad

def apply_momentum(updates, params, momentum):
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow = True)
        var = numpy.zeros(value.shape, dtype = value.dtype)
        velocity = theano.shared(var, broadcastable = param.broadcastable)
        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x

    return updates

def apply_nesterov_momentum(updates, params, momentum):
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        var = numpy.zeros(value.shape, dtype=value.dtype)
        velocity = theano.shared(var, broadcastable = param.broadcastable)
        x = momentum * velocity + updates[param] - param
        updates[velocity] = x
        updates[param] = momentum * x + updates[param]

    return updates

def sgd_updates(params, grads, lr):
    updates = OrderedDict()

    for p, g in zip(params, grads):
        updates[p] = p - lr * g

    return updates

def adagrad_updates(params, grads, lr, epsilon):
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow = True)
        var = numpy.zeros_like(value)
        accu = theano.shared(var, broadcastable = param.broadcastable)
        accu_new = accu + (grad ** 2)
        delta = lr * grad / theano.tensor.sqrt(accu_new + epsilon)
        updates[accu] = accu_new
        updates[param] = param - delta

    return updates

def rmsprop_updates(params, grads, lr, rho, epsilon, variant = 'hinton'):
    updates = OrderedDict()

    if variant == 'hinton':
        for param, grad in zip(params, grads):
            value = param.get_value(borrow = True)
            var = numpy.zeros_like(value)
            accu = theano.shared(var, broadcastable = param.broadcastable)
            accu_new = rho * accu + (1 - rho) * grad ** 2
            delta = lr * grad / (theano.tensor.sqrt(accu_new) + epsilon)
            updates[accu] = accu_new
            updates[param] = param - delta
    elif variant == 'graves':
        for param, grad in zip(params, grads):
            value = numpy.zeros_like(param.get_value(borrow = True))
            accu = theano.shared(value, broadcastable = param.broadcastable)
            gaccu = theano.shared(value, broadcastable = param.broadcastable)

            accu_new = rho * accu + (1 - rho) * (grad ** 2)
            gaccu_new = rho * gaccu + (1 - rho) * grad

            updates[accu] = accu_new
            updates[gaccu] = gaccu_new

            denorm = theano.tensor.sqrt(accu_new - gaccu_new ** 2 + epsilon)
            delta = lr * grad / denorm
            updates[param] = param - delta
    else:
        raise RuntimeError('error: unknown variant')

    return updates

def rmsprop_momentum_updates(params, grads, lr, rho, epsilon, momentum):
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = numpy.zeros_like(param.get_value(borrow = True))
        accu = theano.shared(value, broadcastable = param.broadcastable)
        grad_accu = theano.shared(value, broadcastable = param.broadcastable)
        velocity = theano.shared(value, broadcastable = param.broadcastable)

        accu_new = rho * accu + (1 - rho) * (grad ** 2)
        grad_accu_new = rho * grad_accu + (1 - rho) * grad

        updates[accu] = accu_new
        updates[grad_accu] = grad_accu_new

        denorm = theano.tensor.sqrt(accu_new - grad_accu_new ** 2 + epsilon)
        velocity_new = momentum * velocity - lr * grad / denorm
        updates[velocity] = velocity_new
        updates[param] = param + velocity_new

    return updates

def adadelta_updates(params, grads, lr, rho, epsilon):
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow = True)
        var = numpy.zeros_like(value)
        accu = theano.shared(var, broadcastable = param.broadcastable)
        delta_accu = theano.shared(var, broadcastable = param.broadcastable)

        accu_new = rho * accu + (1 - rho) * (grad ** 2)
        updates[accu] = accu_new

        update = (grad * theano.tensor.sqrt(delta_accu + epsilon) /
                  theano.tensor.sqrt(accu_new + epsilon))
        updates[param] = param - lr * update

        delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates

def adam_updates(params, grads, lr, beta1, beta2, epsilon):
    t_prev = theano.shared(numpy.asarray(0.0, dtype = theano.config.floatX))
    updates = OrderedDict()

    t = t_prev + 1
    a_t = lr * theano.tensor.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

    for param, g_t in zip(params, grads):
        value = param.get_value(borrow = True)
        var = numpy.zeros_like(value)
        m_prev = theano.shared(var, broadcastable = param.broadcastable)
        v_prev = theano.shared(var, broadcastable = param.broadcastable)

        m_t = beta1 * m_prev + (1 - beta1) * g_t
        v_t = beta2 * v_prev + (1 - beta2) * (g_t ** 2)
        step = a_t * m_t / (theano.tensor.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates

class optimizer:

    def __init__(self, model, **option):
        information = {}

        information['sgd'] = (1, [1.0])
        information['adagrad'] = (2, [1.0, 1e-6])
        information['rmsprop'] = (3, [1e-2, 0.99, 1e-8])
        # torch default: 1.0, 0.9, 1e-6
        information['adadelta'] = (3, [1.0, 0.95, 1e-6])
        information['adam'] = (4, [0.001, 0.9, 0.999, 1e-8])
        information['rmsprop_momentum'] = (4, [1e-4, 0.95, 0.9, 1e-4])

        params = model.parameter
        grads = model.gradient
        inputs = model.input
        outputs = model.output

        vec = [theano.shared(numpy.zeros_like(p.get_value())) for p in params]

        if 'algorithm' not in option:
            option['algorithm'] = 'sgd'

        if 'variant' not in option:
            option['variant'] = None

        if 'constraint' not in option:
            option['constraint'] = None

        if 'momentum' not in option:
            option['momentum'] = False

        if 'norm' not in option:
            option['norm'] = True

        if 'nesterov' not in option:
            option['nesterov'] = False

        algorithm = option['algorithm']
        variant = option['variant']
        variant = [variant] if variant != None else []

        if option['norm']:
            normval = grad_norm(grads)
            outputs = outputs[:]
            outputs.insert(1, normval)

        if option['constraint']:
            method, value = option['constraint']
            if method == 'value':
                grads = grad_clip(grads, value[0], value[1])
            if method == 'norm':
                grads = grad_renormalize(grads, value)

        if option['nesterov']:
            option['momentum'] = False

        gup = [(v, g) for v, g in zip(vec, grads)]

        if algorithm == 'sgd':
            alpha = theano.tensor.scalar()
            hparams = [alpha]
            defaults = [('alpha', 1.0)]
            pup = sgd_updates(params, vec, *hparams)
        elif algorithm == 'adagrad':
            alpha = theano.tensor.scalar()
            epsilon = theano.tensor.scalar()
            hparams = [alpha, epsilon]
            defaults = [('alpha', 1.0), ('epsilon', 1e-6)]
            pup = adagrad_updates(params, vec, *hparams)
        elif algorithm == 'rmsprop':
            alpha = theano.tensor.scalar()
            rho = theano.tensor.scalar()
            epsilon = theano.tensor.scalar()
            hparams = [alpha, rho, epsilon]
            defaults = [('alpha', 1e-2), ('rho', 0.99), ('epsilon', 1e-8)]
            pup = rmsprop_updates(params, vec, *(hparams + variant))
        elif algorithm == 'rmsprop_momentum':
            alpha = theano.tensor.scalar()
            rho = theano.tensor.scalar()
            epsilon = theano.tensor.scalar()
            momentum = theano.tensor.scalar()
            hparams = [alpha, rho, epsilon, momentum]
            defaults = [('alpha', 1e-4), ('rho', 0.95), ('epsilon', 1e-4)]
            defaults.append(('moment', 0.9))
            pup = rmsprop_momentum_updates(params, vec, *hparams)
        elif algorithm == 'adadelta':
            alpha = theano.tensor.scalar()
            rho = theano.tensor.scalar()
            epsilon = theano.tensor.scalar()
            hparams = [alpha, rho, epsilon]
            defaults = [('alpha', 1.0), ('rho', 0.95), ('epsilon', 1e-6)]
            pup = adadelta_updates(params, vec, *hparams)
        elif algorithm == 'adam':
            alpha = theano.tensor.scalar()
            beta1 = theano.tensor.scalar()
            beta2 = theano.tensor.scalar()
            epsilon = theano.tensor.scalar()
            hparams = [alpha, beta1, beta2, epsilon]
            defaults = [('alpha', 0.001), ('beta1', 0.9), ('beta2', 0.999)]
            defaults.append(('epsilon', 1e-8))
            pup = adam_updates(params, vec, *hparams)
        else:
            raise 'Error: ' + algorithm + ' is not supported'

        if option['momentum']:
            momentum = theano.tensor.scalar()
            hparams.append(momentum)
            defaults.append(('momentum', 0.9))
            pup = apply_momentum(pup, params, momentum)

        if option['nesterov']:
            momentum = theano.tensor.scalar()
            hparams.append(momentum)
            defaults.append(('momentum', 0.9))
            pup = apply_momentum(pup, params, momentum)

        optimize = theano.function(inputs, outputs, updates = gup)
        update = theano.function(hparams, [], updates = pup)

        def wrapper(**option):
            values = []
            for item in defaults:
                name = item[0]
                val = item[1]
                if name not in option:
                    option[name] = val
                values.append(option[name])
            return update(*values)

        self.optimize = optimize
        self.update = wrapper
        self.option = option
        self.algorithm = algorithm
        self.information = information
