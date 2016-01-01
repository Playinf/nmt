# nmtdebug.py
# numpy version nmt, used for debugging
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import itertools

def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))
    
def softmax(mat):
    e = numpy.exp(mat)
    z = numpy.sum(e, -1)
    if mat.ndim == 1:
        return e / z
    else:
        return e / z[:, None]

class nnunit:

    def __init__(self, isize, osize, func = sigmoid):
        w = numpy.random.uniform(-0.05, 0.05, (osize, isize)).astype('float32')
        b = numpy.zeros(osize).astype('float32')

        info = [[(isize,), None]]

        self.flag = 0
        self.module = []
        self.function = func
        self.parameter = [w, b]
        self.information = [(1, 1), info]

    def __call__(self, x):
        weight = self.parameter[0]
        bias = self.parameter[1]
        function = self.function
        return function(numpy.dot(x, weight.transpose()) + bias)

class gru:
    def __init__(self, isize, hsize, osize):
        gates = nnunit(isize + hsize, 2 * osize, lambda x: x)
        transform = nnunit(isize + hsize, 2 * osize, numpy.tanh)

        module = []
        module.append(gates)
        module.append(transform)

        params = [item.parameter for item in module]

        self.module = module
        self.parameter = list(itertools.chain(*params))

    def __call__(self, x, h):
        gates = self.module[0]
        transform = self.module[1]

        t = gates(numpy.concatenate([x, h], 1))
        n = t.shape[1] / 2

        r = sigmoid(t[:, :n])
        z = sigmoid(t[:, -n:])
        c = transform(numpy.concatenate([x, r * h], 1))
        h = z * h + (1 - z) * c

        return h

class attunit:

    def __init__(self, size):
        vec = numpy.random.uniform(-0.08, 0.08, (size, 1)).astype('float32')
        bias = numpy.zeros((1,)).astype('float32')

        params = [vec, bias]

        self.parameter = params

    def __call__(self, s, h, m):
        vec = self.parameter[0]
        bias = self.parameter[1]

        z = numpy.tanh(s[None, :, :] + h)
        alpha = numpy.dot(z, vec) + bias
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1]))
        alpha = numpy.exp(alpha) * m
        alpha = alpha / numpy.sum(alpha, 0)[None, :]

        return alpha

class nmtmodel:

    def __init__(self, **option):
        sedim, tedim = option['embdim']
        shdim, thdim, ahdim = option['hidden']
        dhdim = option['deephid']
        svocab, tvocab = option['vocabulary']
        sw2id, sid2w = svocab
        tw2id, tid2w = tvocab

        femb = numpy.random.uniform(-0.08, 0.08, (len(sid2w), sedim))
        femb = femb.astype('float32')

        bemb = numpy.random.uniform(-0.08, 0.08, (len(sid2w), sedim))
        bemb = bemb.astype('float32')

        temb = numpy.random.uniform(-0.08, 0.08, (len(tid2w), tedim))
        temb = temb.astype('float32')

        params = [femb, bemb, temb]
        module = []

        forward_encoder = gru(sedim, shdim, shdim)
        backward_encoder = gru(sedim, shdim, shdim)
        hidden_decoder = gru(tedim, thdim, thdim)
        top_decoder = gru(ahdim, thdim, thdim)
        context_map = nnunit(2 * shdim, ahdim, lambda x: x)
        hidden_map = nnunit(thdim, ahdim, lambda x: x)
        aligner = attunit(ahdim)
        deepout = nnunit(tedim + ahdim + shdim, dhdim, numpy.tanh)
        classifier = nnunit(dhdim, len(tid2w), softmax)

        module.append(forward_encoder)
        module.append(backward_encoder)
        module.append(hidden_decoder)
        module.append(top_decoder)
        module.append(context_map)
        module.append(hidden_map)
        module.append(aligner)
        module.append(deepout)
        module.append(classifier)

        params.extend(forward_encoder.parameter)
        params.extend(backward_encoder.parameter)
        params.extend(hidden_decoder.parameter)
        params.extend(top_decoder.parameter)
        params.extend(context_map.parameter)
        params.extend(hidden_map.parameter)
        params.extend(aligner.parameter)
        params.extend(deepout.parameter)
        params.extend(classifier.parameter)
        
        def build_training(x, y, xmask, ymask):
            # input embeddings
            fnetin = femb[x.flatten()]
            fnetin = fnetin.reshape((x.shape[0], x.shape[1], sedim))
            bnetin = bemb[x.flatten()]
            bnetin = bnetin.reshape((x.shape[0], x.shape[1], sedim))
            tnetin = temb[y.flatten()]
            tnetin = tnetin.reshape((y.shape[0], y.shape[1], tedim))

            # shifted target input embeddings
            tnetins = numpy.zeros_like(tnetin)
            tnetins[1:] = tnetin[:-1]

            # forward encoding step
            def fencstep(x, m, h):
                nh = forward_encoder(x, h)
                h = m[:, None] * nh + (1.0 - m[:, None]) * h
                return [h]

            # backward encoding step
            def bencstep(x, m, h):
                nh = backward_encoder(x, h)
                h = m[:, None] * nh + (1.0 - m[:, None]) * h
                return [h]

            # initial state
            ih = numpy.zeros((x.shape[1], shdim))

            fseq = [fnetin, xmask]
            hf, uf = theano.scan(fencstep, fseq, [ih])

            # reverse input
            bseq = [bnetin[::-1], xmask[::-1]]
            hb, ub = theano.scan(bencstep, bseq, [ih])
            hb = hb[::-1]

            context = theano.tensor.concatenate([hf, hb], 2)
            # initial state
            inis = theano.tensor.sum(context * xmask[:, :, None], 0)
            inis = inis / theano.tensor.sum(xmask, 0)[:, None]
            context = context_map(context)

            # decoding step
            def decstep(e, m, s, h, k):
                z = hidden_decoder(e, s)
                z = m[:, None] * z + (1.0 - m[:, None]) * s                
                t = hidden_map(z)
                alpha = aligner(t, h, k)
                c = theano.tensor.sum(alpha[:, :, None] * h, 0)
                s = top_decoder(c, z)
                s = m[:, None] * s + (1.0 - m[:, None]) * z

                return [s, s]
                
            seq = [tnetins, ymask]
            o, u = theano.scan(decstep, seq, [inis, None], [context, xmask])
            s, c = o
            logit = deepout(theano.tensor.concatenate([tnetins, s, c], 2))
            logit = logit.reshape((logit.shape[0] * logit.shape[1], -1))
            probs = classifier(logit)

            idx = theano.tensor.arange(y.flatten().shape[0])
            cost = -theano.tensor.log(probs[idx, y.flatten()])
            cost = cost.reshape((y.shape[0], y.shape[1]))
            cost = theano.tensor.sum(cost * ymask, 0)
            cost = theano.tensor.mean(cost)

            return [x, xmask, y, ymask], [cost]
            
        def build_sampling():
            x = theano.tensor.imatrix()
            xmask = theano.tensor.matrix()

            # input embeddings
            fnetin = femb[x.flatten()]
            fnetin = fnetin.reshape((x.shape[0], x.shape[1], sedim))
            bnetin = bemb[x.flatten()]
            bnetin = bnetin.reshape((x.shape[0], x.shape[1], sedim))

            # forward encoding step
            def fencstep(x, m, h):
                nh = forward_encoder(x, h)
                h = m[:, None] * nh + (1.0 - m[:, None]) * h
                return [h]

            # backward encoding step
            def bencstep(x, m, h):
                nh = backward_encoder(x, h)
                h = m[:, None] * nh + (1.0 - m[:, None]) * h
                return [h]

            # initial state
            ih = theano.tensor.zeros((x.shape[1], shdim))

            fseq = [fnetin, xmask]
            hf, uf = theano.scan(fencstep, fseq, [ih])

            # reverse input
            bseq = [bnetin[::-1], xmask[::-1]]
            hb, ub = theano.scan(bencstep, bseq, [ih])
            hb = hb[::-1]

            context = theano.tensor.concatenate([hf, hb], 2)
            # initial state
            inis = theano.tensor.sum(context * xmask[:, :, None], 0)
            inis = inis / theano.tensor.sum(xmask, 0)[:, None]
            context = context_map(context)

            # decoding step
            def decstep(e, s, h, k):
                z = hidden_decoder(e, s)
                t = hidden_map(z)
                alpha = aligner(t, h, k)
                c = theano.tensor.sum(alpha[:, :, None] * h, 0)
                s = top_decoder(c, z)
                d = deepout(theano.tensor.concatenate([e, s, c], 1))
                p = classifier(d)

                return [p, s]

            def build_encoder():
                encfunc = theano.function([x, xmask], [context, inis])
                return encfunc

            def build_init_sample():
                # context
                c = theano.tensor.tensor3()
                # state
                s = theano.tensor.matrix()
                # init embedding
                iniy = theano.tensor.zeros((xmask.shape[1], tedim))

                prob, state = decstep(iniy, s, c, xmask)
                sample = theano.function([s, c, xmask], [prob, state])
                return sample

            def build_sample():
                y = theano.tensor.ivector()
                s = theano.tensor.matrix()
                c = theano.tensor.tensor3()
                tnetin = temb[y]
                tnetin = tnetin.reshape((y.shape[0], tedim))
                prob, state = decstep(tnetin, s, c, xmask)
                sample = theano.function([y, s, c, xmask], [prob, state])
                return sample

            getstate = build_encoder()
            isample = build_init_sample()
            sample = build_sample()

            return [getstate, isample, sample]

        inputs, outputs = build_training()
        gradient = theano.grad(outputs[0], params)

        sample = build_sampling()

        self.input = inputs
        self.output = outputs
        self.option = option
        self.sample = sample
        self.gradient = gradient
        self.parameter = params
        self.vocabulary = option['vocabulary']

