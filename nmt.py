# basenmt.py
# a baseline model of nerual machine translation, based on dl4mt
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import math
import numpy
import theano

from nn import nnunit, gru
from beamsearch import beam, hypothesis

# attention unit
class attunit:

    def __init__(self, size):
        vec = numpy.random.uniform(-0.08, 0.08, (size, 1))
        vec = theano.shared(vec.astype(theano.config.floatX))
        bias = numpy.zeros((1,))
        bias = theano.shared(bias.astype(theano.config.floatX))

        params = [vec, bias]

        self.parameter = params

    def __call__(self, s, h, m):
        # assumes:
        # s => (batch, ahiddim)
        # h => (seq, batch, ahiddim)
        # m => (seq, batch)
        # vec => (ahiddim, 1)
        # bias => (1,)
        vec = self.parameter[0]
        bias = self.parameter[1]

        z = theano.tensor.tanh(s[None, :, :] + h)
        alpha = theano.dot(z, vec) + bias
        # shape: (seq, batch)
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1]))
        alpha = theano.tensor.exp(alpha) * m
        alpha = alpha / theano.tensor.sum(alpha, 0)[None, :]

        return alpha

# neural machine translation model decoder
class decoder:

    def __init__(self, model, size, threshold):
        self.model = model
        self.sample = model.sample
        self.option = [size, threshold]
        self.vocabulary = model.vocabulary

    def decode(self, xdata, xmask):
        vocab = self.vocabulary[1][1]
        size, threshold = self.option
        encode, sample = self.sample
        beamlist = []
        
        minlen = xdata.shape[0] / 2
        init_hypo = hypothesis()
        init_beam = beam(size, threshold)
        final_beam = beam(size, threshold)
        
        # encoding source sentence
        hidden, context, state = encode(xdata, xmask)
        init_hypo.state = [numpy.array([-1]).astype('int32'), state]
        init_beam.insert(init_hypo)
        state_size = state.shape[1]
        
        def batch_next(prevbeam):
            hypolist = prevbeam.sort()
            n = len(hypolist)            
            indexs = numpy.zeros((n,), 'int32')
            states = numpy.zeros((n, state_size), 'float32')
            hids = numpy.repeat(hidden, n, 1)
            ctxs = numpy.repeat(context, n, 1)
            xmasks = numpy.repeat(xmask, n, 1)
            
            for i in range(n):
                hypo = hypolist[i]
                index, state = hypo.state
                indexs[i] = index
                states[i] = state
                
            probs, next_states = sample(indexs, states, hids, ctxs, xmasks)
            next_beam = beam(size, threshold)
            
            # explore
            for i in range(n):
                hypo = hypolist[i]
                score = hypo.score
                trans = hypo.translation
                prob = probs[i]
                next_state = next_states[i]
                nbest = numpy.argpartition(prob, -size)[-size:]
                
                for j in range(size):
                    ind = nbest[j]
                    p = prob[ind]
                    if p == 0.0:
                        continue
                    newhypo = hypothesis()
                    newhypo.score = score + math.log(p)
                    newhypo.state = [ind, next_state]
                    newhypo.translation = trans[:]
                    newhypo.translation.append(vocab[ind])
                    
                    if newhypo.translation[-1] == '</s>':
                        if len(newhypo.translation) > minlen:
                            final_beam.insert(newhypo)
                    else:
                        next_beam.insert(newhypo)
                    
            return next_beam
            
        beamlist.append(init_beam)
        
        while final_beam.size < size and len(beamlist) < 200:
            prev_beam = beamlist[-1]
            next_beam = batch_next(prev_beam)
            
            if next_beam.size == 0:
                break
            
            beamlist.append(next_beam)
            
        return final_beam.sort()

class nmtmodel:

    def __init__(self, **option):
        sedim, tedim = option['embdim']
        shdim, thdim, ahdim = option['hidden']
        dhdim = option['deephid']
        svocab, tvocab = option['vocabulary']
        sw2id, sid2w = svocab
        tw2id, tid2w = tvocab

        femb = numpy.random.uniform(-0.08, 0.08, (len(sid2w), sedim))
        femb = theano.shared(femb.astype(theano.config.floatX))

        bemb = numpy.random.uniform(-0.08, 0.08, (len(sid2w), sedim))
        bemb = theano.shared(bemb.astype(theano.config.floatX))

        temb = numpy.random.uniform(-0.08, 0.08, (len(tid2w), tedim))
        temb = theano.shared(temb.astype(theano.config.floatX))

        params = [femb, bemb, temb]
        module = []

        # compute source hidden state h
        forward_encoder = gru(sedim, shdim, shdim)
        backward_encoder = gru(sedim, shdim, shdim)
        # compute target hidden state s
        hidden_decoder = gru(tedim, thdim, thdim)
        top_decoder = gru(ahdim, thdim, thdim)
        # mapping context to initial target state
        state_map = nnunit(2 * shdim, thdim, theano.tensor.tanh)
        # linear mapping of context and target hidden state
        # used for accelerate training
        context_map = nnunit(2 * shdim, ahdim, lambda x: x)
        hidden_map = nnunit(thdim, ahdim, lambda x: x)
        # compute attention probability
        aligner = attunit(ahdim)
        # deep output
        deepout = nnunit(tedim + ahdim + shdim, dhdim, theano.tensor.tanh)
        # classification
        classifier = nnunit(dhdim, len(tid2w), theano.tensor.nnet.softmax)

        module.append(forward_encoder)
        module.append(backward_encoder)
        module.append(hidden_decoder)
        module.append(top_decoder)
        module.append(state_map)
        module.append(context_map)
        module.append(hidden_map)
        module.append(aligner)
        module.append(deepout)
        module.append(classifier)

        params.extend(forward_encoder.parameter)
        params.extend(backward_encoder.parameter)
        params.extend(hidden_decoder.parameter)
        params.extend(top_decoder.parameter)
        params.extend(state_map.parameter)
        params.extend(context_map.parameter)
        params.extend(hidden_map.parameter)
        params.extend(aligner.parameter)
        params.extend(deepout.parameter)
        params.extend(classifier.parameter)
        
        def build_training():
            x = theano.tensor.imatrix()
            y = theano.tensor.imatrix()
            xmask = theano.tensor.matrix()
            ymask = theano.tensor.matrix()

            # input embeddings
            fnetin = femb[x.flatten()]
            fnetin = fnetin.reshape((x.shape[0], x.shape[1], sedim))
            bnetin = bemb[x.flatten()]
            bnetin = bnetin.reshape((x.shape[0], x.shape[1], sedim))
            tnetin = temb[y.flatten()]
            tnetin = tnetin.reshape((y.shape[0], y.shape[1], tedim))

            # shifted target input embeddings
            tnetins = theano.tensor.zeros_like(tnetin)
            tnetins = theano.tensor.set_subtensor(tnetins[1:], tnetin[:-1])

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

            shidden = theano.tensor.concatenate([hf, hb], 2)
            # initial state
            inis = theano.tensor.sum(shidden * xmask[:, :, None], 0)
            inis = inis / theano.tensor.sum(xmask, 0)[:, None]
            inis = state_map(inis)
            context = context_map(shidden)

            # decoding step
            def decstep(e, m, s, c, h, k):
                z = hidden_decoder(e, s)
                z = m[:, None] * z + (1.0 - m[:, None]) * s                
                t = hidden_map(z)
                alpha = aligner(t, c, k)
                c = theano.tensor.sum(alpha[:, :, None] * h, 0)
                s = top_decoder(c, z)
                s = m[:, None] * s + (1.0 - m[:, None]) * z

                return [s, c]
                
            seq = [tnetins, ymask]
            nonseq = [shidden, context, xmask]
            o, u = theano.scan(decstep, seq, [inis, None], nonseq)
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

            shidden = theano.tensor.concatenate([hf, hb], 2)
            # initial state
            inis = theano.tensor.sum(shidden * xmask[:, :, None], 0)
            inis = inis / theano.tensor.sum(xmask, 0)[:, None]
            inis = state_map(inis)
            context = context_map(shidden)

            # decoding step
            def decstep(e, s, h, c, k):
                z = hidden_decoder(e, s)
                t = hidden_map(z)
                alpha = aligner(t, c, k)
                c = theano.tensor.sum(alpha[:, :, None] * h, 0)
                s = top_decoder(c, z)
                d = deepout(theano.tensor.concatenate([e, s, c], 1))
                p = classifier(d)

                return [p, s]

            def build_encoder():
                encfunc = theano.function([x, xmask], [shidden, context, inis])
                return encfunc

            def build_sample():
                y = theano.tensor.ivector()
                s = theano.tensor.matrix()
                h = theano.tensor.tensor3()
                c = theano.tensor.tensor3()
                tnetin = temb[y]
                cond = theano.tensor.ge(y, 0)
                tnetin = tnetin * cond[:, None]
                prob, state = decstep(tnetin, s, h, c, xmask)
                sample = theano.function([y, s, h, c, xmask], [prob, state])
                return sample

            encode = build_encoder()
            sample = build_sample()

            return [encode, sample]

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
