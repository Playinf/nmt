# rnnsearch.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano
import itertools

from nn import linear, gru, maxout, embedding, embedder
from beamsearch import beam, hypothesis

# rnnsearch decoder
class decoder:

    def __init__(self, model, **option):
        self.model = model
        self.sample = model.sample
        self.option = option
        self.size = option['size']
        self.threshold = option['threshold']
        self.vocabulary = model.vocabulary

    # based on groundhog's impelmentation
    def decode(self, seq, minlen = None, maxlen = None, size = None):
        beamsize = self.size
        threshold = self.threshold
        size = beamsize if size == None else size
        encode, compute_prob, compute_state = self.sample
        vocab = self.vocabulary[1][1]
        eosid = self.vocabulary[1][0]['<eos>']

        if maxlen == None:
            maxlen = len(seq) * 3

        if minlen == None:
            minlen = len(seq) / 2

        anno, manno, state = encode(seq)

        dim = state.shape[1]
        states = state

        trans = [[]]
        costs = [0.0]
        final_trans = []
        final_costs = []

        for k in range(maxlen):
            if size == 0:
                break

            num = len(trans)

            if k > 0:
                last_words = numpy.array(map(lambda t: t[-1], trans))
                last_words = last_words.astype('int32')
            else:
                last_words = numpy.zeros(num, 'int32')

            annos = numpy.repeat(anno, num, 1)
            mannos = numpy.repeat(manno, num, 1)
            probs = compute_prob(last_words, annos, mannos, states)
            logprobs = numpy.log(probs)

            if k < minlen:
                logprobs[:, eosid] = -numpy.inf

            ncosts = numpy.array(costs)[:, None] - logprobs
            fcosts = ncosts.flatten()
            nbest = numpy.argpartition(fcosts, size)[:size]

            vocsize = logprobs.shape[1]
            tinds = nbest / vocsize
            winds = nbest % vocsize
            costs = fcosts[nbest]

            newtrans = [[]] * size
            newcosts = numpy.zeros(size)
            newstates = numpy.zeros((size, dim), 'float32')
            inputs = numpy.zeros(size, 'int32')

            for i, (idx, nword, ncost) in enumerate(zip(tinds, winds, costs)):
                newtrans[i] = trans[idx] + [nword]
                newcosts[i] = ncost
                newstates[i] = states[idx]
                inputs[i] = nword

            annos = numpy.repeat(anno, size, 1)
            mannos = numpy.repeat(manno, size, 1)
            newstates = compute_state(inputs, annos, mannos, newstates)

            trans = []
            costs = []
            indices = []

            for i in range(size):
                if newtrans[i][-1] != eosid:
                    trans.append(newtrans[i])
                    costs.append(newcosts[i])
                    indices.append(i)
                else:
                    size -= 1
                    final_trans.append(newtrans[i])
                    final_costs.append(newcosts[i])
            states = newstates[indices]

        if not len(final_trans):
            final_trans = [[]]
            final_costs = [0.0]

        final_beam = beam(beamsize, threshold)

        for cost, trans in zip(final_costs, final_trans):
            hypo = hypothesis()
            count = len(trans)
            # normalize
            if count > 0:
                hypo.score = -cost / count
            hypo.translation = map(lambda x: vocab[x], trans)
            final_beam.insert(hypo)

        return final_beam.sort()

    def enhanced_decode(self, data):
        option = self.option
        vocab = self.vocabulary[1][1]
        size = self.size
        threshold = self.threshold
        encode, compute_prob, compute_state = self.sample
        beamlist = []

        if 'maxlen' not in option or option['maxlen'] == None:
            option['maxlen'] = data.shape[0] * 2

        if 'minlen' not in option or option['minlen'] == None:
            option['minlen'] = data.shape[0] / 2

        maxlen = option['maxlen']
        minlen = option['minlen']

        init_hypo = hypothesis()
        init_beam = beam(size, threshold)
        final_beam = beam(size, threshold)

        # encoding source sentence
        annot, tannot, state = encode(data)

        init_hypo.score = 0.0
        init_hypo.state = [numpy.array([-1]).astype('int32'), state]
        init_beam.insert(init_hypo)
        init_beam.sort()

        dim = state.shape[1]

        def batch_next(prevbeam):
            hypolist = prevbeam.ordereditems
            n = len(hypolist)
            indexs = numpy.zeros((n,), 'int32')
            states = numpy.zeros((n, dim), 'float32')
            annots = numpy.repeat(annot, n, 1)
            tannots = numpy.repeat(tannot, n, 1)
            scores = numpy.zeros((n,), 'float32')

            # previous hypothesis
            for i in range(n):
                hypo = hypolist[i]
                index, state = hypo.state
                scores[i] = hypo.score
                indexs[i] = index
                states[i] = state

            probs = compute_prob(indexs, annots, tannots, states)
            next_beam = beam(size, threshold)

            logprobs = numpy.log(probs)

            #if k < minlen:
            #    logprobs[:, eosid] = -numpy.inf

            nscores = scores[:, None] - logprobs
            fscores = nscores.flatten()
            nbest = numpy.argpartition(fscores, -size)[:-size]

            vocsize = logprobs.shape[1]
            tinds = nbest / vocsize
            winds = nbest % vocsize
            scores = fscores[nbest]

            newstates = numpy.zeros((size, dim), 'float32')
            inputs = numpy.zeros(size, 'int32')
            hlist = []

            for i, (idx, nword, nsc) in enumerate(zip(tinds, winds, scores)):
                newstates[i] = states[idx]
                inputs[i] = nword
                hypo = hypothesis()
                hypo.score = nsc
                hypo.translation = hypolist[i].translation[:]
                hypo.translation.append(vocab[nword])
                hypo.state = [idx, None]
                hlist.append(hypo)

            annots = numpy.repeat(annot, size, 1)
            tannots = numpy.repeat(tannot, size, 1)
            newstates = compute_state(inputs, annots, tannots, newstates)

            for hypo, state in zip(hlist, newstates):
                hypo.state[1] = state

                if hypo.translation[-1] == '<eos>':
                    if len(hypo.translation) > minlen:
                        final_beam.insert(hypo)
                else:
                    next_beam.insert(hypo)

            next_beam.sort()

        while len(beamlist) <= maxlen:
            prev_beam = beamlist[-1]
            next_beam = batch_next(prev_beam)

            if next_beam.size == 0:
                break

            beamlist.append(next_beam)

        return final_beam.sort()

class rnnsearch:

    def __init__(self, **option):
        sedim, tedim = option['embdim']
        shdim, thdim, ahdim = option['hidden']
        maxdim = option['maxhid']
        deephid = option['deephid']
        k = option['maxpart']
        svocab, tvocab = option['vocabulary']
        sw2id, sid2w = svocab
        tw2id, tid2w = tvocab
        svsize = len(sid2w)
        tvsize = len(tid2w)

        mhdim = maxdim * k

        module = []

        # encoder
        source_embedding = embedding(svsize, sedim)
        source_embedder = embedder(sedim)
        forward_encoder = gru(sedim, shdim, shdim)
        backward_encoder = gru(sedim, shdim, shdim)

        # decoder
        target_embedding = embedding(tvsize, tedim)
        target_embedder = embedder(tedim)
        init_state_transform = linear(shdim, thdim)
        annotation_transform = linear(2 * shdim, ahdim, bias = False)
        state_transform = linear(thdim, ahdim, bias = False)
        context_transform = linear(ahdim, 1, bias = False)
        decoder = gru(tedim, 2 * shdim, thdim, thdim)
        maxout_transform = maxout(thdim, tedim, 2 * shdim, mhdim, maxpart = k)
        deepout_transform = linear(maxdim, deephid, bias = False)
        classify_transform = linear(deephid, tvsize)

        module.append(source_embedding)
        module.append(source_embedder)
        module.append(forward_encoder)
        module.append(backward_encoder)
        module.append(target_embedding)
        module.append(target_embedder)
        module.append(init_state_transform)
        module.append(annotation_transform)
        module.append(state_transform)
        module.append(context_transform)
        module.append(decoder)
        module.append(maxout_transform)
        module.append(deepout_transform)
        module.append(classify_transform)

        params = list(itertools.chain(*[m.parameter for m in module]))

        def build_training():
            sseq = theano.tensor.imatrix()
            smask = theano.tensor.matrix()
            tseq = theano.tensor.imatrix()
            tmask = theano.tensor.matrix()

            x = source_embedder(source_embedding(), sseq)
            h = theano.tensor.zeros((sseq.shape[1], shdim))

            def forward_step(x, m, h):
                nh = forward_encoder(x, h)
                nh = (1.0 - m[:, None]) * h + m[:, None] * nh
                return [nh]

            def backward_step(x, m, h):
                nh = backward_encoder(x, h)
                nh = (1.0 - m[:, None]) * h + m[:, None] * nh
                return [nh]

            seq = [x, smask]
            hf, u = theano.scan(forward_step, seq, [h])

            seq = [x[::-1], smask[::-1]]
            hb, u = theano.scan(backward_step, seq, [h])
            hb = hb[::-1]

            annotation = theano.tensor.concatenate([hf, hb], 2)
            state = theano.tensor.tanh(init_state_transform(hb[0]))
            mannotation = annotation_transform(annotation)

            y = target_embedder(target_embedding(), tseq)
            yshift = theano.tensor.zeros_like(y)
            yshift = theano.tensor.set_subtensor(yshift[1:], y[:-1])

            def decode_step(y, e, m, s, a, t, k):
                mstate = state_transform(s)
                hidden = theano.tensor.tanh(t + mstate[None, :, :])
                hidden = theano.tensor.exp(context_transform(hidden))
                hidden = hidden.reshape((hidden.shape[0], hidden.shape[1]))
                hidden = hidden * k
                alpha = hidden / theano.tensor.sum(hidden, 0)
                context = theano.tensor.sum(alpha[:, :, None] * a, 0)
                readout = maxout_transform(s, e, context)
                deepout = deepout_transform(readout)
                prob = theano.tensor.nnet.softmax(classify_transform(deepout))
                state = decoder(y, context, s)
                state = (1.0 - m[:, None]) * s + m[:, None] * state

                return [prob, state]

            seq = [y, yshift, tmask]
            info = [None, state]
            nonseq = [annotation, mannotation, smask]
            [probs, states], u = theano.scan(decode_step, seq, info, nonseq)

            probs = probs.reshape((probs.shape[0] * probs.shape[1], -1))
            idx = theano.tensor.arange(tseq.flatten().shape[0])
            cost = -theano.tensor.log(probs[idx, tseq.flatten()])
            cost = cost.reshape((tseq.shape[0], tseq.shape[1]))
            cost = theano.tensor.sum(cost * tmask)

            return [sseq, smask, tseq, tmask], [cost]

        def build_sampling():
            def encode():
                seq = theano.tensor.imatrix()
                x = source_embedder(source_embedding(), seq)
                h = theano.tensor.zeros((seq.shape[1], shdim))

                def forward_step(x, h):
                    h = forward_encoder(x, h)
                    return [h]

                def backward_step(x, h):
                    h = backward_encoder(x, h)
                    return [h]

                hf, u = theano.scan(forward_step, [x], [h])
                hb, u = theano.scan(backward_step, [x[::-1]], [h])
                hb = hb[::-1]

                annot = theano.tensor.concatenate([hf, hb], 2)
                state = theano.tensor.tanh(init_state_transform(hb[0]))
                mannot = annotation_transform(annot)

                return theano.function([seq], [annot, mannot, state])

            def compute_prob():
                a = theano.tensor.tensor3()
                m = theano.tensor.tensor3()
                y = theano.tensor.ivector()
                s = theano.tensor.matrix()

                cond = theano.tensor.neq(y, 0)

                emb = target_embedder(target_embedding(), y)
                emb = emb * cond[:, None]

                mstate = state_transform(s)
                hidden = theano.tensor.tanh(m + mstate[None, :, :])
                hidden = theano.tensor.exp(context_transform(hidden))
                hidden = hidden.reshape((hidden.shape[0], hidden.shape[1]))
                alpha = hidden / theano.tensor.sum(hidden, 0)
                context = theano.tensor.sum(alpha[:, :, None] * a, 0)
                readout = maxout_transform(s, emb, context)
                deepout = deepout_transform(readout)
                prob = theano.tensor.nnet.softmax(classify_transform(deepout))

                return theano.function([y, a, m, s], prob)

            def compute_state():
                a = theano.tensor.tensor3()
                m = theano.tensor.tensor3()
                y = theano.tensor.ivector()
                s = theano.tensor.matrix()

                emb = target_embedder(target_embedding(), y)

                mstate = state_transform(s)
                hidden = theano.tensor.tanh(m + mstate[None, :, :])
                hidden = theano.tensor.exp(context_transform(hidden))
                hidden = hidden.reshape((hidden.shape[0], hidden.shape[1]))
                alpha = hidden / theano.tensor.sum(hidden, 0)
                context = theano.tensor.sum(alpha[:, :, None] * a, 0)
                state = decoder(emb, context, s)

                return theano.function([y, a, m, s], state)

            return encode(), compute_prob(), compute_state()

        inputs, outputs = build_training()
        gradient = theano.grad(outputs[0], params)

        self.input = inputs
        self.output = outputs
        self.option = option
        self.gradient = gradient
        self.sample = build_sampling()
        self.parameter = params
        self.vocabulary = option['vocabulary']
