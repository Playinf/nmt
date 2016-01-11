# sampler.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy

# rnnsearch sampler
class sampler:

    def __init__(self, model, **option):
        self.model = model
        self.sample = model.sample
        self.option = option
        self.size = option['size']
        self.vocabulary = model.vocabulary

    # based on groundhog's impelmentation
    def decode(self, seq, size = None):
        option = self.option
        beamsize = self.size
        size = beamsize if size == None else size
        encode, compute_istate, compute_prob, compute_state = self.sample
        vocab = self.vocabulary[1][1]
        eosid = self.vocabulary[1][0]['<eos>']

        if 'maxlen' not in option:
            option['maxlen'] = None

        if 'minlen' not in option:
            option['minlen'] = None

        maxlen = option['maxlen']
        minlen = option['minlen']

        if maxlen == None:
            maxlen = len(seq) * 3

        if minlen == None:
            minlen = len(seq) / 2

        anno = encode(seq)
        state = compute_istate(anno)

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
            probs = compute_prob(last_words, annos, states)
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
            newstates = compute_state(inputs, annos, newstates)

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

        for i, (cost, trans) in enumerate(zip(final_costs, final_trans)):
            count = len(trans)
            if count > 0:
                final_costs[i] = cost / count

        final_trans = numpy.array(final_trans)[numpy.argsort(final_costs)]
        final_costs = numpy.array(sorted(final_costs))

        translations = []

        for cost, trans in zip(final_costs, final_trans):
            trans = map(lambda x: vocab[x], trans)
            translations.append((trans, cost))

        return translations
