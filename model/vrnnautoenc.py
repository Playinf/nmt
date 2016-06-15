# vrnnautoenc.py
# variational recurrent autoencoder
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano
import theano.sandbox.rng_mrg

from utils import extract_option, update_option
from utils import add_if_not_exsit, add_parameters
from nn import linear, embedding, feedforward, gru, maxout

# standard rnnsearch configuration
def vrnnautoenc_config():
    opt = {}

    # embedding
    opt['source-embedding/bias'] = True
    opt['target-embedding/bias'] = True

    # encoder
    opt['encoder/forward-rnn/variant'] = 'standard'
    opt['encoder/backward-rnn/variant'] = 'standard'
    opt['encoder/forward-rnn/reset-gate/weight'] = True
    opt['encoder/forward-rnn/reset-gate/bias'] = False
    opt['encoder/forward-rnn/update-gate/weight'] = True
    opt['encoder/forward-rnn/update-gate/bias'] = False
    opt['encoder/forward-rnn/transform/weight'] = True
    opt['encoder/forward-rnn/transform/bias'] = True
    opt['encoder/backward-rnn/reset-gate/weight'] = True
    opt['encoder/backward-rnn/reset-gate/bias'] = False
    opt['encoder/backward-rnn/update-gate/weight'] = True
    opt['encoder/backward-rnn/update-gate/bias'] = False
    opt['encoder/backward-rnn/transform/weight'] = True
    opt['encoder/backward-rnn/transform/bias'] = True
    opt['encoder/latent-transform/weight'] = True
    opt['encoder/latent-transform/bias'] = True
    opt['encoder/latent-mu-transform/weight'] = True
    opt['encoder/latent-mu-transform/bias'] = True
    opt['encoder/latent-sigma-transform/weight'] = True
    opt['encoder/latent-sigma-transform/bias'] = True

    # decoder
    opt['decoder/init-transform/variant'] = 'standard'
    opt['decoder/annotation-transform/variant'] = 'standard'
    opt['decoder/state-transform/variant'] = 'standard'
    opt['decoder/context-transform/variant'] = 'standard'
    opt['decoder/rnn/variant'] = 'standard'
    opt['decoder/maxout/variant'] = 'standard'
    opt['decoder/deepout/variant'] = 'standard'
    opt['decoder/classify/variant'] = 'standard'

    opt['decoder/init-transform/weight'] = True
    opt['decoder/init-transform/bias'] = True
    opt['decoder/context-transform/weight'] = True
    opt['decoder/context-transform/bias'] = False
    opt['decoder/rnn/reset-gate/weight'] = True
    opt['decoder/rnn/reset-gate/bias'] = False
    opt['decoder/rnn/update-gate/weight'] = True
    opt['decoder/rnn/update-gate/bias'] = False
    opt['decoder/rnn/transform/weight'] = True
    opt['decoder/rnn/transform/bias'] = True
    opt['decoder/maxout/weight'] = True
    opt['decoder/maxout/bias'] = True
    opt['decoder/deepout/weight'] = True
    opt['decoder/deepout/bias'] = False
    opt['decoder/classify/weight'] = True
    opt['decoder/classify/bias'] = True

    return opt

class encoder:

    def __init__(self, input_size, hidden_size, latent_size, **option):
        opt = option

        fopt = extract_option(opt, 'forward-rnn')
        bopt = extract_option(opt, 'backward-rnn')
        lopt = extract_option(opt, 'latent-transform')
        mopt = extract_option(opt, 'latent-mu-transform')
        sopt = extract_option(opt, 'latent-sigma-transform')
        fopt['name'] = 'forward-rnn'
        bopt['name'] = 'backward-rnn'
        lopt['name'] = 'latent-transform'
        mopt['name'] = 'latent-mu-transform'
        sopt['name'] = 'latent-sigma-transform'
        lopt['function'] = theano.tensor.tanh

        forward_encoder = gru(input_size, hidden_size, **fopt)
        backward_encoder = gru(input_size, hidden_size, **bopt)

        latent_transform = feedforward(2 * hidden_size, latent_size, **lopt)
        mu_transform = linear(latent_size, latent_size, **mopt)
        sigma_transform = linear(latent_size, latent_size, **sopt)

        params = []
        add_parameters(params, 'encoder', *forward_encoder.parameter)
        add_parameters(params, 'encoder', *backward_encoder.parameter)
        add_parameters(params, 'encoder', *latent_transform.parameter)
        add_parameters(params, 'encoder', *mu_transform.parameter)
        add_parameters(params, 'encoder', *sigma_transform.parameter)

        def forward(x, mask, epsilon, initstate):
            def forward_step(x, m, h):
                nh = forward_encoder(x, h)
                nh = (1.0 - m[:, None]) * h + m[:, None] * nh
                return [nh]

            def backward_step(x, m, h):
                nh = backward_encoder(x, h)
                nh = (1.0 - m[:, None]) * h + m[:, None] * nh
                return [nh]

            seq = [x, mask]
            hf, u = theano.scan(forward_step, seq, [initstate])

            seq = [x[::-1], mask[::-1]]
            hb, u = theano.scan(backward_step, seq, [initstate])
            hb = hb[::-1]

            #annotation = theano.tensor.concatenate([hf, hb], 2)
            #hidden = theano.tensor.mean(annotation, 0)
            hidden = theano.tensor.concatenate([hf[-1], hb[0]], 1)

            latent = latent_transform(hidden)
            mu = mu_transform(latent)
            sigma = theano.tensor.exp(sigma_transform(latent))
            sigma = theano.tensor.sqrt(sigma)

            return mu + epsilon * sigma, mu, sigma

        self.name = 'encoder'
        self.option = option
        self.forward = forward
        self.parameter = params

    def __call__(self, x, mask, epsilon, initstate):
        return self.forward(x, mask, epsilon, initstate)

class decoder:

    def __init__(self, emb_size, latent_size, thidden_size, chidden_size,
                 mhidden_size, maxpart, dhidden_size, voc_size, **option):
        opt = option

        iopt = extract_option(opt, 'init-transform')
        topt = extract_option(opt, 'context-transform')
        ropt = extract_option(opt, 'rnn')
        mopt = extract_option(opt, 'maxout')
        dopt = extract_option(opt, 'deepout')
        copt = extract_option(opt, 'classify')

        topt['name'] = 'context-transform'
        ropt['name'] = 'rnn'
        mopt['name'] = 'maxout'
        dopt['name'] = 'deepout'
        copt['name'] = 'classify'
        iopt['function'] = theano.tensor.tanh
        topt['function'] = theano.tensor.tanh
        mopt['maxpart'] = maxpart
        mopt['maxpart'] = maxpart

        init_transform = feedforward(latent_size, thidden_size, **iopt)

        context_transform = linear(latent_size, chidden_size, **topt)
        # decoder rnn
        rnn = gru([emb_size, chidden_size], thidden_size, **ropt)
        maxout_transform = maxout([thidden_size, emb_size, chidden_size],
                                  mhidden_size, **mopt)
        deepout_transform = linear(mhidden_size, dhidden_size, **dopt)
        classify_transform = linear(dhidden_size, voc_size, **copt)

        params = []
        add_parameters(params, 'decoder', *init_transform.parameter)
        add_parameters(params, 'decoder', *context_transform.parameter)
        add_parameters(params, 'decoder', *rnn.parameter)
        add_parameters(params, 'decoder', *maxout_transform.parameter)
        add_parameters(params, 'decoder', *deepout_transform.parameter)
        add_parameters(params, 'decoder', *classify_transform.parameter)

        def compute_initstate(latent):
            state = init_transform(latent)
            context = context_transform(latent)
            return state, context

        def compute_probability(yemb, state, context):
            maxhid = maxout_transform([state, yemb, context])
            readout = deepout_transform(maxhid)
            preact = classify_transform(readout)
            prob = theano.tensor.nnet.softmax(preact)

            return prob

        def compute_state(yemb, ymask, state, context):
            new_state = rnn([yemb, context], state)
            ymask = ymask[:, None]
            new_state = (1.0 - ymask) * state + ymask * new_state

            return new_state

        def forward(yseq, ymask, latent):
            yshift = theano.tensor.zeros_like(yseq)
            yshift = theano.tensor.set_subtensor(yshift[1:], yseq[:-1])

            initstate, context = compute_initstate(latent)

            def step(yemb, ymask, state, context):
                new_state = compute_state(yemb, ymask, state, context)
                return [new_state]

            seq = [yseq, ymask]
            oinfo = [initstate]
            nonseq = [context]
            states, updates = theano.scan(step, seq, oinfo, nonseq)

            inis = initstate[None, :, :]
            all_states = theano.tensor.concatenate([inis, states], 0)
            prev_states = all_states[:-1]
            n = yseq.shape[0]
            contexts = theano.tensor.repeat(context[None, :, :], n, 0)

            maxhid = maxout_transform([prev_states, yshift, contexts])
            readout = deepout_transform(maxhid)
            preact = classify_transform(readout)
            preact = preact.reshape((preact.shape[0] * preact.shape[1], -1))
            prob = theano.tensor.nnet.softmax(preact)

            return prob

        self.name = 'decoder'
        self.option = opt
        self.forward = forward
        self.parameter = params
        self.compute_initstate = compute_initstate
        self.compute_probability = compute_probability
        self.compute_state = compute_state

    def __call__(self, yseq, ymask, latent):
        return self.forward(yseq, ymask, latent)

class vrnnautoenc:

    def __init__(self, **option):
        opt = vrnnautoenc_config()

        update_option(opt, option)
        sedim, tedim = option['embdim']
        shdim, thdim, chdim = option['hidden']
        lhdim = option['latent']
        maxdim = option['maxhid']
        deephid = option['deephid']
        k = option['maxpart']
        vocab, ivocab = option['vocabulary']
        vsize = len(vocab)

        sopt = extract_option(opt, 'source-embedding')
        topt = extract_option(opt, 'target-embedding')
        eopt = extract_option(opt, 'encoder')
        dopt = extract_option(opt, 'decoder')
        sopt['name'] = 'source-embedding'
        topt['name'] = 'target-embedding'

        source_embedding = embedding(vsize, sedim, **sopt)
        target_embedding = embedding(vsize, tedim, **topt)
        rnn_encoder = encoder(sedim, shdim, lhdim, **eopt)
        rnn_decoder = decoder(tedim, lhdim, thdim, chdim, maxdim, k,
                              deephid, vsize, **dopt)

        params = []
        add_parameters(params, 'rnnsearch', *source_embedding.parameter)
        add_parameters(params, 'rnnsearch', *target_embedding.parameter)
        add_parameters(params, 'rnnsearch', *rnn_encoder.parameter)
        add_parameters(params, 'rnnsearch', *rnn_decoder.parameter)

        random_stream = theano.sandbox.rng_mrg.MRG_RandomStreams()

        def build_training():
            seq = theano.tensor.imatrix()
            mask = theano.tensor.matrix()
            dseq = theano.tensor.imatrix()
            factor = theano.tensor.scalar()

            xemb = source_embedding(seq)
            yemb = target_embedding(dseq)


            initstate = theano.tensor.zeros((xemb.shape[1], shdim))

            # sample normal distribution
            epsilon = random_stream.normal((xemb.shape[1], lhdim))

            latent, mu, sigma = rnn_encoder(xemb, mask, epsilon, initstate)
            probs = rnn_decoder(yemb, mask, latent)

            idx = theano.tensor.arange(seq.flatten().shape[0])
            cost = -theano.tensor.log(probs[idx, seq.flatten()])
            cost = cost.reshape(seq.shape)
            cost = theano.tensor.sum(cost * mask, 0)
            cost = theano.tensor.mean(cost)

            # kl cost
            kl = 1 + theano.tensor.log(sigma ** 2) - mu ** 2 - sigma ** 2
            kl = theano.tensor.mean(theano.tensor.sum(-kl, 1), 0)

            cost = 0.5 * factor * kl + cost

            return [seq, dseq, mask, factor], [cost, kl]

        def build_sampling():

            def encode():
                seq = theano.tensor.imatrix()
                mask = theano.tensor.matrix()

                xemb = source_embedding(seq)
                state = theano.tensor.zeros((seq.shape[1], shdim))
                #epsilon = random_stream.normal((xemb.shape[1], lhdim))
                epsilon = theano.tensor.zeros((xemb.shape[1], lhdim))

                latent, mu, sigma = rnn_encoder(xemb, mask, epsilon, state)

                return theano.function([seq, mask], [latent, mu, sigma])

            def compute_initstate():
                latent = theano.tensor.matrix()

                # initstate, mapped_annotation
                outputs = rnn_decoder.compute_initstate(latent)

                return theano.function([latent], outputs)

            def compute_probability():
                y = theano.tensor.ivector()
                state = theano.tensor.matrix()
                context = theano.tensor.matrix()

                # 0 for initial index
                cond = theano.tensor.neq(y, 0)
                yemb = target_embedding(y)
                # zeros out embedding if y is 0
                yemb = yemb * cond[:, None]
                probs = rnn_decoder.compute_probability(yemb, state, context)

                return theano.function([y, state, context], probs)

            def compute_state():
                y = theano.tensor.ivector()
                ymask = theano.tensor.vector()
                state = theano.tensor.matrix()
                context = theano.tensor.matrix()

                yemb = target_embedding(y)
                inputs = [yemb, ymask, state, context]
                new_state = rnn_decoder.compute_state(*inputs)

                return theano.function([y, ymask, state, context], new_state)

            functions = []
            functions.append(encode())
            functions.append(compute_initstate())
            functions.append(compute_probability())
            functions.append(compute_state())

            return functions

        train_inputs, train_outputs = build_training()
        functions = build_sampling()

        self.cost = train_outputs[0]
        self.inputs = train_inputs
        self.outputs = train_outputs
        self.updates = []
        self.parameter = params
        self.sample = functions
        self.option = opt

# based on groundhog's impelmentation
def beamsearch(model, seq, **option):
    add_if_not_exsit(option, 'beamsize', 10)
    add_if_not_exsit(option, 'normalize', True)
    add_if_not_exsit(option, 'maxlen', None)
    add_if_not_exsit(option, 'minlen', None)

    functions = model.sample

    encode = functions[0]
    compute_istate = functions[1]
    compute_probs = functions[2]
    compute_state = functions[3]

    vocabulary = model.option['vocabulary']
    eos = model.option['eos']
    vocab = vocabulary[1]
    eosid = vocabulary[0][eos]

    size = option['beamsize']
    maxlen = option['maxlen']
    minlen = option['minlen']
    normalize = option['normalize']

    if maxlen == None:
        maxlen = len(seq) * 3

    if minlen == None:
        minlen = len(seq) / 2

    mask = numpy.ones(seq.shape, 'float32')
    latent, mu, sigma = encode(seq, mask)
    state, context = compute_istate(latent)

    hdim = state.shape[1]
    cdim = context.shape[1]
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

        contexts = numpy.repeat(context, num, 0)

        probs = compute_probs(last_words, states, contexts)
        logprobs = numpy.log(probs)

        if k < minlen:
            logprobs[:, eosid] = -numpy.inf

        # force to add eos symbol
        if k == maxlen - 1:
            # copy
            eosprob = logprobs[:, eosid].copy()
            logprobs[:, :] = -numpy.inf
            logprobs[:, eosid] = eosprob

        ncosts = numpy.array(costs)[:, None] - logprobs
        fcosts = ncosts.flatten()
        nbest = numpy.argpartition(fcosts, size)[:size]

        vocsize = logprobs.shape[1]
        tinds = nbest / vocsize
        winds = nbest % vocsize
        costs = fcosts[nbest]

        newtrans = [[]] * size
        newcosts = numpy.zeros(size)
        newstates = numpy.zeros((size, hdim), 'float32')
        newcontexts = numpy.zeros((size, cdim), 'float32')
        inputs = numpy.zeros(size, 'int32')

        for i, (idx, nword, ncost) in enumerate(zip(tinds, winds, costs)):
            newtrans[i] = trans[idx] + [nword]
            newcosts[i] = ncost
            newstates[i] = states[idx]
            newcontexts[i] = context
            inputs[i] = nword

        ymask = numpy.ones((size,), 'float32')
        newstates = compute_state(inputs, ymask, newstates, newcontexts)

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

    if len(final_trans) == 0:
        final_trans = [[]]
        final_costs = [0.0]

    for i, (cost, trans) in enumerate(zip(final_costs, final_trans)):
        count = len(trans)
        if count > 0:
            if normalize:
                final_costs[i] = cost / count
            else:
                final_costs[i] = cost

    final_trans = numpy.array(final_trans)[numpy.argsort(final_costs)]
    final_costs = numpy.array(sorted(final_costs))

    translations = []

    for cost, trans in zip(final_costs, final_trans):
        trans = map(lambda x: vocab[x], trans)
        translations.append((trans, cost))

    return translations
