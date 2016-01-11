# rnnsearch.py
# fast version of rnnsearch
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano

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

# gated recurrent unit
class gru:

    def __init__(self, isize, osize):
        # transform
        w = numpy.random.uniform(-0.05, 0.05, (isize, osize))
        w = theano.shared(w.astype(theano.config.floatX))
        u = numpy.random.uniform(-0.05, 0.05, (osize, osize))
        u = theano.shared(u.astype(theano.config.floatX))
        b = numpy.zeros((osize,))
        b = theano.shared(b.astype(theano.config.floatX))
        # reset gate
        wr = numpy.random.uniform(-0.05, 0.05, (isize, osize))
        wr = theano.shared(wr.astype(theano.config.floatX))
        ur = numpy.random.uniform(-0.05, 0.05, (osize, osize))
        ur = theano.shared(ur.astype(theano.config.floatX))
        # update gate
        wz = numpy.random.uniform(-0.05, 0.05, (isize, osize))
        wz = theano.shared(wz.astype(theano.config.floatX))
        uz = numpy.random.uniform(-0.05, 0.05, (osize, osize))
        uz = theano.shared(uz.astype(theano.config.floatX))

        self.parameter = [w, u, b, wr, ur, wz, uz]

    def build(self, seq, mask, ih):
        tanh = theano.tensor.tanh
        sigmoid = theano.tensor.nnet.sigmoid
        w, u, b, wr, ur, wz, uz = self.parameter

        def step(x, m, h, w, u, b, wr, ur, wz, uz):
            r = sigmoid(theano.dot(x, wr) + theano.dot(h, ur))
            z = sigmoid(theano.dot(x, wz) + theano.dot(h, uz))
            c = tanh(theano.dot(x, w) + theano.dot(r * h, u) + b)
            nh = (1.0 - z) * h + z * c
            nh = (1.0 - m[:, None]) * h + m[:, None] * nh
            return [nh]

        o, u = theano.scan(step, [seq, mask], [ih], [w, u, b, wr, ur, wz, uz])

        return o

# gated recurrent unit with search
class grusearch:

    def __init__(self, esize, ssize, tsize, asize):
        csize = 2 * ssize
        params = []

        # annotation transform
        wa = numpy.random.uniform(-0.05, 0.05, (csize, asize))
        wa = theano.shared(wa.astype(theano.config.floatX))
        params.extend([wa])

        # state transform
        ws = numpy.random.uniform(-0.05, 0.05, (tsize, asize))
        ws = theano.shared(ws.astype(theano.config.floatX))
        params.extend([ws])

        # context transform
        wc = numpy.random.uniform(-0.05, 0.05, (asize, 1))
        wc = theano.shared(wc.astype(theano.config.floatX))
        params.extend([wc])

        # transform
        w = numpy.random.uniform(-0.05, 0.05, (esize, tsize))
        w = theano.shared(w.astype(theano.config.floatX))
        c = numpy.random.uniform(-0.05, 0.05, (csize, tsize))
        c = theano.shared(c.astype(theano.config.floatX))
        u = numpy.random.uniform(-0.05, 0.05, (tsize, tsize))
        u = theano.shared(u.astype(theano.config.floatX))
        b = numpy.zeros((tsize,))
        b = theano.shared(b.astype(theano.config.floatX))
        params.extend([w, c, u, b])
        # reset gate
        wr = numpy.random.uniform(-0.05, 0.05, (esize, tsize))
        wr = theano.shared(wr.astype(theano.config.floatX))
        cr = numpy.random.uniform(-0.05, 0.05, (csize, tsize))
        cr = theano.shared(cr.astype(theano.config.floatX))
        ur = numpy.random.uniform(-0.05, 0.05, (tsize, tsize))
        ur = theano.shared(ur.astype(theano.config.floatX))
        params.extend([wr, cr, ur])
        # update gate
        wz = numpy.random.uniform(-0.05, 0.05, (esize, tsize))
        wz = theano.shared(wz.astype(theano.config.floatX))
        cz = numpy.random.uniform(-0.05, 0.05, (csize, tsize))
        cz = theano.shared(cz.astype(theano.config.floatX))
        uz = numpy.random.uniform(-0.05, 0.05, (tsize, tsize))
        uz = theano.shared(uz.astype(theano.config.floatX))
        params.extend([wz, cz, uz])

        self.parameter = params

    def build(self, yseq, xmask, ymask, anno, inis):
        tanh = theano.tensor.tanh
        sigmoid = theano.tensor.nnet.sigmoid
        wa, ws, wc = self.parameter[:3]
        w, c, u, b = self.parameter[3:7]
        wr, cr, ur, wz, cz, uz = self.parameter[7:]

        manno = theano.dot(anno, wa)

        def step(y, m, s, xm, a, ma, ws, wc, *p):
            w, c, u, b, wr, cr, ur, wz, cz, uz = p
            ms = theano.dot(s, ws)
            e = theano.dot(tanh(ms + ma), wc)
            e = e.reshape((e.shape[0], e.shape[1]))
            alpha = theano.tensor.exp(e)
            alpha = alpha * xm
            alpha = alpha / theano.tensor.sum(alpha, 0)
            ct = theano.tensor.sum(alpha[:, :, None] * a, 0)
            r = theano.dot(y, wr) + theano.dot(ct, cr) + theano.dot(s, ur)
            z = theano.dot(y, wz) + theano.dot(ct, cz) + theano.dot(s, uz)
            r = sigmoid(r)
            z = sigmoid(z)
            t = theano.dot(y, w) + theano.dot(ct, c) + theano.dot(r * s, u)
            t = tanh(t + b)
            ns = (1.0 - z) * s + z * t
            ns = (1.0 - m[:, None]) * s + m[:, None] * ns
            return [ns, ct]

        seq = [yseq, ymask]
        oinfo = [inis, None]
        nonseq = [xmask, anno, manno, ws, wc]
        nonseq += [w, c, u, b, wr, cr, ur, wz, cz, uz]
        (state, context), u = theano.scan(step, seq, oinfo, nonseq)

        return [state, context]

class encoder:

    def __init__(self, esize, hsize):
        forward_encoder = gru(esize, hsize)
        backward_encoder = gru(esize, hsize)

        params = forward_encoder.parameter + backward_encoder.parameter

        self.module = [forward_encoder, backward_encoder]
        self.parameter = params
        self.size = [esize, hsize]

    def build(self, xseq, xmask):
        esize, hsize = self.size
        fencoder, bencoder = self.module

        h = theano.tensor.zeros((xseq.shape[1], hsize))

        hf = fencoder.build(xseq, xmask, h)
        hb = bencoder.build(xseq[::-1], xmask[::-1], h)
        hb = hb[::-1]

        h = theano.tensor.concatenate([hf, hb], 2)

        return h

class decoder:

    def __init__(self, esize, ssize, tsize, asize, msize, mnum, dsize, vsize):
        csize = 2 * ssize

        params = []

        # initial state transform
        wi = numpy.random.uniform(-0.05, 0.05, (ssize, tsize))
        wi = theano.shared(wi.astype(theano.config.floatX))
        bi = numpy.zeros((tsize,))
        bi = theano.shared(bi.astype(theano.config.floatX))
        params.extend([wi, bi])

        # main decoder
        dec = grusearch(esize, ssize, tsize, asize)
        params.extend(dec.parameter)

        # maxout
        um = numpy.random.uniform(-0.05, 0.05, (tsize, msize * mnum))
        um = theano.shared(um.astype(theano.config.floatX))
        vm = numpy.random.uniform(-0.05, 0.05, (esize, msize * mnum))
        vm = theano.shared(vm.astype(theano.config.floatX))
        cm = numpy.random.uniform(-0.05, 0.05, (csize, msize * mnum))
        cm = theano.shared(cm.astype(theano.config.floatX))
        bm = numpy.zeros((msize * mnum,))
        bm = theano.shared(bm.astype(theano.config.floatX))
        params.extend([um, vm, cm, bm])

        # deep out
        wd = numpy.random.uniform(-0.05, 0.05, (msize, dsize))
        wd = theano.shared(wd.astype(theano.config.floatX))
        params.extend([wd])

        # classification
        w = numpy.random.uniform(-0.05, 0.05, (dsize, vsize))
        w = theano.shared(w.astype(theano.config.floatX))
        b = numpy.zeros((vsize,))
        b = theano.shared(b.astype(theano.config.floatX))
        params.extend([w, b])

        self.module = [dec]
        self.parameter = params
        self.maxpart = mnum

    def build(self, yseq, xmask, ymask, annotation):
        dec = self.module[0]
        wi, bi = self.parameter[:2]
        um, vm, cm, bm, wd, w, b = self.parameter[-7:]
        k = self.maxpart

        yshift = theano.tensor.zeros_like(yseq)
        yshift = theano.tensor.set_subtensor(yshift[1:], yseq[:-1])

        # init state
        hb = annotation[0, :, -annotation.shape[2] / 2:]
        inis = theano.tensor.tanh(theano.dot(hb, wi) + bi)

        states, contexts = dec.build(yseq, xmask, ymask, annotation, inis)
        states = theano.tensor.concatenate([inis[None, :, :], states], 0)

        pstates = states[:-1]

        z = theano.dot(pstates, um)
        z += theano.dot(yshift, vm)
        z += theano.dot(contexts, cm)
        z += bm
        mhid = z.reshape((z.shape[0], z.shape[1], z.shape[2] / k, k))
        mhid = theano.tensor.max(mhid, 3)

        deepout = theano.dot(mhid, wd)
        preact = theano.dot(deepout, w) + b
        preact = preact.reshape((preact.shape[0] * preact.shape[1], -1))
        prob = theano.tensor.nnet.softmax(preact)

        return prob

    def build_compute_init_state(self, annotation):
        wi, bi = self.parameter[:2]
        hb = annotation[0, :, -annotation.shape[2] / 2:]
        inis = theano.tensor.tanh(theano.dot(hb, wi) + bi)

        return inis

    def build_compute_prob(self, yemb, xmask, ymask, state, annotation):
        dec = self.module[0]
        um, vm, cm, bm, wd, w, b = self.parameter[-7:]
        k = self.maxpart

        yemb = yemb[None, :, :]

        dummy, context = dec.build(yemb, xmask, ymask, annotation, state)

        state = state[None, :, :]

        z = theano.dot(state, um)
        z += theano.dot(yemb, vm)
        z += theano.dot(context, cm)
        z += bm
        mhid = z.reshape((z.shape[0], z.shape[1], z.shape[2] / k, k))
        mhid = theano.tensor.max(mhid, 3)

        deepout = theano.dot(mhid, wd)
        preact = theano.dot(deepout, w) + b
        preact = preact.reshape((preact.shape[0] * preact.shape[1], -1))
        prob = theano.tensor.nnet.softmax(preact)

        return prob

    def build_compute_state(self, yemb, xmask, ymask, state, annotation):
        dec = self.module[0]

        yemb = yemb[None, :, :]

        state, context = dec.build(yemb, xmask, ymask, annotation, state)

        return state[0]

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

        semb = embedding(svsize, sedim)
        sembedder = embedder(sedim)
        temb = embedding(tvsize, tedim)
        tembedder = embedder(tedim)

        enc = encoder(sedim, shdim)
        dec = decoder(tedim, shdim, thdim, ahdim, maxdim, k, deephid, tvsize)

        params = []
        params.extend(semb.parameter)
        params.extend(sembedder.parameter)
        params.extend(enc.parameter)
        params.extend(temb.parameter)
        params.extend(tembedder.parameter)
        params.extend(dec.parameter)

        def build_training():
            x = theano.tensor.imatrix()
            y = theano.tensor.imatrix()
            xmask = theano.tensor.matrix()
            ymask = theano.tensor.matrix()

            xseq = sembedder(semb(), x)
            yseq = tembedder(temb(), y)

            annotation = enc.build(xseq, xmask)
            probs = dec.build(yseq, xmask, ymask, annotation)

            idx = theano.tensor.arange(y.flatten().shape[0])
            cost = -theano.tensor.log(probs[idx, y.flatten()])
            cost = cost.reshape((y.shape[0], y.shape[1]))
            cost = theano.tensor.sum(cost * ymask)

            return [x, xmask, y, ymask], [cost]

        def build_sampling():

            def encode():
                x = theano.tensor.imatrix()
                xmask = theano.tensor.ones_like(x, theano.config.floatX)

                xseq = sembedder(semb(), x)

                annotation = enc.build(xseq, xmask)

                return theano.function([x], annotation)

            def compute_istate():
                a = theano.tensor.tensor3()

                state = dec.build_compute_init_state(a)

                return theano.function([a], state)

            def compute_prob():
                y = theano.tensor.ivector()
                a = theano.tensor.tensor3()
                s = theano.tensor.matrix()

                cond = theano.tensor.neq(y, 0)

                xmask = theano.tensor.ones((a.shape[0], a.shape[1]))
                ymask = theano.tensor.ones((1, y.shape[0]))

                yemb = tembedder(temb(), y)
                yemb = yemb * cond[:, None]
                prob = dec.build_compute_prob(yemb, xmask, ymask, s, a)

                return theano.function([y, a, s], prob)

            def compute_state():
                y = theano.tensor.ivector()
                a = theano.tensor.tensor3()
                s = theano.tensor.matrix()

                xmask = theano.tensor.ones((a.shape[0], a.shape[1]))
                ymask = theano.tensor.ones((1, y.shape[0]))

                yemb = tembedder(temb(), y)
                state = dec.build_compute_state(yemb, xmask, ymask, s, a)

                return theano.function([y, a, s], state)

            return encode(), compute_istate(), compute_prob(), compute_state()

        inputs, outputs = build_training()
        gradient = theano.grad(outputs[0], params)

        self.input = inputs
        self.output = outputs
        self.option = option
        self.module = [enc, dec]
        self.gradient = gradient
        self.sample = build_sampling()
        self.parameter = params
        self.vocabulary = option['vocabulary']
