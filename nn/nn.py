# nn.py
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

# embedding lookup
class embedder:

    def __init__(self, embdim = None):
        params = []

        if embdim != None:
            bias = numpy.zeros((embdim,))
            bias = theano.shared(bias.astype(theano.config.floatX))
            params.append(bias)

        self.parameter = params

    def __call__(self, emb, indices):
        shape = list(indices.shape) + [-1]
        values = emb[indices.flatten()]
        values = values.reshape(shape)

        if len(self.parameter) == 0:
            return values
        else:
            bias = self.parameter[0]
            return values + bias

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

    def __call__(self, seq, mask, ih):
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

    def __call__(self, yseq, xmask, ymask, anno, inis):
        tanh = theano.tensor.tanh
        sigmoid = theano.tensor.nnet.sigmoid
        wa, ws, wc = self.parameter[:3]
        w, c, u, b = self.parameter[3:7]
        wr, cr, ur, wz, cz, uz = self.parameter[7:]

        manno = theano.dot(anno, wa)

        def attention(xm, s, ma, ws, wc):
            ms = theano.dot(s, ws)
            e = theano.dot(tanh(ms + ma), wc)
            e = e.reshape((e.shape[0], e.shape[1]))
            alpha = theano.tensor.exp(e)
            alpha = alpha * xm
            alpha = alpha / theano.tensor.sum(alpha, 0)
            return alpha

        def grustep(y, ct, s, wr, cr, ur, wz, cz, uz, w, c, u):
            r = theano.dot(y, wr) + theano.dot(ct, cr) + theano.dot(s, ur)
            z = theano.dot(y, wz) + theano.dot(ct, cz) + theano.dot(s, uz)
            r = sigmoid(r)
            z = sigmoid(z)
            t = theano.dot(y, w) + theano.dot(ct, c) + theano.dot(r * s, u)
            t = tanh(t + b)
            ns = (1.0 - z) * s + z * t
            return ns

        def step(y, m, s, xm, a, ma, ws, wc, *p):
            w, c, u, b, wr, cr, ur, wz, cz, uz = p
            alpha = attention(xm, s, ma, ws, wc)
            ct = theano.tensor.sum(alpha[:, :, None] * a, 0)
            ns = grustep(y, ct, s, wr, cr, ur, wz, cz, uz, w, c, u)
            ns = (1.0 - m[:, None]) * s + m[:, None] * ns
            return [ns, ct]

        seq = [yseq, ymask]
        oinfo = [inis, None]
        nonseq = [xmask, anno, manno, ws, wc]
        nonseq += [w, c, u, b, wr, cr, ur, wz, cz, uz]
        (state, context), u = theano.scan(step, seq, oinfo, nonseq)

        return [state, context]
