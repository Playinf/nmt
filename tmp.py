# tmp.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano

def aligncenter(s, slen, wp, vp):
    pos = theano.dot(theano.dot(s, wp), vp)[:, 0]
    pos = theano.tensor.nnet.sigmoid(pos) * (slen - 1)
    return pos

def getlocal(xm, a, ma, pos, idx1, idx2):
    pos = theano.tensor.floor(pos)
    pos = theano.tensor.cast(pos, 'int32')
    idx1 = pos[None, :] + idx1
    lvec = ma[idx1, idx2]
    lmask = xm[idx1, idx2]
    la = a[idx1, idx2]
    return lmask, lvec, la

def attention(s, lvec, lmask, ws, wc):
    # linear mapping
    ms = theano.dot(s, ws)
    e = theano.dot(theano.tensor.tanh(ms + lvec), wc)
    e = e.reshape((e.shape[0], e.shape[1]))
    alpha = theano.tensor.exp(e)
    alpha = alpha * lmask
    alpha = alpha / theano.tensor.sum(alpha, 0)
    return alpha

# exp(-\frac{(x - mu)^2}{2 * sigma^2})
def gaussian(x, mu, sigma):
    scale = (x - mu) / sigma
    return theano.tensor.exp(-0.5 * (scale ** 2))

def padding(xmask, annot, mannot, wsize):
    # index for finding local source annotation
    idx1 = theano.tensor.arange(2 * wsize + 1)[:, None]
    idx2 = theano.tensor.arange(xmask.shape[1])
    idx2 = theano.tensor.repeat(idx2[None, :], 2 * wsize + 1, 0)

    bat = xmask.shape[1]
    # padding mask
    pad = theano.tensor.zeros((wsize, bat), theano.config.floatX)
    pmask = theano.tensor.concatenate([pad, xmask, pad], 0)
    # padding annotation
    dim = annot.shape[2]
    pad = theano.tensor.zeros((wsize, bat, dim), theano.config.floatX)
    pannot = theano.tensor.concatenate([pad, annot, pad], 0)
    # padding mapped annotation
    dim = mannot.shape[2]
    pad = theano.tensor.zeros((wsize, bat, dim), theano.config.floatX)
    pmannot = theano.tensor.concatenate([pad, mannot, pad], 0)

    return [idx1, idx2, pmask, pannot, pmannot]

def build(tsize, asize, wsize):
    ws = numpy.random.uniform(-0.05, 0.05, (tsize, asize))
    ws = theano.shared(ws.astype(theano.config.floatX))

    # context transform
    wc = numpy.random.uniform(-0.05, 0.05, (asize, 1))
    wc = theano.shared(wc.astype(theano.config.floatX))

    wp = numpy.random.uniform(-0.05, 0.05, (tsize, asize))
    wp = theano.shared(wp.astype(theano.config.floatX), 'wp')
    vp = numpy.random.uniform(-0.05, 0.05, (asize, 1))
    vp = theano.shared(vp.astype(theano.config.floatX), 'vp')

    s = theano.tensor.matrix()
    xmask = theano.tensor.matrix()
    annot = theano.tensor.tensor3()
    mannot = theano.tensor.tensor3()

    slen = theano.tensor.sum(xmask, 0)
    idx1, idx2, pmask, pannot, pmannot = padding(xmask, annot, mannot, wsize)

    pos = aligncenter(s, slen, wp, vp)
    lmask, lvec, la = getlocal(pmask, pannot, pmannot, pos, idx1, idx2)
    alpha = attention(s, lvec, lmask, ws, wc)
    dist = gaussian(slen, pos, wsize / 2)
    alpha = alpha * dist
    ct = theano.tensor.sum(alpha[:, :, None] * la, 0)
    cost = theano.tensor.sum(ct)

    grad = theano.grad(cost, [ws, wc, wp, vp])

    return theano.function([s, xmask, annot, mannot], grad, on_unused_input = 'ignore')

def generate_mask(maxlen, batch):
    length = numpy.random.randint(1, maxlen + 1, (batch,))
    mask = numpy.zeros((numpy.max(length), batch)).astype('float32')

    for i, mlen in enumerate(length):
        mask[:mlen, i] = 1.0

    return mask

if __name__ == '__main__':
    mask = generate_mask(20, 128)
    s = numpy.random.uniform(size = (128, 1000)).astype('float32')
    annot = numpy.random.uniform(size = (20, 128, 1000)).astype('float32')
    mannot = numpy.random.uniform(size = (20, 128, 1000)).astype('float32')

    func = build(1000, 1000, 10)

    result = func(s, mask, annot, mannot)

