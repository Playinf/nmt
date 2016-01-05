# savebin.py

import sys
import numpy
import struct
import cPickle

def saveparam(fd, param):
    shape = param.shape
    
    if len(shape) == 1:
        fd.write(' '.join([str(s) for s in shape]) + '\n')
        for i in range(shape[0]):
            fd.write(struct.pack('f', param[i]))
        fd.write('\n')
    elif len(shape) == 2:
        fd.write(' '.join([str(s) for s in shape]) + '\n')
        for i in range(shape[0]):
            for j in range(shape[1]):
                fd.write(struct.pack('f', param[i, j]))
        fd.write('\n')
    else:
        raise RuntimeError('unsupported shape')
        
def saveparams(name, params):
    fd = open(name, 'wb')
    
    for param in params:
        saveparam(fd, param)

    fd.close()
        
def readbin(filename):
    fd = open(filename, 'rb')
    tlist = []

    while True:
        line = fd.readline()

        if line == '':
            break

        ilist = line.strip().split(' ')
        slist = [int(item) for item in ilist]
        stup = tuple(slist)
        tensor = numpy.zeros(stup, 'float32')

        if len(stup) == 1:
            for i in xrange(stup[0]):
                tensor[i] = struct.unpack('f', fd.read(4))[0]
        else:
            for i in xrange(stup[0]):
                for j in xrange(stup[1]):
                    tensor[i, j] = struct.unpack('f', fd.read(4))[0]

        tlist.append(tensor)
        fd.readline()

    fd.close()

    return tlist

if __name__ == '__main__':
    fdr = open(sys.argv[1], 'r')

    options = cPickle.load(fdr)
    params = cPickle.load(fdr)
    fdr.close()
    
    saveparams(sys.argv[2], params)
