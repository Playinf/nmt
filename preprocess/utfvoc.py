# utfvoc.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys
import cPickle

def convertvoc(voc):
    newvoc = {}

    for key in voc:
        newvoc[key.decode('utf-8')] = voc[key]

    return newvoc

if __name__ == '__main__':
    fd = open(sys.argv[1], 'r')
    voc = cPickle.load(fd)
    fd.close()

    cvoc = convertvoc(voc)

    fd = open(sys.argv[2], 'w')
    cPickle.dump(cvoc, fd)
    fd.close()
