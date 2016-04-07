# buildvoc.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys
import cPickle

def getcvoc(wvoc):
    cvoc = {}

    for key in wvoc:
        key = key.decode('utf-8')
        for char in key:
            char = char.encode('utf-8')
            cvoc[char] = 1 if char not in cvoc else cvoc[char] + 1

    clist = [item for item in cvoc]

    clist.insert(0, 'UNK')
    clist.insert(0, '</s>')

    vocab = {}
    count = 0

    for item in clist:
        vocab[item] = count
        count = count + 1

    return vocab

if __name__ == '__main__':
    fd = open(sys.argv[1], 'r')
    voc = cPickle.load(fd)
    fd.close()

    cvoc = getcvoc(voc)

    fd = open(sys.argv[2], 'w')
    cPickle.dump(cvoc, fd)
    fd.close()
