# charvoc.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys
import cPickle

def invertvoc(voc):
    ivoc = {}

    for key in voc:
        ivoc[voc[key]] = key

    return ivoc

def convertvoc(voc):
    newvoc = {}
    count = 2

    for key in voc:
        chars = voc[key].decode('utf-8')

        for char in chars:
            char = char.encode('utf-8')
            if char in newvoc:
                continue
            newvoc[char] = count
            count = count + 1

    return newvoc

if __name__ == '__main__':
    fd = open(sys.argv[1], 'r')
    voc = cPickle.load(fd)
    fd.close()

    del voc['</s>']
    del voc['UNK']

    ivoc = invertvoc(voc)
    cvoc = convertvoc(ivoc)

    cvoc['</s>'] = 0
    cvoc['UNK'] = 1

    fd = open(sys.argv[2], 'w')
    cPickle.dump(cvoc, fd)
    fd.close()
