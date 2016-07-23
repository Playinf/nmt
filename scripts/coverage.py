# converage.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys
import cPickle

def processlattice(line):
    nstate, chars, lattice = line.strip().split('|||')
    chars = chars.strip().split()
    arcs = lattice.strip().split()
    wordlist = []

    for arc in arcs:
        val = arc[1:-1].split(',')
        sid = int(val[0])
        eid = int(val[1])
        wordlist.append(''.join(chars[sid:eid]))

    return wordlist

def countword(name):
    fd = open(name, 'r')
    vocab = {}

    for line in fd:
        wordlist = processlattice(line)
        for word in wordlist:
            vocab[word] = 1 if word not in vocab else vocab[word] + 1

    fd.close()

    return vocab

def coverage(voc, counts):
    n = 0
    total = sum(counts.itervalues())

    for key in voc:
        if key in counts:
            n += counts[key]

    return float(n) / float(total)

if __name__ == '__main__':
    fd = open(sys.argv[1])
    voc = cPickle.load(fd)
    fd.close()

    counts = countword(sys.argv[2])

    print coverage(voc, counts)
