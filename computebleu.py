# computebleu.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import argparse

from bleu import bleu
from utils import batchstream

def loadreferences(names):
    references = []
    stream = batchstream(names)

    for data in stream:
        newdata= []
        for batch in data:
            line = batch[0]
            words = line.strip().split()
            lower = [word.lower() for word in words]
            newdata.append(lower)

        references.append(newdata)

    stream.close()

    return references

def loadtranslation(name):
    translation = []
    fd = open(name)

    for line in fd:
        line = line.strip().split()
        translation.append(line)

    fd.close()

    return translation

def parseargs(args = None):
    desc = 'compute bit bper sequence'
    parser = argparse.ArgumentParser(description = desc)

    # training corpus
    desc = 'references'
    parser.add_argument('--ref', nargs = '+', required = True, help = desc)
    # translation
    desc = 'translation'
    parser.add_argument('--trans', required = True, help = desc)

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parseargs()
    refs = loadreferences(args.ref)
    trans = loadtranslation(args.trans)
    bleuscore = bleu(trans, refs)
    print 'bleu:', bleuscore
