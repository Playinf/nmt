# buildvocab.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import argparse
import operator
import cPickle

def loadpkl(name):
    fd = open(name, 'r')
    vocab = cPickle.load(fd)
    fd.close()
    return vocab

def processline(line, mode = 'plain'):
    if mode == 'plain':
        return processplain(line)
    if mode == 'lattice':
        return processlattice(line)

def processplain(line):
    return line.strip().split()

def processlattice(line):
    line = line.split(' ')
    vlist = [v.split(':') for v in line]
    vlist = [v[0] for v in vlist]
    return vlist

def countword(name, mode):
    fd = open(name, 'r')
    vocab = {}

    for line in fd:
        wordlist = processline(line, mode)
        for word in wordlist:
            vocab[word] = 1 if word not in vocab else vocab[word] + 1

    fd.close()

    return vocab

def countchar(name, mode):
    fd = open(name, 'r')
    vocab = {}

    for line in fd:
        wordlist = processline(line, mode)
        for word in wordlist:
            word = word.decode('utf-8')
            for char in word:
                char = char.encode('utf-8')
                vocab[char] = 1 if char not in vocab else vocab[char] + 1

    fd.close()

    return vocab

def sortbyfreq(vocab):
    tup = [(item[0], item[1]) for item in vocab.items()]
    tup = sorted(tup, key = operator.itemgetter(0))
    tup = sorted(tup, key = operator.itemgetter(1), reverse = True)
    return [item[0] for item in tup]

def sortbyalpha(vocab):
    tup = sorted(vocab)
    return tup

def save(name, voc):
    newvoc = {}
    for i, v in enumerate(voc):
        newvoc[v] = i

    fd = open(name, 'w')
    cPickle.dump(newvoc, fd)
    fd.close()

def parsetokens(s):
    tlist = s.split(';')
    return tlist

def removespecial(vocab, tokens):
    for tok in tokens:
        if tok in vocab:
            del vocab[tok]

    return vocab

def inserttokens(vocab, tokens):
    tokens = tokens[::-1]

    for tok in tokens:
        vocab.insert(0, tok)

    return vocab

def parsearg():
    desc = 'build vocabulary'
    parser = argparse.ArgumentParser(description = desc)

    desc = 'corpus'
    parser.add_argument('--corpus', required = True, help = desc)
    desc = 'output'
    parser.add_argument('--output', required = True, help = desc)
    desc = 'limit'
    parser.add_argument('--limit', default = 0, type = int, help = desc)
    desc = 'character mode'
    parser.add_argument('--char', action = 'store_true', help = 'desc')
    desc = 'sort by alphabet'
    parser.add_argument('--alpha', action = 'store_true', help = 'desc')
    desc = 'add token'
    parser.add_argument('--token', type = str, help = desc)
    desc = 'mode'
    parser.add_argument('--mode', default = 'plain', type = str, help = desc)

    return parser.parse_args()

if __name__ == '__main__':
    args = parsearg()

    if args.char:
        vocab = countchar(args.corpus, args.mode)
    else:
        vocab = countword(args.corpus, args.mode)

    if args.token != None:
        tokens = parsetokens(args.token)
    else:
        tokens = []

    vocab = removespecial(vocab, tokens)
    vocab = sortbyfreq(vocab)
    vocab = inserttokens(vocab, tokens)

    if args.limit != 0:
        vocab = vocab[:args.limit]

    if args.alpha:
        n = len(tokens)
        vocab = vocab[:n] + sorted(vocab[n:])

    save(args.output, vocab)
