# convert.py
# convert groundhog's search_model.npz to our format
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import numpy
import cPickle
import argparse

from trainnmt import getoption, override, getfilename

def convert(name):
    mapping = get_mapping()

    fd = numpy.load(name)
    params = {}

    for item in fd:
        params[item] = fd[item]

    plist = list(params)

    for item in mapping:
        plist[mapping[item]] = params[item]

    fd.close()

    return plist

def serialize(name, params, option):
    fd = open(name, 'w')
    cPickle.dump(option, fd)
    cPickle.dump(params, fd)
    fd.close()

def get_mapping():
    mapping = {}
    # encoder
    mapping['W_0_enc_approx_embdr'] = 0
    mapping['b_0_enc_approx_embdr'] = 1
    mapping['W_0_enc_input_embdr_0'] = 2
    mapping['W_enc_transition_0'] = 3
    mapping['b_0_enc_input_embdr_0'] = 4
    mapping['W_0_enc_reset_embdr_0'] = 5
    mapping['R_enc_transition_0'] = 6
    mapping['W_0_enc_update_embdr_0'] = 7
    mapping['G_enc_transition_0'] = 8
    mapping['W_0_back_enc_input_embdr_0'] = 9
    mapping['W_back_enc_transition_0'] = 10
    mapping['b_0_back_enc_input_embdr_0'] = 11
    mapping['W_0_back_enc_reset_embdr_0'] = 12
    mapping['R_back_enc_transition_0'] = 13
    mapping['W_0_back_enc_update_embdr_0'] = 14
    mapping['G_back_enc_transition_0'] = 15
    # decoder
    mapping['W_0_dec_approx_embdr'] = 16
    mapping['b_0_dec_approx_embdr'] = 17
    mapping['W_0_dec_initializer_0'] = 18
    mapping['b_0_dec_initializer_0'] = 19
    mapping['A_dec_transition_0'] = 20
    mapping['B_dec_transition_0'] = 21
    mapping['D_dec_transition_0'] = 22
    mapping['W_0_dec_hid_readout_0'] = 23
    mapping['W_0_dec_prev_readout_0'] = 24
    mapping['W_0_dec_repr_readout'] = 25
    mapping['b_0_dec_hid_readout_0'] = 26
    mapping['W1_dec_deep_softmax'] = 27
    mapping['W2_dec_deep_softmax'] = 28
    mapping['b_dec_deep_softmax'] = 29
    # decoder rnn
    mapping['W_0_dec_input_embdr_0'] = 30
    mapping['W_0_dec_dec_inputter_0'] = 31
    mapping['W_dec_transition_0'] = 32
    mapping['b_0_dec_input_embdr_0'] = 33
    mapping['W_0_dec_reset_embdr_0'] = 34
    mapping['W_0_dec_dec_reseter_0'] = 35
    mapping['R_dec_transition_0'] = 36
    mapping['W_0_dec_update_embdr_0'] = 37
    mapping['W_0_dec_dec_updater_0'] = 38
    mapping['G_dec_transition_0'] = 39

    return mapping

def parseargs(args = None):
    desc = 'convert search_model.npz to our format'
    parser = argparse.ArgumentParser(description = desc)

    # training corpus
    desc = 'source and target corpus'
    parser.add_argument('--corpus', nargs = 2, help = desc)
    # training vocabulary
    desc = 'source and target vocabulary'
    parser.add_argument('--vocabulary', nargs = 2, help = desc)
    # output model
    desc = 'saved model'
    parser.add_argument('--model', required = True, help = desc)

    # embedding size
    desc = 'source and target embedding size'
    parser.add_argument('--embdim', nargs = 2, type = int, help = desc)
    # hidden size
    desc = 'source, target and alignment hidden size'
    parser.add_argument('--hidden', nargs = 3, type = int, help = desc)
    # maxout dim
    desc = 'maxout hidden dimension'
    parser.add_argument('--maxhid', default = 500, type = int, help = desc)
    # maxout number
    desc = 'maxout number'
    parser.add_argument('--maxpart', default = 2, type = int, help = desc)
    # deepout dim
    desc = 'deepout hidden dimension'
    parser.add_argument('--deephid', default = 620, type = int, help = desc)

    # epoch
    desc = 'maximum training epoch'
    parser.add_argument('--maxepoch', default = 10, type = int, help = desc)
    # learning rate
    desc = 'learning rate'
    parser.add_argument('--alpha', default = 1e-4, type = float, help = desc)
    # momentum
    desc = 'momentum'
    parser.add_argument('--momentum', default = 0.0, type = float, help = desc)
    # batch
    desc = 'batch size'
    parser.add_argument('--batch', type = int, default = 128, help = desc)
    # training algorhtm
    desc = 'optimizer'
    parser.add_argument('--optimizer', type = str, help = desc)
    # gradient renormalization
    desc = 'gradient renormalization'
    parser.add_argument('--norm', type = int, default = 1.0, help = desc)

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parseargs()
    option = getoption()
    init = True

    override(option, args)
    init = True

    svocabs, tvocabs = option['vocabulary']
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    pathname, basename = os.path.split(args.model)
    modelname = getfilename(basename)
    savedname = modelname + '.converted.pkl'
    params = convert(basename)
    serialize(savedname, params, option)
