import sys
import numpy
import theano
import cPickle

sys.path.append('/home/playinf/Workspace/scripts/theano/nmtbaseline/groundhog')

from experiments.nmt.state import prototype_state
from experiments.nmt.encdec import RNNEncoderDecoder, parse_input
from experiments.nmt import get_batch_iterator
from rnnsearch import rnnsearch
from fastrnnsearch import rnnsearch as fastrnnsearch

if __name__ == '__main__':
    state = prototype_state()
    with open('search_state.pkl', 'r') as fd:
        state.update(cPickle.load(fd))
    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load('search_model.npz')
    comp_repr = enc_dec.create_representation_computer()
    comp_init_states = enc_dec.create_initializers()
    comp_next_probs = enc_dec.create_next_probs_computer()
    comp_next_states = enc_dec.create_next_states_computer()

    fd = open('sentence', 'r')
    line = fd.readline()
    line = line.strip()
    idict_src = cPickle.load(open(state['indx_word'],'r'))
    indx_word = cPickle.load(open(state['word_indx'],'rb'))
    seq, parsed_in = parse_input(state, indx_word, line, idx2word = idict_src)

    c = comp_repr(seq)[0]
    states = comp_init_states(c)
    states = map(lambda x : x[None, :], states)
    p = comp_next_probs(c, 1, numpy.array([0]).astype('int64'), *states)[0]
    new_states = comp_next_states(c, 1, numpy.array([2440]).astype('int64'), *states)

    option = {}
    option['embdim'] = [620, 620]
    option['hidden'] = [1000, 1000, 1000]
    option['maxhid'] = 500
    option['deephid'] = 620
    option['maxpart'] = 2
    option['vocabulary'] = [[idict_src, indx_word], [idict_src, indx_word]]

    model = rnnsearch(**option)
    fmodel = fastrnnsearch(**option)
    params = model.parameter
    params = [item.get_value() for item in params]
    fparams = [item.get_value() for item in fmodel.parameter]

    train_lm = theano.function(lm_model.inputs, [lm_model.train_cost])

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
    # decoder rnn
    mapping['W_0_dec_input_embdr_0'] = 23
    mapping['W_0_dec_dec_inputter_0'] = 24
    mapping['W_dec_transition_0'] = 25
    mapping['b_0_dec_input_embdr_0'] = 26
    mapping['W_0_dec_reset_embdr_0'] = 27
    mapping['W_0_dec_dec_reseter_0'] = 28
    mapping['R_dec_transition_0'] = 29
    mapping['W_0_dec_update_embdr_0'] = 30
    mapping['W_0_dec_dec_updater_0'] = 31
    mapping['G_dec_transition_0'] = 32
    # maxout, deepout and classification
    mapping['W_0_dec_hid_readout_0'] = 33
    mapping['W_0_dec_prev_readout_0'] = 34
    mapping['W_0_dec_repr_readout'] = 35
    mapping['b_0_dec_hid_readout_0'] = 36
    mapping['W1_dec_deep_softmax'] = 37
    mapping['W2_dec_deep_softmax'] = 38
    mapping['b_dec_deep_softmax'] = 39

    zparams = numpy.load('search_model.npz')
    dparams = {}

    for item in zparams:
        dparams[item] = zparams[item]

    for item in mapping:
        index = mapping[item]
        model.parameter[index].set_value(dparams[item])
        fmodel.parameter[index].set_value(dparams[item])

    encode, compute_prob, compute_state = model.sample
    fencode, fcompute_istate, fcompute_prob, fcompute_state = fmodel.sample

    anno, manno, istate = encode(seq.astype('int32')[:, None])
    prob = compute_prob(numpy.array([0]).astype('int32'), anno, manno, istate)
    newstate = compute_state(numpy.array([2440]).astype('int32'), anno, manno, istate)

    fanno = fencode(seq.astype('int32')[:, None])
    fistate = fcompute_istate(fanno)
    fprob = fcompute_prob(numpy.array([0]).astype('int32'), fanno, fistate)
    fnewstate = fcompute_state(numpy.array([2440]).astype('int32'), fanno, fistate)

    train_data = get_batch_iterator(state)
    train_data.start(1)
    data = train_data.next()

    x = data['x']
    xmask = data['x_mask']
    y = data['y']
    ymask = data['y_mask']
    cost = train_lm(y, x, xmask, ymask)
    train = theano.function(model.input, model.output, on_unused_input = 'ignore')
    cost2 = train(x.astype('int32'), xmask, y.astype('int32'), ymask)

    train1 = theano.function(fmodel.input, fmodel.output, on_unused_input = 'ignore')
    cost1 = train1(x.astype('int32'), xmask, y.astype('int32'), ymask)

    import sys
    sys.exit()

    test_func = enc_dec.create_test_model()

    c = comp_repr(x[:, 126])[0]
    istates = comp_init_states(c)
    istates = map(lambda x : x[None, :], istates)
    p = comp_next_probs(c, 1, numpy.array([0]).astype('int64'), *istates)
    new_states = comp_next_states(c, 1, y[0, 126:127], *istates)
    p = comp_next_probs(c, 1, y[0, 126:127], *new_states)

    anno, manno, istate = encode(x[:, 126].astype('int32')[:, None])
    prob = compute_prob(numpy.array([0]).astype('int32'), anno, manno, istate)
    newstate = compute_state(y[0, 126:127].astype('int32'), anno, manno, istate)
    prob = compute_prob(y[0, 126:127].astype('int32'), anno, manno, newstate)

    align = test_func(x, y, xmask, ymask)
