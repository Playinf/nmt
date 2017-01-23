#!/usr/bin/python
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import ops
import sys
import math
import time
import numpy
import cPickle
import argparse

from metric import bleu
from optimizer import optimizer
from data import textreader, textiterator
from data.plain import convert_data, data_length
from model.rnnsearch import rnnsearch, beamsearch


def load_vocab(file):
    fd = open(file, "r")
    vocab = cPickle.load(fd)
    fd.close()
    return vocab


def invert_vocab(vocab):
    v = {}
    for k, idx in vocab.iteritems():
        v[idx] = k

    return v


def count_parameters(variables):
    n = 0

    for item in variables:
        v = item.get_value()
        n += v.size

    return n


def serialize(name, option):
    fd = open(name, "w")
    params = ops.trainable_variables()
    names = [p.name for p in params]
    vals = dict([(p.name, p.get_value()) for p in params])

    if option["indices"] != None:
        indices = option["indices"]
        vals["indices"] = indices
        option["indices"] = None
    else:
        indices = None

    cPickle.dump(option, fd)
    cPickle.dump(names, fd)
    # compress
    numpy.savez(fd, **vals)

    # restore
    if indices is not None:
        option["indices"] = indices

    fd.close()


# load model from file
def load_model(name):
    fd = open(name, "r")
    option = cPickle.load(fd)
    names = cPickle.load(fd)
    vals = dict(numpy.load(fd))

    params = [(n, vals[n]) for n in names]

    if "indices" in vals:
        option["indices"] = vals["indices"]

    fd.close()

    return option, params


def match_variables(variables, values, ignore_prefix=True):
    var_dict = {}
    val_dict = {}
    matched = []
    not_matched = []

    for var in variables:
        if ignore_prefix:
            name = "/".join(var.name.split("/")[1:])
        var_dict[name] = var

    for (name, val) in values:
        if ignore_prefix:
            name = "/".join(name.split("/")[1:])
        val_dict[name] = val

    # matching
    for name in var_dict:
        var = var_dict[name]

        if name in val_dict:
            val = val_dict[name]
            matched.append([var, val])
        else:
            not_matched.append(var)

    return matched, not_matched


def restore_variables(matched, not_matched):
    for var, val in matched:
        var.set_value(val)

    for var in not_matched:
        sys.stderr.write("%s NOT restored\n" % var.name)


def set_variables(variables, values):
    values = [item[1] for item in values]

    for p, v in zip(variables, values):
        p.set_value(v)


def load_references(names, case=True):
    references = []
    reader = textreader(names)
    stream = textiterator(reader, size=[1, 1])

    for data in stream:
        newdata= []
        for batch in data:
            line = batch[0]
            words = line.strip().split()
            if not case:
                lower = [word.lower() for word in words]
                newdata.append(lower)
            else:
                newdata.append(words)

        references.append(newdata)

    stream.close()

    return references


def translate(model, corpus, **opt):
    fd = open(corpus, "r")
    svocab = model.option["vocabulary"][0][0]
    unk_symbol = model.option["unk"]
    eos_symbol = model.option["eos"]

    trans = []

    for line in fd:
        line = line.strip()
        data, mask = convert_data([line], svocab, unk_symbol, eos_symbol)
        hypo_list = beamsearch(model, data, **opt)
        if len(hypo_list) > 0:
            best, score = hypo_list[0]
            trans.append(best[:-1])
        else:
            trans.append([])

    fd.close()

    return trans


def parseargs_train(args):
    msg = "training rnnsearch"
    usage = "rnnsearch.py train [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    # corpus and vocabulary
    msg = "source and target corpus"
    parser.add_argument("--corpus", nargs=2, help=msg)
    msg = "source and target vocabulary"
    parser.add_argument("--vocab", nargs=2, help=msg)
    msg = "model name to save or saved model to initialize, required"
    parser.add_argument("--model", required=True, help=msg)

    # model parameters
    msg = "source and target embedding size, default 620"
    parser.add_argument("--embdim", nargs=2, type=int, help=msg)
    msg = "source, target and alignment hidden size, default 1000"
    parser.add_argument("--hidden", nargs=3, type=int, help=msg)
    msg = "maxout hidden dimension, default 500"
    parser.add_argument("--maxhid", type=int, help=msg)
    msg = "maxout number, default 2"
    parser.add_argument("--maxpart", type=int, help=msg)
    msg = "deepout hidden dimension, default 620"
    parser.add_argument("--deephid", type=int, help=msg)
    msg = "maximum training epoch, default 5"
    parser.add_argument("--maxepoch", type=int, help=msg)

    # tuning options
    msg = "learning rate, default 5e-4"
    parser.add_argument("--alpha", type=float, help=msg)
    msg = "momentum, default 0.0"
    parser.add_argument("--momentum", type=float, help=msg)
    msg = "batch size, default 128"
    parser.add_argument("--batch", type=int, help=msg)
    msg = "optimizer, default rmsprop"
    parser.add_argument("--optimizer", type=str, help=msg)
    msg = "gradient clipping, default 1.0"
    parser.add_argument("--norm", type=float, help=msg)
    msg = "early stopping iteration, default 0"
    parser.add_argument("--stop", type=int, help=msg)
    msg = "decay factor, default 0.5"
    parser.add_argument("--decay", type=float, help=msg)
    msg = "initialization scale, default 0.08"
    parser.add_argument("--scale", type=float, help=msg)
    msg = "L1 regularizer scale"
    parser.add_argument("--l1-scale", type=float, help=msg)
    msg = "L2 regularizer scale"
    parser.add_argument("--l2-scale", type=float, help=msg)
    msg = "dropout keep probability"
    parser.add_argument("--keep-prob", type=float, help=msg)

    # validation
    msg = "random seed, default 1234"
    parser.add_argument("--seed", type=int, help=msg)
    msg = "validation dataset"
    parser.add_argument("--validation", type=str, help=msg)
    msg = "reference data"
    parser.add_argument("--references", type=str, nargs="+", help=msg)

    # data processing
    msg = "sort batches"
    parser.add_argument("--sort", type=int, help=msg)
    msg = "shuffle every epcoh"
    parser.add_argument("--shuffle", type=int, help=msg)
    msg = "source and target sentence limit, default 50 (both), 0 to disable"
    parser.add_argument("--limit", type=int, nargs='+', help=msg)


    # control frequency
    msg = "save frequency, default 1000"
    parser.add_argument("--freq", type=int, help=msg)
    msg = "sample frequency, default 50"
    parser.add_argument("--sfreq", type=int, help=msg)
    msg = "validation frequency, default 1000"
    parser.add_argument("--vfreq", type=int, help=msg)

    # control beamsearch
    msg = "beam size, default 10"
    parser.add_argument("--beamsize", type=int, help=msg)
    msg = "normalize probability by the length of candidate sentences"
    parser.add_argument("--normalize", type=int, help=msg)
    msg = "max translation length"
    parser.add_argument("--maxlen", type=int, help=msg)
    msg = "min translation length"
    parser.add_argument("--minlen", type=int, help=msg)

    msg = "initialize from another model"
    parser.add_argument("--initialize", type=str, help=msg)
    msg = "fine tune model"
    parser.add_argument("--finetune", action="store_true", help=msg)
    msg = "reset count"
    parser.add_argument("--reset", action="store_true", help=msg)
    msg = "disable validation"
    parser.add_argument("--no-validation", action="store_true", help=msg)

    return parser.parse_args(args)


def parseargs_decode(args):
    msg = "translate using exsiting nmt model"
    usage = "rnnsearch.py translate [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "trained model"
    parser.add_argument("--model", type=str, required=True, help=msg)
    msg = "beam size"
    parser.add_argument("--beamsize", default=10, type=int, help=msg)
    msg = "normalize probability by the length of candidate sentences"
    parser.add_argument("--normalize", action="store_true", help=msg)
    msg = "max translation length"
    parser.add_argument("--maxlen", type=int, help=msg)
    msg = "min translation length"
    parser.add_argument("--minlen", type=int, help=msg)

    return parser.parse_args(args)


def default_option():
    option = {}

    # training corpus and vocabulary
    option["corpus"] = None
    option["vocab"] = None

    # model parameters
    option["embdim"] = [620, 620]
    option["hidden"] = [1000, 1000, 1000]
    option["maxpart"] = 2
    option["maxhid"] = 500
    option["deephid"] = 620

    # tuning options
    option["alpha"] = 5e-4
    option["batch"] = 128
    option["momentum"] = 0.0
    option["optimizer"] = "rmsprop"
    option["norm"] = 1.0
    option["stop"] = 0
    option["decay"] = 0.5
    option["scale"] = 0.08
    option["l1_scale"] = None
    option["l2_scale"] = None
    option["keep_prob"] = None

    # runtime information
    option["cost"] = 0.0
    # batch/reader count
    option["count"] = [0, 0]
    option["epoch"] = 0
    option["maxepoch"] = 5
    option["sort"] = 20
    option["shuffle"] = False
    option["limit"] = [50, 50]
    option["freq"] = 1000
    option["vfreq"] = 1000
    option["sfreq"] = 50
    option["seed"] = 1234
    option["validation"] = None
    option["references"] = None
    option["bleu"] = 0.0
    option["indices"] = None

    # beam search
    option["beamsize"] = 10
    option["normalize"] = False
    option["maxlen"] = None
    option["minlen"] = None

    # special symbols
    option["unk"] = "UNK"
    option["eos"] = "<eos>"

    return option


def args_to_dict(args):
    return args.__dict__


def override_if_not_none(opt1, opt2, key):
    if key in opt2:
        value = opt2[key]
    else:
        value = None

    opt1[key] = value if value != None else opt1[key]


# override default options
def override(option, args):

    # training corpus
    if args["corpus"] == None and option["corpus"] == None:
        raise ValueError("error: no training corpus specified")

    # vocabulary
    if args["vocab"] == None and option["vocab"] == None:
        raise ValueError("error: no training vocabulary specified")

    if args["limit"] and len(args["limit"]) > 2:
        raise ValueError("error: invalid number of --limit argument (<=2)")

    if args["limit"] and len(args["limit"]) == 1:
        args["limit"] = args["limit"] * 2

    override_if_not_none(option, args, "corpus")

    # vocabulary and model paramters cannot be overrided
    if option["vocab"] == None:
        option["vocab"] = args["vocab"]
        svocab = load_vocab(args["vocab"][0])
        tvocab = load_vocab(args["vocab"][1])
        isvocab = invert_vocab(svocab)
        itvocab = invert_vocab(tvocab)

        # append a new symbol "<eos>" to vocabulary, it is not necessary
        # because we can reuse "</s>" symbol in vocabulary
        # but here we retain compatibility with GroundHog
        svocab[option["eos"]] = len(isvocab)
        tvocab[option["eos"]] = len(itvocab)
        isvocab[len(isvocab)] = option["eos"]
        itvocab[len(itvocab)] = option["eos"]

        # <s> and </s> have the same id 0, used for decoding (target side)
        option["bosid"] = 0
        option["eosid"] = len(itvocab) - 1

        option["vocabulary"] = [[svocab, isvocab], [tvocab, itvocab]]

        # model parameters
        override_if_not_none(option, args, "embdim")
        override_if_not_none(option, args, "hidden")
        override_if_not_none(option, args, "maxhid")
        override_if_not_none(option, args, "maxpart")
        override_if_not_none(option, args, "deephid")

    # training options
    override_if_not_none(option, args, "maxepoch")
    override_if_not_none(option, args, "alpha")
    override_if_not_none(option, args, "momentum")
    override_if_not_none(option, args, "batch")
    override_if_not_none(option, args, "optimizer")
    override_if_not_none(option, args, "norm")
    override_if_not_none(option, args, "stop")
    override_if_not_none(option, args, "decay")
    override_if_not_none(option, args, "scale")
    override_if_not_none(option, args, "l1_scale")
    override_if_not_none(option, args, "l2_scale")
    override_if_not_none(option, args, "keep_prob")

    # runtime information
    override_if_not_none(option, args, "cost")
    override_if_not_none(option, args, "count")
    override_if_not_none(option, args, "epoch")
    override_if_not_none(option, args, "maxepoch")
    override_if_not_none(option, args, "sort")
    override_if_not_none(option, args, "shuffle")
    override_if_not_none(option, args, "limit")
    override_if_not_none(option, args, "freq")
    override_if_not_none(option, args, "vfreq")
    override_if_not_none(option, args, "sfreq")
    override_if_not_none(option, args, "seed")
    override_if_not_none(option, args, "validation")
    override_if_not_none(option, args, "references")
    override_if_not_none(option, args, "bleu")
    override_if_not_none(option, args, "indices")

    # beamsearch
    override_if_not_none(option, args, "beamsize")
    override_if_not_none(option, args, "normalize")
    override_if_not_none(option, args, "maxlen")
    override_if_not_none(option, args, "minlen")


def print_option(option):
    isvocab = option["vocabulary"][0][1]
    itvocab = option["vocabulary"][1][1]

    print ""
    print "options"

    print "corpus:", option["corpus"]
    print "vocab:", option["vocab"]
    print "vocabsize:", [len(isvocab), len(itvocab)]

    print "embdim:", option["embdim"]
    print "hidden:", option["hidden"]
    print "maxhid:", option["maxhid"]
    print "maxpart:", option["maxpart"]
    print "deephid:", option["deephid"]

    print "maxepoch:", option["maxepoch"]
    print "alpha:", option["alpha"]
    print "momentum:", option["momentum"]
    print "batch:", option["batch"]
    print "optimizer:", option["optimizer"]
    print "norm:", option["norm"]
    print "stop:", option["stop"]
    print "decay:", option["decay"]
    print "scale:", option["scale"]
    print "L1-scale:", option["l1_scale"]
    print "L2-scale:", option["l2_scale"]
    print "keep-prob:", option["keep_prob"]

    print "validation:", option["validation"]
    print "references:", option["references"]
    print "freq:", option["freq"]
    print "vfreq:", option["vfreq"]
    print "sfreq:", option["sfreq"]
    print "seed:", option["seed"]
    print "sort:", option["sort"]
    print "shuffle:", option["shuffle"]
    print "limit:", option["limit"]

    print "beamsize:", option["beamsize"]
    print "normalize:", option["normalize"]
    print "maxlen:", option["maxlen"]
    print "minlen:", option["minlen"]

    # special symbols
    print "unk:", option["unk"]
    print "eos:", option["eos"]


def skip_stream(stream, count):
    for i in range(count):
        stream.next()


def get_filename(name):
    s = name.split(".")
    return s[0]


def train(args):
    option = default_option()

    # predefined model names
    pathname, basename = os.path.split(args.model)
    modelname = get_filename(basename)
    autoname = os.path.join(pathname, modelname + ".autosave.pkl")
    bestname = os.path.join(pathname, modelname + ".best.pkl")

    # load models
    if os.path.exists(args.model):
        opt, params = load_model(args.model)
        override(option, opt)
        init = False
    else:
        init = True

    if args.initialize:
        init_params = load_model(args.initialize)
        init_params = init_params[1]
        restore = True
    else:
        restore = False

    override(option, args_to_dict(args))
    print_option(option)

    # load references
    if option["references"]:
        references = load_references(option["references"])
    else:
        references = None

    if args.no_validation:
        references = None

    # input corpus
    batch = option["batch"]
    sortk = option["sort"] or 1
    shuffle = option["seed"] if option["shuffle"] else None
    reader = textreader(option["corpus"], shuffle)
    processor = [data_length, data_length]
    stream = textiterator(reader, [batch, batch * sortk], processor,
                          option["limit"], option["sort"])

    if shuffle and option["indices"] is not None:
        reader.set_indices(option["indices"])

    if args.reset:
        option["count"] = [0, 0]
        option["epoch"] = 0
        option["cost"] = 0.0

    skip_stream(reader, option["count"][1])
    epoch = option["epoch"]
    maxepoch = option["maxepoch"]

    # create model
    regularizer = []

    if option["l1_scale"]:
        regularizer.append(ops.l1_regularizer(option["l1_scale"]))

    if option["l2_scale"]:
        regularizer.append(ops.l2_regularizer(option["l2_scale"]))

    scale = option["scale"]
    initializer = ops.random_uniform_initializer(-scale, scale)
    regularizer = ops.sum_regularizer(regularizer)
    # set seed
    numpy.random.seed(option["seed"])
    model = rnnsearch(initializer=initializer, regularizer=regularizer,
                      **option)

    variables = None

    if restore:
        matched, not_matched = match_variables(ops.trainable_variables(),
                                               init_params)
        if args.finetune:
            variables = not_matched
            if not variables:
                raise RuntimeError("no variables to finetune")

    if not init:
        set_variables(ops.trainable_variables(), params)

    if restore:
        restore_variables(matched, not_matched)

    print "parameters:", count_parameters(ops.trainable_variables())

    # tuning option
    tune_opt = {}
    tune_opt["algorithm"] = option["optimizer"]
    tune_opt["constraint"] = ("norm", option["norm"])
    tune_opt["norm"] = True
    tune_opt["variables"] = variables

    # create optimizer
    trainer = optimizer(model, **tune_opt)

    # beamsearch option
    search_opt = {}
    search_opt["beamsize"] = option["beamsize"]
    search_opt["normalize"] = option["normalize"]
    search_opt["maxlen"] = option["maxlen"]
    search_opt["minlen"] = option["minlen"]

    # vocabulary and special symbol
    svocabs, tvocabs = option["vocabulary"]
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs
    unk_sym = option["unk"]
    eos_sym = option["eos"]

    # summary
    count = option["count"][0]
    totcost = option["cost"]
    best_score = option["bleu"]
    alpha = option["alpha"]

    for i in range(epoch, maxepoch):
        for data in stream:
            xdata, xmask = convert_data(data[0], svocab, unk_sym, eos_sym)
            ydata, ymask = convert_data(data[1], tvocab, unk_sym, eos_sym)

            t1 = time.time()
            cost, norm = trainer.optimize(xdata, xmask, ydata, ymask)
            trainer.update(alpha = alpha)
            t2 = time.time()

            count += 1

            cost = cost * ymask.shape[1] / ymask.sum()
            totcost += cost / math.log(2)
            print i + 1, count, cost, norm, t2 - t1

            # autosave
            if count % option["freq"] == 0:
                option["indices"] = reader.get_indices()
                option["bleu"] = best_score
                option["cost"] = totcost
                option["count"] = [count, reader.count]
                serialize(autoname, option)

            if count % option["vfreq"] == 0:
                if option["validation"] and references:
                    trans = translate(model, option["validation"],
                                      **search_opt)
                    bleu_score = bleu(trans, references)
                    print "bleu: %2.4f" % bleu_score
                    if bleu_score > best_score:
                        best_score = bleu_score
                        option["indices"] = reader.get_indices()
                        option["bleu"] = best_score
                        option["cost"] = totcost
                        option["count"] = [count, reader.count]
                        serialize(bestname, option)

            if count % option["sfreq"] == 0:
                n = len(data[0])
                ind = numpy.random.randint(0, n)
                sdata = data[0][ind]
                tdata = data[1][ind]
                xdata = xdata[:, ind : ind + 1]
                xmask = xmask[:, ind : ind + 1]
                hls = beamsearch(model, xdata, xmask)
                best, score = hls[0]
                print sdata
                print tdata
                print " ".join(best[:-1])

        print "--------------------------------------------------"

        if option["validation"] and references:
            trans = translate(model, option["validation"], **search_opt)
            bleu_score = bleu(trans, references)
            print "iter: %d, bleu: %2.4f" % (i + 1, bleu_score)
            if bleu_score > best_score:
                best_score = bleu_score
                option["indices"] = reader.get_indices()
                option["bleu"] = best_score
                option["cost"] = totcost
                option["count"] = [count, reader.count]
                serialize(bestname, option)

        print "averaged cost: ", totcost / count
        print "--------------------------------------------------"

        # early stopping
        if i + 1 >= option["stop"]:
            alpha = alpha * option["decay"]

        count = 0
        totcost = 0.0
        stream.reset()

        # update autosave
        option["epoch"] = i + 1
        option["alpha"] = alpha
        option["indices"] = reader.get_indices()
        option["bleu"] = best_score
        option["cost"] = totcost
        option["count"] = [0, 0]
        serialize(autoname, option)

    print "best(bleu): %2.4f" % best_score

    stream.close()


def decode(args):
    option, params = load_model(args.model)
    model = rnnsearch(**option)
    set_variables(ops.trainable_variables(), params)

    # use the first model
    svocabs, tvocabs = model.option["vocabulary"]
    unk_sym = model.option["unk"]
    eos_sym = model.option["eos"]

    count = 0

    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    option = {}
    option["maxlen"] = args.maxlen
    option["minlen"] = args.minlen
    option["beamsize"] = args.beamsize
    option["normalize"] = args.normalize

    while True:
        line = sys.stdin.readline()

        if line == "":
            break

        data = [line]
        seq, mask = convert_data(data, svocab, unk_sym, eos_sym)
        t1 = time.time()
        tlist = beamsearch(model, seq, **option)
        t2 = time.time()

        if len(tlist) == 0:
            translation = ""
            score = -10000.0
        else:
            best, score = tlist[0]
            translation = " ".join(best[:-1])

        sys.stdout.write(translation)
        sys.stdout.write("\n")

        count = count + 1
        sys.stderr.write(str(count) + " ")
        sys.stderr.write(str(score) + " " + str(t2 - t1) + "\n")


def helpinfo():
    print "usage:"
    print "\trnnsearch.py <command> [<args>]"
    print "use 'rnnsearch.py train --help' to see training options"
    print "use 'rnnsearch.py translate' --help to see decoding options"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        helpinfo()
    else:
        command = sys.argv[1]
        if command == "train":
            print "training command:"
            print " ".join(sys.argv)
            args = parseargs_train(sys.argv[2:])
            train(args)
        elif command == "translate":
            sys.stderr.write(" ".join(sys.argv))
            sys.stderr.write("\n")
            args = parseargs_decode(sys.argv[2:])
            decode(args)
        else:
            helpinfo()
