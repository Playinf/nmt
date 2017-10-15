# blind_prune.py

import sys
import numpy
import cPickle


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


def serialize(name, option, params):
    fd = open(name, "w")
    names = [p[0] for p in params]
    vals = dict([(p[0], p[1]) for p in params])

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


def prune(tensor, percent):
    mask = numpy.zeros_like(tensor)
    abs_tensor = numpy.absolute(tensor)
    abs_array = abs_tensor.reshape(-1)
    indices = numpy.argsort(abs_array)
    keep_ind = int(numpy.floor(abs_array.shape[0] * percent))
    keep_indices = indices[keep_ind:]

    for ind in keep_indices:
        ind = numpy.unravel_index(ind, tensor.shape)
        mask[ind] = 1.0

    return mask


def combine(tensors):
    reshaped = []

    for t in tensors:
        reshaped.append(t.reshape(-1))

    return numpy.concatenate(reshaped)


def split(combined, tensors):
    splits = []
    offset = 0

    for t in tensors:
        splits.append(combined[offset : offset + t.size].reshape(t.shape))
        offset += t.size

    return splits


def main():
    option, params = load_model(sys.argv[1])
    masks = {}
    combined = combine([p[1] for p in params])
    combined_mask = prune(combined, float(sys.argv[3]))
    mask_list = split(combined_mask, [p[1] for p in params])

    for i, (name, param) in enumerate(params):
        masks[name] = mask_list[i]
        params[i] = (name, param * mask_list[i])

    option["mask"] = masks
    serialize(sys.argv[2], option, params)


main()
