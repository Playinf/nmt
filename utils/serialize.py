# serialize.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

# write model into file
def serialize(name, model):
    fd = open(name, 'w')
    option = model.option
    params = model.parameter
    pval = [item.get_value() for item in params]
    cPickle.dump(option, fd)
    cPickle.dump(pval, fd)
    fd.close()

# load model from file
def loadmodel(name):
    fd = open(name, 'r')
    option = cPickle.load(fd)
    params = cPickle.load(fd)
    model = rnnsearch(**option)

    for val, param in zip(params, model.parameter):
        param.set_value(val)

    fd.close()

    return model
