# nmt.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

from rnnsearch import rnnsearch

# wrapper for various nmt models
class nmt:

    def __init__(self, **option):

        if option['model'] == 'rnnsearch':
            model = rnnsearch(**option)

        self.model = model
        self.inputs = model.inputs
        self.outputs = model.outputs
        self.option = model.option
        self.gradient = model.gradient
        self.sample = model.sample
        self.compute = model.compute
        self.parameter = model.parameter
        self.vocabulary = model.vocabulary
