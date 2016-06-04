# highway.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import config

from feedforward import feedforward
from utils import update_option, add_if_not_exsit
from utils import add_parameters, extract_option

# feedforward neural network
# y = f(Wx + b) or y = f(xW + b)
# input_size: dimension of x
# output_size: dimension of y
# available options:
# 1. name: str, default 'feedforward'
# 2. bias: boolean, True to use bias, False to not use bias
# 3. weight: boolean, True stands for Wx, False stands for xW
# 4. function: activation function, default: theano.tensor.nnet.sigmoid
# 5. target: target device, default theano.config.device
class highwaynet:

    def __init__(self, input_size, output_size, **option):
        # inherit option
        if 'target' in option:
            add_if_not_exsit(option, 'carray-gate/target', option['target'])
            add_if_not_exsit(option, 'transform-gate/target', option['target'])
            add_if_not_exsit(option, 'transform/target', option['target'])

        if 'variant' in option:
            add_if_not_exsit(option, 'carry-gate/variant', option['variant'])
            add_if_not_exsit(option, 'transform-gate/variant', option['variant'])
            add_if_not_exsit(option, 'transform/variant', option['variant'])

        opt = config.gru_option()
        update_option(opt, option)

        copt = extract_option(opt, 'carry-gate')
        fopt = extract_option(opt, 'transform-gate')
        topt = extract_option(opt, 'transform')
        copt['name'] = 'carry-gate'
        fopt['name'] = 'transform-gate'
        topt['name'] = 'transform'
        topt['function'] = opt['transform']

        modules = []

        if not isinstance(input_size, (list, tuple)):
            input_size = [input_size]

        cgate = feedforward(input_size, output_size, **copt)
        tgate = feedforward(input_size, output_size, **fopt)
        trans = feedforward(input_size, output_size, **topt)
        modules.append(cgate)
        modules.append(tgate)
        modules.append(trans)

        name = opt['name']
        params = []

        for m in modules:
            add_parameters(params, name, *m.parameter)

        def forward(x):
            if not isinstance(x, (list, tuple)):
                x = [x]

            h = trans(x)
            t = tgate(x)
            c = cgate(x)

            return h * t + x * (1 - t)

        self.name = name
        self.option = opt
        self.forward = forward
        self.parameter = params

    def __call__(self, x, h):
        return self.forward(x, h)
