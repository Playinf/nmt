import theano

from nn import linear

from model.rnnsearch import rnnsearch

option = {}
option['embdim'] = [620, 620]
option['hidden'] = [1000, 1000, 1000]
option['maxhid'] = 500
option['deephid'] = 620
option['maxpart'] = 2
voc = [None] * 16001
option['vocabulary'] = [[voc, voc], [voc, voc]]

model = rnnsearch(**option)
for i, param in enumerate(model.parameter):
    print i, param.name, param.get_value(borrow = True).shape

pmap = []
# source embedding
pmap.append([0, 0])
pmap.append([1, 1])
# target embedding
pmap.append([2, 16])
pmap.append([3, 17])
# forward-encoder
pmap.append([4, 5])
pmap.append([5, 6])
pmap.append([6, 7])
pmap.append([7, 8])
pmap.append([8, 2])
pmap.append([9, 3])
pmap.append([10, 4])
# backward-encoder
pmap.append([11, 12])
pmap.append([12, 13])
pmap.append([13, 14])
pmap.append([14, 15])
pmap.append([15, 9])
pmap.append([16, 10])
pmap.append([17, 11])
# init-transform
pmap.append([18, 18])
pmap.append([19, 19])
# annot-transform
pmap.append([20, 20])
# state-transform
pmap.append([21, 21])
# context-transform
pmap.append([22, 22])
# decoding gru
pmap.append([23, 27])
pmap.append([24, 28])
pmap.append([25, 29])
pmap.append([26, 30])
pmap.append([27, 31])
pmap.append([28, 32])
pmap.append([29, 23])
pmap.append([30, 24])
pmap.append([31, 25])
pmap.append([32, 26])
# maxout
pmap.append([33, 33])
pmap.append([34, 34])
pmap.append([35, 35])
pmap.append([36, 36])
# deepout
pmap.append([37, 37])
# classify
pmap.append([38, 38])
pmap.append([39, 39])

#theano.printing.debugprint(model.outputs[0])
'''
x = theano.tensor.matrix()
h = theano.tensor.matrix()
m = linear([100, 100], 100)
y = m([x, h])
theano.printing.debugprint(y)
'''
