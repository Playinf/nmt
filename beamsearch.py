# beamsearch.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import heapq

def hash_combine(v1, v2):
    val = 0
    val ^= v2 + 0x9e3779b9 + (v1 << 6) + (v1 >> 2)

    return val

class hypothesis:

    def __init__(self):
        self.state = None
        self.score = float('-Inf')
        self.translation = []

    def hashcode(self):
        hashval = 0

        for item in self.translation:
            hashval = hash_combine(hashval, hash(item))

        return hashval

    def equals_to(self, hypo):
        t1 = self.translation
        t2 = hypo.translation

        if len(t1) != len(t2):
            return False

        for item1, item2 in zip(t1, t2):
            if item1 != item2:
                return False

        return True

# implement a beam
class beam:

    def __init__(self, size, threshold):

        self.items = {}
        self.size = 0
        self.score = float('-Inf')
        self.histogram = size
        self.threshold = threshold
        self.parameter = [size, threshold]

    def sort(self):
        if self.size > self.histogram:
            self.prune()

        hypo_list = []

        for item in self.items.itervalues():
            for hypo in item:
                hypo_list.append(hypo)

        hypo_list = sorted(hypo_list, key = lambda x: -x.score)

        self.ordereditems = hypo_list

        return hypo_list

    def insert(self, hypo):
        # apply threshold pruning
        if hypo.score < self.score + self.threshold:
            return

        if hypo.score > self.score:
            self.score = hypo.score

        code = hypo.hashcode()

        if code not in self.items:
            # insert into beam
            self.items[code] = [hypo]
        else:
            recombine = False
            hypolist = self.items[code]
            for i in range(len(hypolist)):
                h = hypolist[i]
                if hypo.equals_to(h):
                    hypolist[i] = hypo if hypo.score > h.score else h
                    recombine = True
                    break
                if not recombine:
                    hypolist.append(hypo)

        self.size += 1

        if self.size >= 2 * self.histogram:
            self.prune()

    def prune(self):
        queue = []

        for item in self.items.itervalues():
            for hypo in item:
                heapq.heappush(queue, -hypo.score)

        # prune
        for i in xrange(self.histogram):
            heapq.heappop(queue)

        minscore = -queue[0]
        last_found = False
        items = self.items
        self.items = {}
        self.size = 0

        for item in items.itervalues():
            for hypo in item:
                score = hypo.score

                # prune
                if score < minscore:
                    continue

                if score == minscore:
                    if last_found:
                        continue
                    last_found = True

                self.insert(hypo)
