# -*- coding:utf-8 -*-
class RMQTree(object):
    def __init__(self, seq):
        t = []
        t.extend(seq)
        self._container = [None] * 4 * len(seq)
        self._indexes = {}
        self._n = len(seq)
        self._build(t, 1, 0, self._n - 1)
        for n, item in enumerate(seq):
            self._indexes[item] = n


    def _build(self, t, v, tl, tr):
        if tl == tr:
            self._container[v] = t[tl]
        else:
            tm = (tl + tr) // 2
            self._build(t, 2*v, tl, tm)
            self._build(t, 2*v + 1, tm + 1, tr)
            self._container[v] = max(self._container[2*v],
                                     self._container[2*v + 1])

    def _rmax(self, v, tl, tr, l, r):
        if l == tl and r == tr:
            return self._container[v]
        tm = (tl + tr) // 2
        if min(r, tm) < l:
            item = self._rmax(v*2 + 1, tm + 1, tr, max(l, tm + 1), r)
        elif max(l, tm + 1) > r:
            item = self._rmax(v*2, tl, tm, l, min(r, tm))
        else:
            item = max(self._rmax(v*2 + 1, tm + 1, tr, max(l, tm + 1), r),
                       self._rmax(v*2, tl, tm, l, min(r, tm)))
        return item

    def rmax(self, l, r):
        item = self._rmax(1, 0, self._n - 1, l, r)
        index = min(max(self._indexes[item], l), r)
        return item, index
