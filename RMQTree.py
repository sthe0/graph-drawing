# -*- coding:utf-8 -*-
class RMQTree(object):
    def __init__(self, seq_max):
        tmax = []
        tmax.extend(seq_max)
        self._max_buf = [None] * 4 * len(seq_max)
        self._n = len(seq_max)
        self._build_max(tmax, 1, 0, self._n - 1)

    def _build_max(self, tmax, v, tl, tr):
        if tl == tr:
            self._max_buf[v] = tmax[tl]
        else:
            tm = (tl + tr) // 2
            self._build_max(tmax, 2*v, tl, tm)
            self._build_max(tmax, 2*v + 1, tm + 1, tr)
            self._max_buf[v] = max(self._max_buf[2*v],
                                     self._max_buf[2*v + 1])

    def _rmax(self, v, tl, tr, l, r):
        if l == tl and r == tr:
            return self._max_buf[v]
        tm = (tl + tr) // 2
        if min(r, tm) < l:
            item = self._rmax(v*2 + 1, tm + 1, tr, max(l, tm + 1), r)
        elif max(l, tm + 1) > r:
            item = self._rmax(v*2, tl, tm, l, min(r, tm))
        else:
            item = max(self._rmax(v*2 + 1, tm + 1, tr, max(l, tm + 1), r),
                       self._rmax(v*2, tl, tm, l, min(r, tm)))
        return item

    def get_max(self, l, r):
        return self._rmax(1, 0, self._n - 1, l, r)
