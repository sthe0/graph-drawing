# -*- coding:utf-8 -*-
class RSQTree(object):
    def __init__(self, seq_sum):
        tsum = []
        tsum.extend(seq_sum)
        self._sum_buf = [None] * 4 * len(seq_sum)
        self._n = len(seq_sum)
        self._build_sum(tsum, 1, 0, self._n - 1)

    def _build_sum(self, tsum, v, tl, tr):
        try:
            if tl == tr:
                self._sum_buf[v] = tsum[tl]
            else:
                tm = (tl + tr) // 2
                self._build_sum(tsum, 2 * v, tl, tm)
                self._build_sum(tsum, 2 * v + 1, tm + 1, tr)
                self._sum_buf[v] = sum([self._sum_buf[2 * v],
                                        self._sum_buf[2 * v + 1]])
        except:
            raise Exception("{0} {1}".format(tl, tr))

    def _rsum(self, v, tl, tr, l, r):
        if l > r:
            return 0
        if l == tl and r == tr:
            return self._sum_buf[v]
        tm = (tl + tr) // 2
        item = sum([self._rsum(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r),
                    self._rsum(v * 2, tl, tm, l, min(r, tm))])
        return item

    def get_sum(self, l, r):
        return self._rsum(1, 0, self._n - 1, l, r)
