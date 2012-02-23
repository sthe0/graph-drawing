#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Digraph


class RelationEdge(Digraph.Edge):
    def __init__(self, x, y, lt=False, le=False, gt=False, ge=False):
        super(RelationEdge, self).__init__(x, y)
        self.__lt = lt
        self.__le = le
        self.__gt = gt
        self.__ge = ge

    def get_ge(self):
        return self.__ge
    
    def set_ge(self, value):
        self.__ge = value

    def get_gt(self):
        return self.__gt

    def set_gt(self, value):
        self.__gt = value

    def get_le(self):
        return self.__le

    def set_le(self, value):
        self.__le = value

    def get_lt(self):
        return self.__lt

    def set_lt(self, value):
        self.__lt = value

    ge = property(get_ge, set_ge)
    gt = property(get_gt, set_gt)
    le = property(get_le, set_le)
    lt = property(get_lt, set_lt)


class InclusionVertex(Digraph.Vertex):
    def __init__(self, level):
        super(InclusionVertex, self).__init__()
        self.__level = level

    def get_level(self):
        return self.__level

    level = property(get_level)


class CompoundLayer(list):
    def __cmp__(self, other):
        for i in range(min(len(self), len(other))):
            if self[i] < other[i]:
                return -1
            elif self[i] > other[i]:
                return 1
        return len(other) - len(self)


class CompoundGraphDrawer(object):
    def __init__(self):
        self.__compound_layers = []
        self.__relation_graph = Digraph.Digraph()

    def __topological_sort(self, graph):
        pass

    def __get_top_level(self, graph):
        pass

    def __assign_compound_layers_to_level(self, vertex_list):
        pass

    def __assign_compound_layers(self, graph):
        self.__relation_graph = self.__topological_sort(graph)
        self.__assign_compound_layers_to_level(self.__get_top_level(self.__relation_graph))

    def draw(self, graph):
        self.__assign_compound_layers(graph)
