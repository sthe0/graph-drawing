#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict
import copy
from collections import namedtuple

class ObjectWithId(object):
    def __init__(self, object_id=None):
        if object_id is None:
            object_id = id(self)
        self.__id = object_id

    def get_id(self):
        return self.__id

    def __hash__(self):
        return self.__id

    def __eq__(self, other):
        return self.__id == other.__id

    def __repr__(self):
        return str(self.__id)

    id = property(get_id)


class Vertex(ObjectWithId):
    def __init__(self, id=None):
        if id is not None:
            super(Vertex, self).__init__(id)
        else:
            super(Vertex, self).__init__()


class MutableVertex(Vertex):
    def __init__(self, id=None, **kwargs):
        super(MutableVertex, self).__init__(id)
        self.__data = namedtuple('vertex_data', kwargs.keys())(*kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_data(self):
        return self.__data

    data = property(get_data)


class Edge(ObjectWithId):
    pass


class Digraph(object):
    def __init__(self):
        self.__vertices = set()
        self.__edges = OrderedDict()
        self.__inverted_edges = OrderedDict()

    def add_vertex(self, vertex=None):
        if vertex is None:
            vertex = Vertex()
        if vertex in self.__vertices:
            return False
        self.__vertices.add(vertex)
        self.__edges[vertex] = {}
        self.__inverted_edges[vertex] = {}
        return True

    def add_vertices(self, vertices):
        if not self.__vertices.isdisjoint(vertices):
            return False
        self.__vertices.update(vertices)
        self.__edges.update({vertex: {} for vertex in vertices})
        self.__inverted_edges.update({vertex: {} for vertex in vertices})
        return True

    def has_vertex(self, vertex):
        return vertex in self.__vertices

    def remove_vertex(self, vertex):
        for dst in [dst for dst, edge in self.get_inverted_neighbours(vertex)]:
            self.remove_edge(dst, vertex)
        del self.__edges[vertex]
        del self.__inverted_edges[vertex]
        self.__vertices.remove(vertex)

    def add_edge(self, src, dst, edge=None):
        if edge is None:
            edge = Edge()
        self.__edges[src][dst] = edge
        self.__inverted_edges[dst][src] = edge

    def has_edge(self, src, dst):
        return dst in self.__edges[src]

    def remove_edge(self, src, dst):
        del self.__edges[src][dst]
        del self.__inverted_edges[dst][src]

    def get_edge(self, src, dst):
        if not self.has_edge(src, dst):
            return None
        return self.__edges[src][dst]

    def has_neighbours(self, vertex):
        return len(self.__edges[vertex]) != 0

    def has_inverted_neighbours(self, vertex):
        return len(self.__inverted_edges[vertex]) != 0

    def get_neighbours(self, vertex):
        return self.__edges[vertex].items()

    def get_inverted_neighbours(self, vertex):
        return self.__inverted_edges[vertex].items()

    def invert_edge(self, src, dst):
        self.__edges[dst][src] = self.__edges[src][dst]
        del self.__edges[src][dst]
        self.__inverted_edges[src][dst] = self.__inverted_edges[dst][src]
        del self.__inverted_edges[dst][src]

    def get_vertices(self):
        return copy.copy(self.__vertices)

    def get_edges(self):
        return [(src, dst, edge) for src in self.__vertices for dst, edge in
                                 self.get_neighbours(src)]

    def copy(self):
        graph = Digraph()
        graph.add_vertices(self.__vertices)

        for src in self.__vertices:
            for dst, edge in self.get_neighbours(src):
                graph.add_edge(src, dst, edge)

        return graph

    def copy_inverted(self):
        graph = self.copy()
        graph.__edges, graph.__inverted_edges = graph.__inverted_edges, graph.__edges
        return graph

    def remove_all_edges(self):
        self.__edges = OrderedDict({vertex: {} for vertex in self.__vertices})
        self.__inverted_edges = OrderedDict({vertex: {} for vertex in self.__vertices})

    def __topological_sort_r(self, vertex, tmp_set, result):
        if tmp_set[vertex]:
            return

        tmp_set[vertex] = True
        for dst, edge in self.get_neighbours(vertex):
            self.__topological_sort_r(dst, tmp_set, result)

        result.append(vertex)

    def topological_sort(self):
        tmp_set = {vertex: False for vertex in self.__vertices}
        result = []

        for vertex in self.__vertices:
            self.__topological_sort_r(vertex, tmp_set, result)
        result.reverse()

        return result

    def __is_connected_r(self, vertex, tmp_set):
        if tmp_set[vertex]:
            return
        tmp_set[vertex] = True
        for dst, edge in self.get_neighbours(vertex):
            self.__is_connected_r(dst, tmp_set)

    def is_connected(self):
        possible_root = self.topological_sort()[0]
        tmp_set = {vertex: False for vertex in self.__vertices}
        self.__is_connected_r(possible_root, tmp_set)
        return False not in tmp_set.values()

    def is_tree(self):
        return self.is_connected() and len(self.__vertices) == len(self.edges) + 1

    vertices = property(get_vertices)
    edges = property(get_edges)

