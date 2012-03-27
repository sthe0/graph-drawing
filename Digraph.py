#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict

class ObjectWithId(object):
    __id = 0

    @classmethod
    def __get_id(cls):
        cls.__id += 1
        return cls.__id

    @classmethod
    def __update_id(cls, id):
        cls.__id = max(id, cls.__id)

    def __init__(self, id=None):
        if id is None:
            id = self.get_id()
        self.__id = id
        self.__update_id(id)

    def get_id(self):
        return self.__id

    def __hash__(self):
        return self.__id

    def __eq__(self, other):
        return self.__id == other.__id

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
        for key, value in kwargs.items():
            setattr(self, key, value)


class Edge(ObjectWithId):
    pass


class Digraph(object):
    def __init__(self):
        self.__vertices = set()
        self.__edges = OrderedDict()

    def add_vertex(self, vertex = Vertex()):
        self.__vertices.add(vertex)
        self.__edges[vertex] = {}

    def add_vertices(self, vertices):
        self.__vertices.update(vertices)
        self.__edges.update({vertex: {} for vertex in vertices})

    def has_vertex(self, vertex):
        return vertex in self.__vertices

    def remove_vertex(self, vertex):
        if not self.has_vertex(vertex):
            return False
        self.__vertices.remove(vertex)
        for dst in self.__edges[vertex].keys():
            del self.__edges[dst][vertex]
        del self.__edges[vertex]
        return True

    def add_edge(self, src, dst, edge = Edge()):
        if not self.has_vertex(src) or not self.has_vertex(dst):
            return False
        self.__edges[src][dst] = edge
        return True

    def has_edge(self, src, dst):
        if not self.has_vertex(src) or not self.has_vertex(dst):
            return False
        return dst in self.__edges[src]

    def remove_edge(self, src, dst):
        if not self.has_edge(src, dst):
            return False
        del self.__edges[src][dst]
        return True

    def get_edge(self, src, dst):
        if not self.has_edge(src, dst):
            return None
        return self.__edges[src][dst]

    def get_neighbours(self, vertex):
        return self.__edges[vertex].items()

    def invert_edge(self, src, dst):
        tmp = self.__edges[dst][src] if self.has_edge(dst, src) else None
        self.__edges[dst][src] = self.__edges[src][dst]
        if tmp is not None: self.__edges[src][dst] = tmp

    def get_vertices(self):
        return tuple(self.__vertices)

    def get_edges(self):
        return [(src, dst, edge) for src in self.__vertices for dst, edge in self.get_neighbours(src)]

    def copy(self):
        graph = Digraph()
        for vertex in self.__vertices:
            graph.add_vertex(vertex)

        for src in graph.vertices:
            for dst, edge in graph.get_neighbours(src):
                graph.add_edge(src, dst, edge)

    def copy_inverted(self):
        graph = self.copy()
        inverted_edges = set()
        for src in graph.vertices:
            for dst, edge in graph.get_neighbours(src):
                if graph.has_edge(src, dst) and edge not in inverted_edges:
                    graph.invert_edge(src, dst)
                    inverted_edges.add(graph.get_edge(dst, src))

        return graph

    def remove_all_edges(self):
        self.__edges = OrderedDict()

    vertices = property(get_vertices)
    edges = property(get_edges)
