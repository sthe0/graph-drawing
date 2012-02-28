#!/usr/bin/env python
# -*- coding: utf-8 -*-
import CallAggregator


class Vertex(object):
    pass


class IndexedVertex(Vertex):
    def __init__(self, index):
        super(IndexedVertex, self).__init__()
        self.__index = index

    def get_index(self):
        return self.__index

    def __hash__(self):
        return self.__index

    index = property(get_index)


class MutableIndexedVertex(IndexedVertex):
    def __init__(self, index, **kwargs):
        super(MutableIndexedVertex, self).__init__(index)
        for key, value in kwargs.items():
            setattr(self, key, value)


class Edge(object):
    pass


class Digraph(object):
    def __init__(self):
        self.__vertices = set()
        self.__edges = {}

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

    def get_edges(self, vertex):
        return self.__edges[vertex].items()

    def invert_edge(self, src, dst):
        tmp = self.__edges[dst][src] if self.has_edge(dst, src) else None
        self.__edges[dst][src] = self.__edges[src][dst]
        if tmp is not None: self.__edges[src][dst] = tmp

    def get_vertices(self):
        return tuple(self.__vertices)

    def copy(self):
        graph = Digraph()
        for vertex in self.__vertices:
            graph.add_vertex(vertex)

        for src in graph.vertices:
            for dst, edge in graph.get_edges(src):
                graph.add_edge(src, dst, edge)

    def copy_inverted(self):
        graph = self.copy()
        inverted_edges = set()
        for src in graph.vertices:
            for dst, edge in graph.get_edges(src):
                if graph.has_edge(src, dst) and edge not in inverted_edges:
                    graph.invert_edge(src, dst)
                    inverted_edges.add(graph.get_edge(dst, src))

        return graph

    vertices = property(get_vertices)
