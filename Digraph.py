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
    def __init__(self, index, flag=False, value=None):
        super(MutableIndexedVertex, self).__init__(index)
        self.__flag = flag
        self.__value = value

    def get_flag(self):
        return self.__flag

    def set_flag(self, value):
        self.__flag = value

    def get_value(self):
        return self.__value

    def set_value(self, value):
        self.__value = value

    flag = property(get_flag, set_flag)
    value = property(get_flag, set_flag)


class Edge(object):
    pass

class Digraph(object):
    def __init__(self):
        self.__vertices = set()
        self.__edges = {}

    def add_vertex(self, vertex):
        self.__vertices.add(vertex)
        self.__edges[vertex] = {}

    def has_vertex(self, vertex):
        return vertex in self.__vertices

    def remove_vertex(self, vertex):
        if not self.has_vertex(vertex):
            return False
        self.__vertices.remove(vertex)
        return True

    def add_edge(self, src, dst, edge):
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
        return tuple(self.__edges[vertex].items())

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

        for src_vertex in graph.vertices:
            for dst_vertex, edge in graph.get_edges(src_vertex):
                graph.add_edge(src_vertex, dst_vertex, edge)

    def copy_inverted(self):
        graph = self.copy()
        inverted_edges = set()
        for src_vertex in graph.vertices:
            for dst_vertex, edge in graph.get_edges(src_vertex):
                if graph.has_edge(src_vertex, dst_vertex) and edge not in inverted_edges:
                    graph.invert_edge(src_vertex, dst_vertex)
                    inverted_edges.add(graph.get_edge(dst_vertex, src_vertex))

        return graph

    vertices = property(get_vertices)
