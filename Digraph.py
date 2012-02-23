#!/usr/bin/env python
# -*- coding: utf-8 -*-
import CallAggregator


class Vertex(object):
    pass


class Edge(object):
    def __init__(self, src, dst):
        self.__src = src
        self.__dst = dst

    def get_src(self):
        return self.__src

    def get_dst(self):
        return self.__dst

    src_vertex = property(get_src)
    dst_vertex = property(get_dst)


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

    def invert_edge(self, src, dst):
        tmp = self.__edges[dst][src] if self.has_edge(dst, src) else None
        self.__edges[dst][src] = self.__edges[src][dst]
        if tmp is not None: self.__edges[src][dst] = tmp

    def get_vertices(self):
        return tuple(self.__vertices)

    vertices = property(get_vertices)
