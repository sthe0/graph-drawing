#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Digraph

class CompoundDigraph(object):
    def __init__(self):
        self._adj_graph = Digraph.Digraph()
        self._inc_graph = Digraph.Digraph()

    def add_vertex(self, vertex):
        return self._inc_graph.add_vertex(vertex) and self._adj_graph.add_vertex(vertex)

    def add_vertices(self, vertices):
        return self._inc_graph.add_vertices(vertices) and self._adj_graph.add_vertex(vertices)

    def has_vertex(self, vertex):
        return self._adj_graph.has_vertex(vertex) and self._inc_graph.has_vertex(vertex)

    def remove_vertex(self, vertex):
        return self._adj_graph.remove_vertex(vertex) and self._inc_graph.remove_vertex(vertex)

    def add_adj_edge(self, src, dst, edge=None):
        return self._adj_graph.add_edge(src, dst, edge)

    def add_inc_edge(self, src, dst, edge=None):
        return self._inc_graph.add_edge(src, dst, edge)

    def has_adj_edge(self, src, dst):
        return self._adj_graph.has_edge(src, dst)

    def has_inc_edge(self, src, dst):
        return self._inc_graph.has_edge(src, dst)

    def get_adj_edge(self, src, dst):
        return self._adj_graph.get_edge(src, dst)

    def get_inc_edge(self, src, dst):
        return self._inc_graph.get_edge(src, dst)

    def remove_adj_edge(self, src, dst):
        return self._adj_graph.remove_edge(src, dst)

    def remove_inc_edge(self, src, dst):
        return self._inc_graph.remove_edge(src, dst)

    def invert_adj_edge(self, src, dst):
        return self._adj_graph.invert_edge(src, dst)

    def invert_inc_edge(self, src, dst):
        return self._inc_graph.invert_edge(src, dst)

    def get_adj_neighbours(self, vertex):
        return self._adj_graph.get_neighbours(vertex)

    def get_inc_neighbours(self, vertex):
        return self._inc_graph.get_neighbours(vertex)

    def copy_adj_graph(self):
        return self._adj_graph.copy()

    def copy_inc_graph(self):
        return self._inc_graph.copy()

    def copy(self):
        graph = CompoundDigraph()
        graph._inc_graph = graph._inc_graph.copy()
        graph._adj_graph = graph._adj_graph.copy()
        return graph

    def copy_inverted(self, invert_adj_graph=True, invert_inc_graph=True):
        graph = self.copy()
        if invert_adj_graph:
            graph._adj_graph = graph._adj_graph.copy_inverted()
        if invert_inc_graph:
            graph._inc_graph = graph._inc_graph.copy_inverted()

    def get_vertices(self):
        return self._adj_graph.vertices

    vertices = property(get_vertices)
