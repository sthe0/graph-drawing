#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Digraph

class CompoundDiraph(object):
    def __init__(self):
        self.__adj_graph = Digraph.Digraph()
        self.__inc_graph = Digraph.Digraph()

    def add_vertex(self, vertex):
        return self.__inc_graph.add_vertex(vertex) and self.__adj_graph.add_vertex(vertex)

    def has_vertex(self, vertex):
        return self.__adj_graph.has_vertex(vertex) and self.__inc_graph.has_vertex(vertex)

    def remove_vertex(self, vertex):
        return self.__adj_graph.remove_vertex(vertex) and self.__inc_graph.remove_vertex(vertex)

    def add_adj_edge(self, src, dst, edge):
        return self.__adj_graph.add_edge(src, dst, edge)

    def add_inc_edge(self, src, dst, edge):
        return self.__inc_graph.add_edge(src, dst, edge)

    def has_adj_edge(self, src, dst):
        return self.__adj_graph.has_edge(src, dst)

    def has_inc_edge(self, src, dst):
        return self.__inc_graph.has_edge(src, dst)

    def get_adj_edge(self, src, dst):
        return self.__adj_graph.get_edge(src, dst)

    def get_inc_edge(self, src, dst):
        return self.__inc_graph.get_edge(src, dst)

    def remove_adj_edge(self, src, dst):
        return self.__adj_graph.remove_edge(src, dst)

    def remove_inc_edge(self, src, dst):
        return self.__inc_graph.remove_edge(src, dst)

    def invert_adj_edge(self, src, dst):
        return self.__adj_graph.invert_edge(src, dst)

    def invert_inc_edge(self, src, dst):
        return self.__inc_graph.invert_edge(src, dst)

    def get_adj_edges(self, vertex):
        return self.__adj_graph.get_edges(vertex)

    def get_inc_edges(self, vertex):
        return self.__inc_graph.get_edges(vertex)

    def copy(self):
        graph = CompoundDiraph()
        graph.__inc_graph = graph.__inc_graph.copy()
        graph.__adj_graph = graph.__adj_graph.copy()
        return graph

    def copy_inverted(self, invert_adj_graph = True, invert_inc_graph = True):
        graph = self.copy()
        if invert_adj_graph:
            graph.__inc_graph = graph.__inc_graph.copy_inverted()
            graph.__inc_graph = graph.__inc_graph.copy_inverted()
