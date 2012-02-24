#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Digraph
import CompoundDigraph


class Relation:
    LT = 0
    LE = 1


class RelationEdge(Digraph.Edge):
    def __init__(self, relation):
        super(RelationEdge, self).__init__()
        self.__relation = relation

    def update(self, relation):
        self.__relation = max(self.__relation, relation)

    def get_relation(self):
        return self.__relation

    relation = property(get_relation)


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
        self.__graph = CompoundDigraph.CompoundDiraph()
        self.__inverted_graph = self.__graph.copy_inverted()_
        self.__relation_graph = Digraph.Digraph()

    def __reset_vertex_flags(self, graph):
        for vertex in graph.vertices:
            vertex.flag = False

    def __get_same_level_vertex(self, src, dst):


    def __create_relation_graph(self, graph):
        self.__relation_graph = Digraph.Digraph()
        for i, vertex in enumerate(graph.vertices):
            self.__relation_graph.vertices.add_vertex(Digraph.MutableIndexedVertex(i))

    def __derive_relations_recursively(self, vertex):
        for adj_vertex, adj_edge in self.__graph.get_adj_edges(vertex):
            same_level_vertex = self.__get_same_level_vertex(vertex, adj_vertex)
            self.__relation_graph.add_edge(vertex, same_level_vertex, RelationEdge(Relation.LT))
            for inc_vertex, inc_edge in self.__graph.get_inc_edges(vertex):
                self.__derive_relations_recursively(inc_vertex)

    def __derive_relation_graph(self):
        self.__relation_graph = self.__create_relation_graph(self.__graph)
        for vertex in self.__relation_graph.vertices:
            if not vertex.flag:
                self.__derive_relations_recursively(vertex)

    def __set_relations(self):
        pass

    def __topological_sort(self):
        pass

    def __get_top_level(self):
        pass

    def __assign_compound_layers_to_level(self, vertex_list):
        layers = [0 for _ in range(len(vertex_list))]
        for vertex in vertex_list:
            if vertex.minimal:
                pass

    def __assign_compound_layers(self, graph):
        self.__graph = graph
        self.__relation_graph = self.__derive_relation_graph()
        self.__assign_compound_layers_to_level(self.__get_top_level(self.__relation_graph))

    def draw(self, graph):
        self.__assign_compound_layers(graph)
