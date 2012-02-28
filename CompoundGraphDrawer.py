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
    #empty declarations
    #just declare object private variables
    #show variables interrelations
    def __init__(self):
        self.__graph = CompoundDigraph.CompoundDigraph()
        self.__inverted_graph = self.__graph.copy_inverted()
        self.__relation_graph = Digraph.Digraph()
        self.__vertex_links = self.__graph.vertices
        self.__uplink_lists = {[] for _ in range(len(self.__inverted_graph))}

    def __reset_vertex_flags(self, graph):
        for vertex in graph.vertices:
            vertex.flag = False

    def __topological_sort(self, graph):
        pass

    def __get_same_level_vertex(self, src, dst):
        if (self.__)

    def __create_uplinks_recursively(self, src):
        for dst, edge in self.__graph.get_inc_edges(src):
            self.__uplink_lists[dst] = [src] + self.__uplink_lists[src]
            self.__create_uplinks_recursively(dst)

    def __create_uplinks(self):
        self.__uplink_lists = {}
        for vertex in self.__inverted_graph.vertices:
            if len(self.__inverted_graph.get_edges(vertex)) == 0:
                self.__uplink_lists[vertex] = [None, ]
                self.__create_uplinks_recursively(vertex)
                break
        for vertex in self.__inverted_graph.vertices:
            self.__uplink_lists[vertex] = [None] + self.__uplink_lists[vertex]

    def __create_inverted_graph(self):
        self.__inverted_graph = Digraph.Digraph()
        for i, vertex in enumerate(self.__graph.vertices):
            self.__inverted_graph.add_vertex(Digraph.MutableIndexedVertex(i, flag=False, value=0, origin=vertex))
        for src in self.__graph.vertices:
            for dst in self.__graph.vertices:
                if self.__graph.has_inc_edge(src, dst):
                    self.__inverted_graph.add_edge(dst, src)

    def __derive_relations_recursively(self, vertex):
        for adj_vertex, adj_edge in self.__graph.get_adj_edges(vertex):
            same_level_vertex = self.__get_same_level_vertex(vertex, adj_vertex)
            self.__relation_graph.add_edge(vertex, same_level_vertex, RelationEdge(Relation.LT))
            for inc_vertex, inc_edge in self.__graph.get_inc_edges(vertex):
                self.__derive_relations_recursively(inc_vertex)

    def __derive_relation_graph(self):
        self.__relation_graph = Digraph.Digraph()

        for i, vertex in enumerate(self.__inverted_graph.vertices):
            self.__relation_graph.vertices.add_vertex(Digraph.MutableIndexedVertex(i, flag=False, level=0, origin=vertex))

        for vertex in self.__graph.vertices:
            if vertex.value == 0:
                self.__derive_relations_recursively(vertex)
                break

    def __set_relations(self):
        pass

    def __get_top_level(self):
        pass

    def __assign_compound_layers_to_level(self, vertex_list):
        layers = [0 for _ in range(len(vertex_list))]
        for vertex in vertex_list:
            if vertex.minimal:
                pass

    def __assign_compound_layers(self):
        self.__create_uplinks()
        self.__create_inverted_graph()
        self.__derive_relation_graph()
        self.__assign_compound_layers_to_level(self.__get_top_level())

    def __prepare_graph(self, graph):
        return graph.copy()

    def draw(self, graph):
        self.__graph = self.__prepare_graph(graph)
        self.__assign_compound_layers()
