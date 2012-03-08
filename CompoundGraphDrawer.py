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
        self.__vertex_set = set()
        self.__vertex_dict = {}
        self.__inc_graph = Digraph.Digraph()
        self.__adj_graph = Digraph.Digraph()
        self.__inc_graph_inverted = self.__inc_graph.copy_inverted()
        self.__relation_graph = Digraph.Digraph()
        self.__vertex_links = self.__adj_graph.vertices
        self.__uplink_lists = {}

    def __reset_vertex_flags(self, graph):
        for vertex in graph.vertices:
            vertex.flag = False

    def __topological_sort(self, graph):
        pass

    def __get_same_level_vertex(self, src, dst):
       if src.level < dst.level:
           return self.__uplink_lists[dst][dst.level - src.level]
       else:
           return self.__uplink_lists[src][src.level - dst.level]

    def __create_uplinks_recursively(self, src):
        for dst, edge in self.__inc_graph.get_edges(src):
            self.__uplink_lists[dst] = [dst] + self.__uplink_lists[src]
            self.__create_uplinks_recursively(dst)

    def __create_uplinks(self):
        self.__uplink_lists = {}
        for vertex in self.__inc_graph_inverted.vertices:
            if not self.__inc_graph_inverted.get_edges(vertex):
                self.__uplink_lists[vertex] = [vertex]
                self.__create_uplinks_recursively(vertex)
                break

    def __create_inverted_graph(self):
        self.__inc_graph_inverted = Digraph.Digraph()
        self.__inc_graph_inverted.add_vertices(self.__vertex_set)
        for src in self.__vertex_set:
            for dst in self.__vertex_set:
                if self.__inc_graph.has_edge(src, dst):
                    self.__inc_graph_inverted.add_edge(dst, src)

    def __derive_strict_relations(self, vertex):
        for adj_vertex, adj_edge in self.__adj_graph.get_edges(vertex):
            same_level_vertex = self.__get_same_level_vertex(vertex, adj_vertex)
            self.__relation_graph.add_edge(vertex, same_level_vertex, RelationEdge(Relation.LT))
            for inc_vertex, inc_edge in self.__inc_graph.get_edges(vertex):
                self.__derive_strict_relations(inc_vertex)

    def __make_acyclic(self, vertex_set):
        pass

    def __derive_all_relations(self, vertex_set):
        next_level_children = set()
        for src in vertex_set:
            for dst, edge in self.__inc_graph.get_edges(src):
                next_level_children.add(dst)
        self.__derive_all_relations(next_level_children)

        for src in next_level_children:
            for dst in next_level_children:
                if self.__relation_graph.has_edge(src, dst):
                    src_parent = tuple(self.__inc_graph_inverted.get_edges(src))[0]
                    dst_parent = tuple(self.__inc_graph_inverted.get_edges(dst))[0]
                    if (src_parent != dst_parent) and (not self.__relation_graph.has_edge(src_parent, dst_parent)):
                        self.__relation_graph.add_edge(src_parent, dst_parent, RelationEdge(Relation.LE))

        self.__make_acyclic(vertex_set)

    def __derive_relation_graph(self):
        self.__relation_graph = Digraph.Digraph()
        self.__relation_graph.add_vertices(self.__vertex_set)

        for vertex in self.__inc_graph_inverted.vertices:
            if not self.__inc_graph_inverted.get_edges(vertex):
                self.__derive_strict_relations(vertex)
                self.__derive_all_relations([vertex])
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
        for vertex in graph.vertices:
            mutable_vertex = Digraph.MutableVertex(flag=False, level=0, origin=vertex)
            self.__vertex_set.add(mutable_vertex)
            self.__vertex_dict[vertex] = mutable_vertex
        self.__inc_graph = Digraph.Digraph()
        self.__inc_graph.add_vertices(self.__vertex_set)
        self.__adj_graph = Digraph.Digraph()
        self.__adj_graph.add_vertices(self.__vertex_set)
        for src in graph.vertices:
            for dst, edge in graph.get_inc_edges(src):
                self.__inc_graph.add_edge(self.__vertex_dict[src], self.__vertex_dict[dst], edge)
            for dst, edge in graph.get_adj_edges(src):
                self.__adj_graph.add_edge(self.__vertex_dict[src], self.__vertex_dict[dst], edge)

    def draw(self, graph):
        self.__prepare_graph(graph)
        self.__assign_compound_layers()
