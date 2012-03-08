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
        self.__root_vertex = Digraph.Vertex()
        self.__compound_layers = {vertex:CompoundLayer() for vertex in self.__vertex_set}

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

    def __create_inverted_graph(self, graph):
        inverted_graph = Digraph.Digraph()
        inverted_graph.add_vertices(self.__vertex_set)
        for src in self.__vertex_set:
            for dst in self.__vertex_set:
                if graph.has_edge(src, dst):
                    inverted_graph.add_edge(dst, src)
        return inverted_graph

    def __derive_strict_relations(self, vertex):
        for adj_vertex, adj_edge in self.__adj_graph.get_edges(vertex):
            same_level_vertex = self.__get_same_level_vertex(vertex, adj_vertex)
            self.__relation_graph.add_edge(vertex, same_level_vertex, RelationEdge(Relation.LT))
            for inc_vertex, inc_edge in self.__inc_graph.get_edges(vertex):
                self.__derive_strict_relations(inc_vertex)

    def __make_acyclic(self, vertex_set):
        pass

    def __derive_all_relations(self, vertex_set):
        child_vertices = set()
        for src in vertex_set:
            for dst, edge in self.__inc_graph.get_edges(src):
                child_vertices.add(dst)
        self.__derive_all_relations(child_vertices)

        for src in child_vertices:
            for dst in child_vertices:
                if self.__relation_graph.has_edge(src, dst):
                    src_parent = tuple(self.__inc_graph_inverted.get_edges(src))[0]
                    dst_parent = tuple(self.__inc_graph_inverted.get_edges(dst))[0]
                    if (src_parent != dst_parent) and (not self.__relation_graph.has_edge(src_parent, dst_parent)):
                        self.__relation_graph.add_edge(src_parent, dst_parent, RelationEdge(Relation.LE))

        self.__make_acyclic(vertex_set)

    def __derive_relation_graph(self):
        self.__relation_graph = Digraph.Digraph()
        self.__relation_graph.add_vertices(self.__vertex_set)
        self.__derive_strict_relations(self.__root_vertex)
        self.__derive_all_relations({self.__root_vertex})

    def __assign_compound_layers_to_level(self, vertex_set):
        layer = {vertex:0 for vertex in vertex_set}
        for vertex in vertex_set:
            if vertex.minimal:
                layer[vertex] = 1
        while True:
            changed = False
            for src in vertex_set:
                if layer[src] >= 1:
                    for dst in vertex_set:
                        if self.__relation_graph.has_edge(src, dst):
                            edge = self.__relation_graph.get_edge(src, dst)
                            if (edge.relation == Relation.LT) and (layer[src] >= layer[dst]):
                                layer[dst] = layer[src] + 1
                                changed = True
                            if (edge.relation == Relation.LE) and (layer[src] > layer[dst]):
                                layer[dst] = layer[src]
                                changed = True
            if not changed:
                break

        child_vertices = set()
        for vertex in vertex_set:
            self.__compound_layers[vertex].append(layer[vertex])
            for dst, edge in self.__inc_graph.get_edges(vertex):
                child_vertices.add(dst)

        self.__assign_compound_layers_to_level(child_vertices)


    def __assign_compound_layers(self):
        self.__create_uplinks()
        self.__inc_graph_inverted = self.__create_inverted_graph(self.__inc_graph)

        for vertex in self.__inc_graph_inverted.vertices:
            if not self.__inc_graph_inverted.get_edges(vertex):
                self.__root_vertex = vertex
                break

        self.__derive_relation_graph()
        self.__relation_graph_inverted = self.__create_inverted_graph(self.__adj_graph)

        for vertex in self.__relation_graph.vertices:
            if not self.__relation_graph_inverted.get_edges(vertex):
                vertex.minimal = True

        self.__assign_compound_layers_to_level([self.__root_vertex])

    def __prepare_graph(self, graph):
        for vertex in graph.vertices:
            mutable_vertex = Digraph.MutableVertex(flag=False, level=0, origin=vertex, minimal=False)
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
