#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Digraph
import CompoundDigraph
from CallAggregator import CallAggregator
from functools import total_ordering

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


class SpecEdge(Digraph.Edge):
    def __init__(self, id=None):
        super(SpecEdge, self).__init__(id)
        self.__inverted = False

    def invert(self):
        self.__inverted = not self.__inverted

    def get_inverted(self):
        return self.__inverted

    inverted = property(get_inverted)


@total_ordering
class CompoundLayer(list):
    def __cmp__(self, other):
        for i in range(min(len(self), len(other))):
            if self[i] < other[i]:
                return -1
            elif self[i] > other[i]:
                return 1
        return len(other) - len(self)

    def __le__(self, other):
        return self.__cmp__(other) <= 0

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __ge__(self, other):
        return self.__cmp__(other) >= 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

    def __eq__(self, other):
        return self.__cmp__(other) == 0

    @classmethod
    def get_common_prefix(cls, layer1, layer2):
        prefix = []
        for i in range(min(len(layer1), len(layer2))):
            if layer1[i] == layer2[i]:
                prefix.append(layer1[i])
            else:
                break
        return prefix


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
        self.__relation_graph_inverted = Digraph.Digraph()
        self.__vertex_links = self.__adj_graph.vertices
        self.__uplink_lists = {}
        self.__root_vertex = Digraph.Vertex()
        self.__compound_layer = {vertex:CompoundLayer() for vertex in self.__vertex_set}
        self.__add_vertex = CallAggregator()

    def __create_mutable_vertex(self, flag=False, level=0, origin=None, minimal=False):
        return Digraph.MutableVertex(flag=flag, level=level, origin=origin, minimal=minimal)

    def __reset_vertex_flags(self, graph):
        for vertex in graph.vertices:
            vertex.flag = False

    def __get_same_level_vertex(self, src, dst):
       if src.level < dst.level:
           return self.__uplink_lists[dst][dst.level - src.level]
       else:
           return self.__uplink_lists[src][src.level - dst.level]

    def __create_uplinks_recursively(self, src):
        for dst, edge in self.__inc_graph.get_neighbours(src):
            self.__uplink_lists[dst] = [dst] + self.__uplink_lists[src]
            self.__create_uplinks_recursively(dst)

    def __create_uplinks(self):
        self.__uplink_lists = {}
        for vertex in self.__inc_graph_inverted.vertices:
            if not self.__inc_graph_inverted.get_neighbours(vertex):
                self.__uplink_lists[vertex] = [vertex]
                self.__create_uplinks_recursively(vertex)
                break

    def __create_inverted_graph(self, graph):
        inverted_graph = Digraph.Digraph()
        inverted_graph.add_vertices(self.__vertex_set)
        for src, dst, edge in graph.edges:
            inverted_graph.add_edge(dst, src)
        return inverted_graph

    def __set_levels(self, src, level=0):
        src.level = level
        for dst, edge in self.__inc_graph.get_neighbours(src):
            self.__set_levels(dst, level + 1)

    def __derive_strict_relations(self, vertex):
        for adj_vertex, adj_edge in self.__adj_graph.get_neighbours(vertex):
            same_level_vertex = self.__get_same_level_vertex(vertex, adj_vertex)
            self.__relation_graph.add_edge(vertex, same_level_vertex, RelationEdge(Relation.LT))
            for inc_vertex, inc_edge in self.__inc_graph.get_neighbours(vertex):
                self.__derive_strict_relations(inc_vertex)

    def __make_acyclic(self, vertex_set):
        pass

    def __get_parent(self, vertex):
        return self.__inc_graph_inverted.get_edge(vertex)[0]

    def __derive_all_relations(self, vertex_set):
        child_vertices = set()
        for src in vertex_set:
            for dst, edge in self.__inc_graph.get_neighbours(src):
                child_vertices.add(dst)
        self.__derive_all_relations(child_vertices)

        for src in child_vertices:
            for dst in child_vertices:
                if self.__relation_graph.has_edge(src, dst):
                    src_parent = self.__get_parent(src)
                    dst_parent = self.__get_parent(dst)
                    if (src_parent != dst_parent) and (not self.__relation_graph.has_edge(src_parent, dst_parent)):
                        self.__relation_graph.add_edge(src_parent, dst_parent, RelationEdge(Relation.LE))

        self.__make_acyclic(vertex_set)

    def __derive_relation_graph(self):
        self.__relation_graph = Digraph.Digraph()
        self.__relation_graph.add_vertices(self.__vertex_set)
        self.__set_levels(self.__root_vertex)
        self.__derive_strict_relations(self.__root_vertex)
        self.__derive_all_relations({self.__root_vertex})
        self.__add_vertex.registerFunction(self.__relation_graph.add_vertex)

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
            self.__compound_layer[vertex].append(layer[vertex])
            for dst, edge in self.__inc_graph.get_neighbours(vertex):
                child_vertices.add(dst)

        self.__assign_compound_layers_to_level(child_vertices)


    def __assign_compound_layers(self):
        self.__create_uplinks()
        self.__inc_graph_inverted = self.__create_inverted_graph(self.__inc_graph)
        self.__add_vertex.registerFunction(self.__inc_graph_inverted.add_vertex)

        for vertex in self.__inc_graph_inverted.vertices:
            if not self.__inc_graph_inverted.get_neighbours(vertex):
                self.__root_vertex = vertex
                break

        self.__derive_relation_graph()
        self.__relation_graph_inverted = self.__create_inverted_graph(self.__adj_graph)
        self.__add_vertex.registerFunction(self.__relation_graph_inverted.add_vertex)

        for vertex in self.__relation_graph.vertices:
            if not self.__relation_graph_inverted.get_neighbours(vertex):
                vertex.minimal = True

        self.__assign_compound_layers_to_level([self.__root_vertex])

        for src, dst, edge in self.__adj_graph.edges:
            if self.__compound_layer[src] > self.__compound_layer[dst]:
                self.__adj_graph.invert_edge(src, dst)
                self.__adj_graph.get_edge(dst, src).invert()

    def __create_dummy_vertex_adj_chain(self, src, dst):
        src_layer = self.__compound_layer[src]
        dst_layer = self.__compound_layer[dst]
        prev_vertex = src
        next_vertex = None
        for i in range(src_layer[-1] + 1, dst_layer[-1]):
            next_vertex = self.__create_mutable_vertex(False, prev_vertex.level, None, False)
            self.__add_vertex(next_vertex)
            self.__compound_layer[next_vertex] = src_layer + [i]
            self.__adj_graph.add_edge(prev_vertex, next_vertex)
            prev_vertex = next_vertex
        self.__adj_graph.add_edge(next_vertex, dst)

    def __create_dummy_vertex_inc_chain(self, src, dst):
        prev_vertex = src
        src_layer = self.__compound_layer[src]
        dst_layer = self.__compound_layer[dst]
        for i in range(len(src_layer), len(dst_layer) - 1):
            next_vertex = self.__create_mutable_vertex(False, i, None, False)
            self.__add_vertex(next_vertex)
            self.__compound_layer[next_vertex] = self.__compound_layer[prev_vertex] + \
                                                 self.__compound_layer[dst][i]
            self.__relation_graph.add_edge(prev_vertex, next_vertex)
            prev_vertex = next_vertex
        self.__relation_graph.add_edge(prev_vertex, dst)

    def __normalize_graph(self):
        for src, dst, edge in self.__adj_graph.edges: #recall there are no parent-child adjacency edges!
            src_layer = self.__compound_layer[src]
            dst_layer = self.__compound_layer[dst]
            src_parent_layer = src_layer[0:-1]
            dst_parent_layer = dst_layer[0:-1]
            if src_parent_layer != dst_parent_layer:
                self.__adj_graph.remove_edge(src, dst)
                common_prefix = CompoundLayer.get_common_prefix(src_layer, dst_layer)
                if common_prefix == src_layer:
                    vertex1 = self.__create_mutable_vertex(False, src.level, None, False)
                    self.__add_vertex(vertex1)
                    self.__compound_layer[vertex1] = src_parent_layer + [dst_layer[len(common_prefix)]]
                    self.__create_dummy_vertex_adj_chain(src, vertex1)

                    vertex2 = self.__create_mutable_vertex(False, dst.level, None, False)
                    self.__compound_layer[vertex2] = dst_parent_layer + [dst_layer[-1] - 1]
                    self.__add_vertex(vertex2)
                    self.__create_dummy_vertex_inc_chain(vertex1, vertex2)
                    self.__adj_graph.add_edge(vertex2, dst)
                elif common_prefix == dst_layer:
                    vertex1 = self.__create_mutable_vertex(False, dst.level, None, False)
                    self.__add_vertex(vertex1)
                    self.__compound_layer[vertex1] = dst_parent_layer + [src_layer[len(common_prefix)]]
                    self.__create_dummy_vertex_adj_chain(vertex1, dst)

                    vertex2 = self.__create_mutable_vertex(False, src.level, None, False)
                    self.__compound_layer[vertex2] = src_parent_layer + [src_layer[-1] + 1]
                    self.__add_vertex(vertex2)
                    self.__create_dummy_vertex_inc_chain(vertex1, vertex2)
                    self.__adj_graph.add_edge(src, vertex2)
                else:
                    top_src = self.__create_mutable_vertex(False, len(common_prefix), None, False)
                    top_dst = self.__create_mutable_vertex(False, len(common_prefix), None, False)
                    self.__add_vertex(top_src)
                    self.__add_vertex(top_dst)
                    self.__compound_layer[top_src] = common_prefix + src_layer[len(common_prefix)]
                    self.__compound_layer[top_dst] = common_prefix + dst_layer[len(common_prefix)]
                    self.__create_dummy_vertex_adj_chain(top_src, top_dst)

                    src_neighbour = self.__create_mutable_vertex(False, src.level, None, False)
                    dst_neighbour = self.__create_mutable_vertex(False, dst.level, None, False)
                    self.__add_vertex(src_neighbour)
                    self.__add_vertex(dst_neighbour)
                    self.__compound_layer[src_neighbour] = src_parent_layer + [src_layer[-1] + 1]
                    self.__compound_layer[dst_neighbour] = dst_parent_layer + [dst_layer[-1] - 1]
                    self.__create_dummy_vertex_inc_chain(top_src, src_neighbour)
                    self.__create_dummy_vertex_inc_chain(top_dst, dst_neighbour)
                    self.__adj_graph.add_edge(src, src_neighbour)
                    self.__adj_graph.add_edge(dst_neighbour, dst)
            elif src_layer[-1] + 1 != dst_layer[-1]:
                self.__adj_graph.remove_edge(src, dst)
                self.__create_dummy_vertex_adj_chain(src, dst)

    def __prepare_graph(self, graph):
        self.__vertex_set.clear()
        for vertex in graph.vertices:
            mutable_vertex = self.__create_mutable_vertex(False, 0, vertex, False)
            self.__vertex_set.add(mutable_vertex)
            self.__vertex_dict[vertex] = mutable_vertex

        self.__inc_graph = Digraph.Digraph()
        self.__inc_graph.add_vertices(self.__vertex_set)
        self.__adj_graph = Digraph.Digraph()
        self.__adj_graph.add_vertices(self.__vertex_set)

        for src in graph.vertices:
            for dst, edge in graph.get_inc_edges(src):
                self.__inc_graph.add_edge(self.__vertex_dict[src], self.__vertex_dict[dst], SpecEdge())
            for dst, edge in graph.get_adj_edges(src):
                self.__adj_graph.add_edge(self.__vertex_dict[src], self.__vertex_dict[dst], SpecEdge())

        self.__add_vertex.unregisterAll()
        self.__add_vertex.registerFunction(self.__vertex_set.add)
        self.__add_vertex.registerFunction(self.__inc_graph.add_vertex)
        self.__add_vertex.registerFunction(self.__adj_graph.add_vertex)

    #we assume that inclusion graph is a rooted tree
    def draw(self, graph):
        self.__prepare_graph(graph)
        #we assume now there are now parent-child adjacency edges in the graph
        self.__assign_compound_layers()
        self.__normalize_graph()
        self.__determine_vertex_order()
        self.__position_graph()
