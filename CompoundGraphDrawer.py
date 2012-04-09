#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Digraph
import CompoundDigraph
import DrawingFramework
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
    def __init__(self, inverted=False, id=None):
        super(SpecEdge, self).__init__(id)
        self.__inverted = inverted

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
    __order_iterations = 10

    #empty declarations
    #just declare object private variables
    #show variables interrelations
    def __init__(self, leaf_width=10, d1=1, d2=1):
        self.__vertex_set = set()
        self.__vertex_dict = {}
        self.__inc_graph = Digraph.Digraph()
        self.__adj_graph = Digraph.Digraph()
        self.__inc_graph_inverted = self.__inc_graph.copy_inverted()
        self.__adj_graph_inverted = self.__inc_graph.copy_inverted()
        self.__relation_graph = Digraph.Digraph()
        self.__relation_graph_inverted = Digraph.Digraph()
        self.__ordered_graph = Digraph.Digraph()
        self.__vertex_links = self.__adj_graph.vertices
        self.__uplink_lists = {}
        self.__root_vertex = Digraph.Vertex()
        self.__compound_layer = {vertex:CompoundLayer() for vertex in self.__vertex_set}
        self.__add_vertex = CallAggregator()
        self.__order_service_graph = Digraph.Digraph()
        self.__leaf_width = leaf_width
        self.__left_top_x = 1
        self.__left_top_y = 1
        self.__d1 = d1
        self.__d2 = d2
        self.__drawer = DrawingFramework.Canvas()

    def __create_mutable_vertex(self,
                                flag=False,
                                level=0,
                                origin=None,
                                minimal=False,
                                adj_left=0,
                                adj_right=0,
                                order_index=0,
                                width=0,
                                x=0,
                                y=0):
        return Digraph.MutableVertex(flag=flag,
                                     level=level,
                                     origin=origin,
                                     minimal=minimal,
                                     adj_left=adj_left,
                                     adj_right=adj_right,
                                     order_index=order_index,
                                     width=width,
                                     x=x,
                                     y=y)

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
                        self.__adj_graph.add_edge(src_parent, dst_parent, RelationEdge(Relation.LE))

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

    def __create_dummy_vertex_adj_chain(self, src, dst, inverted):
        src_layer = self.__compound_layer[src]
        dst_layer = self.__compound_layer[dst]
        prev_vertex = src
        next_vertex = None
        for i in range(src_layer[-1] + 1, dst_layer[-1]):
            next_vertex = self.__create_mutable_vertex(False, prev_vertex.level, None, False)
            self.__add_vertex(next_vertex)
            self.__compound_layer[next_vertex] = src_layer + [i]
            self.__adj_graph.add_edge(prev_vertex, next_vertex, SpecEdge(inverted))
            prev_vertex = next_vertex
        self.__adj_graph.add_edge(next_vertex, dst, SpecEdge(inverted))

    def __create_dummy_vertex_inc_chain(self, src, dst, inverted):
        prev_vertex = src
        src_layer = self.__compound_layer[src]
        dst_layer = self.__compound_layer[dst]
        for i in range(len(src_layer), len(dst_layer) - 1):
            next_vertex = self.__create_mutable_vertex(False, i, None, False)
            self.__add_vertex(next_vertex)
            self.__compound_layer[next_vertex] = self.__compound_layer[prev_vertex] + \
                                                 self.__compound_layer[dst][i]
            self.__adj_graph.add_edge(prev_vertex, next_vertex, SpecEdge(inverted))
            prev_vertex = next_vertex
        self.__relation_graph.add_edge(prev_vertex, dst, SpecEdge(inverted))

    def __normalize_graph(self):
        for src, dst, edge in self.__adj_graph.edges: #recall there are no parent-child adjacency edges!
            src_layer = self.__compound_layer[src]
            dst_layer = self.__compound_layer[dst]

            src_parent_layer = src_layer[0:-1]
            dst_parent_layer = dst_layer[0:-1]
            inverted = edge.inverted
            if src_parent_layer != dst_parent_layer:
                self.__adj_graph.remove_edge(src, dst)
                common_prefix = CompoundLayer.get_common_prefix(src_layer, dst_layer)
                if common_prefix == src_layer:
                    vertex1 = self.__create_mutable_vertex(False, src.level, None, False)
                    self.__add_vertex(vertex1)
                    self.__compound_layer[vertex1] = src_parent_layer + [dst_layer[len(common_prefix)]]
                    self.__create_dummy_vertex_adj_chain(src, vertex1, inverted)

                    vertex2 = self.__create_mutable_vertex(False, dst.level, None, False)
                    self.__compound_layer[vertex2] = dst_parent_layer + [dst_layer[-1] - 1]
                    self.__add_vertex(vertex2)
                    self.__create_dummy_vertex_inc_chain(vertex1, vertex2, inverted)
                    self.__adj_graph.add_edge(vertex2, dst, inverted)
                elif common_prefix == dst_layer:
                    vertex1 = self.__create_mutable_vertex(False, dst.level, None, False)
                    self.__add_vertex(vertex1)
                    self.__compound_layer[vertex1] = dst_parent_layer + [src_layer[len(common_prefix)]]
                    self.__create_dummy_vertex_adj_chain(vertex1, dst, inverted)

                    vertex2 = self.__create_mutable_vertex(False, src.level, None, False)
                    self.__compound_layer[vertex2] = src_parent_layer + [src_layer[-1] + 1]
                    self.__add_vertex(vertex2)
                    self.__create_dummy_vertex_inc_chain(vertex1, vertex2, inverted)
                    self.__adj_graph.add_edge(src, vertex2, SpecEdge(inverted))
                else:
                    top_src = self.__create_mutable_vertex(False, len(common_prefix), None, False)
                    top_dst = self.__create_mutable_vertex(False, len(common_prefix), None, False)
                    self.__add_vertex(top_src)
                    self.__add_vertex(top_dst)
                    self.__compound_layer[top_src] = common_prefix + src_layer[len(common_prefix)]
                    self.__compound_layer[top_dst] = common_prefix + dst_layer[len(common_prefix)]
                    self.__create_dummy_vertex_adj_chain(top_src, top_dst, inverted)

                    src_neighbour = self.__create_mutable_vertex(False, src.level, None, False)
                    dst_neighbour = self.__create_mutable_vertex(False, dst.level, None, False)
                    self.__add_vertex(src_neighbour)
                    self.__add_vertex   (dst_neighbour)
                    self.__compound_layer[src_neighbour] = src_parent_layer + [src_layer[-1] + 1]
                    self.__compound_layer[dst_neighbour] = dst_parent_layer + [dst_layer[-1] - 1]
                    self.__create_dummy_vertex_inc_chain(top_src, src_neighbour, inverted)
                    self.__create_dummy_vertex_inc_chain(top_dst, dst_neighbour, inverted)
                    self.__adj_graph.add_edge(src, src_neighbour, SpecEdge(inverted))
                    self.__adj_graph.add_edge(dst_neighbour, dst, SpecEdge(inverted))
            elif src_layer[-1] + 1 != dst_layer[-1]:
                self.__adj_graph.remove_edge(src, dst)
                self.__create_dummy_vertex_adj_chain(src, dst, inverted)

    def __get_adj_difference(self, vertex):
        return vertex.adj_left - vertex.adj_right

    def __split_into_levels(self, vertex):
        local_vertices = []
        for dst, edge in self.__inc_graph.get_neighbours(vertex):
            local_vertices.append(dst)

        levels = []
        for vrtx in local_vertices:
            levels[self.__compound_layer[vrtx][-1]].append(vrtx)

        return levels

    #TODO: check whether it is correct to use compound layers for order determination!
    def __count_level_neighbours(self, level):
        for vertex in level:
            for dst, edge in self.__order_service_graph.get_neighbours(vertex):
                if dst not in level:
                    if self.__compound_layer[dst][-1] < self.__compound_layer[vertex][-1]:
                        vertex.adj_left += 1
                    else:
                        vertex.adj_right += 1

    def __minimize_closeness(self, level):
        splitted = {"left" : [], "middle" : [], "right" : []}
        for vrtx in level:
            if self.__get_adj_difference(vrtx) < 0:
                splitted["left"].append(vrtx)
            elif self.__get_adj_difference(vrtx) == 0:
                splitted["middle"].append(vrtx)
            else:
                splitted["right"].append(vrtx)
        return splitted

    def __enumerate_level(self, level):
        for n, vertex in enumerate(sorted(level, self.__get_adj_difference)):
            vertex.order_index = n

    def __add_dummy_vertex(self, src, dst, vertex):
        self.__adj_graph.add_vertex(vertex)
        self.__adj_graph_inverted.add_vertex(vertex)
        self.__adj_graph.add_edge(src, vertex)
        self.__adj_graph.add_edge(dst, vertex)
        self.__adj_graph_inverted.add_edge(vertex, src)
        self.__adj_graph_inverted.add_edge(vertex, dst)

    def __create_dummies(self, level):
        dummies = []
        for src in level:
            for dst in level:
                if self.__order_service_graph.has_edge(src, dst):
                    vertex = self.__create_mutable_vertex(False, 0, None, False, 0, 0, 0)
                    self.__add_dummy_vertex(src, dst, vertex)
                    dummies.append(vertex)

    def __compute_barycenter(self, src, level):
        mean = 0
        for dst in level:
            if self.__adj_graph.has_edge(src, dst) or \
               self.__adj_graph_inverted.has_edge(src, dst):
                mean += dst.order_index
        src.order_index = mean / len(level)

    def __normalize_level(self, level):
        for n, item in enumerate(sorted(lambda x: x.order_index, level)):
            item.order_index = n

    def __barycentric_order(self, level1, level2):
        for vertex1 in level1:
            self.__compute_barycenter(vertex1, level2)
        for vertex2 in level2:
            self.__compute_barycenter(vertex2, level1)

        self.__normalize_level(level1)
        self.__normalize_level(level2)

    def __merge_lists(self, list1, list2, base_list):
        for vertex1 in list1:
            self.__compute_barycenter(vertex1, base_list)
        for vertex2 in list2:
            self.__compute_barycenter(vertex2, base_list)

        result = list1 + list2
        self.__normalize_level(result)

        return result

    def __remove_dummy_vertex(self, vertex):
        if vertex.origin is None:
            for dst, edge in self.__adj_graph_inverted.get_neighbours(vertex):
                self.__adj_graph.remove_edge(dst, vertex)
                self.__adj_graph_inverted.remove_edge(vertex, dst)
            self.__adj_graph.remove_vertex(vertex)
            self.__adj_graph_inverted.remove_vertex(vertex)
            return False
        return True

    def __remove_dummies(self, level):
        level = list(filter(self.__remove_dummy_vertex, level))
        return level

    def __make_ordering_step(self, splitted, index, prev):
        dummies = self.__create_dummies(splitted[index]["middle"])
        merged = self.__merge_lists(splitted[prev]["middle"], dummies, splitted[index]["middle"])
        self.__barycentric_order(merged, splitted[index]["middle"])
        splitted[prev]["middle"] = self.__remove_dummies(merged)

    #TODO: store levels somehow!
    def __order_local(self, vertex):
        levels = self.__split_into_levels(vertex)
        splitted = []
        for level in levels:
            self.__count_level_neighbours(level)
            self.__enumerate_level(level)
            splitted.append(self.__minimize_closeness(level))

        for i in range(self.__order_iterations):
            init_dummies = self.__create_dummies(splitted[0]["middle"])
            self.__barycentric_order(init_dummies, splitted[0]["middle"])

            for j in range(1, len(splitted)):
                self.__make_ordering_step(splitted, j, j - 1)

            init_dummies = self.__create_dummies(splitted[-1]["middle"])
            self.__barycentric_order(init_dummies, splitted[-1]["middle"])

            for j in reversed(range(0, len(splitted) - 1)):
                self.__make_ordering_step(splitted, j, j + 1)

    def __order_global(self, vertex):
        self.__order_local(vertex)
        for child, edge in self.__inc_graph.get_neighbours(vertex):
            self.__order_global(child)

    def __init_order_service_graph(self, src):
        for dst, edge in self.__inc_graph.get_neighbours(src):
            self.__init_order_service_graph(dst)

        for dst, edge in self.__adj_graph.get_neighbours(src):
            parent_src = self.__get_parent(src)
            parent_dst = self.__get_parent(dst)
            if parent_src != parent_dst:
                self.__order_service_graph.add_edge(parent_src, parent_dst)
                self.__order_service_graph.add_edge(parent_dst, parent_src)

    def __determine_vertex_order(self):
        self.__order_service_graph = Digraph.Digraph()
        self.__order_service_graph.add_vertices(self.__vertex_set)
        self.__add_vertex.registerFunction(self.__order_service_graph.add_vertex)
        self.__init_order_service_graph(self.__root_vertex)

        self.__adj_graph_inverted = self.__create_inverted_graph(self.__adj_graph)
        self.__add_vertex.registerFunction(self.__adj_graph_inverted.add_vertex)

        self.__ordered_graph = Digraph.Digraph()
        self.__ordered_graph.add_vertices(self.__vertex_set)
        self.__add_vertex.registerFunction(self.__ordered_graph.add_vertex())
        self.__order_global(self.__root_vertex)

    def __compute_connectivity(self, src, level):
        connectivity = 0
        for dst in level:
            if self.__adj_graph.has_edge(src, dst) or\
               self.__adj_graph_inverted.has_edge(src, dst):
                connectivity += 1
        return connectivity

    def __split_level(self, level, top_vertex):
        left_part = []
        rigth_part = []
        for vertex in level:
            if vertex.x < top_vertex.x:
                left_part.append(vertex)
            else:
                rigth_part.append(vertex)
        return left_part, rigth_part

    def __improve_local(self, level, left_bound, right_bound):
        if len(level) == 0:
            return []

        top_vertex = level.pop()
        left_part, right_part = self.__split_level(level, top_vertex)

        right_shift = min(right_bound - top_vertex.order_index, right_bound - top_vertex.x - 1 - len(right_part))
        left_shift = min(top_vertex.order_index - left_bound, top_vertex.x - left_bound - 1 - len(right_part))

        if right_shift > 0:
            top_vertex.x = max(top_vertex.x + right_shift, top_vertex.x)
            for n, vertex in enumerate(right_part):
                new_x = top_vertex.x + n + 1 + right_shift
                if new_x < vertex.x:
                    break
                vertex.x = new_x

        if left_shift > 0:
            top_vertex.x = min(top_vertex.x - right_shift, top_vertex.x)
            for n, vertex in enumerate(left_part):
                new_x = top_vertex.x - n - 1 - right_shift
                if vertex.x < new_x:
                    break
                vertex.x = new_x

        return self.__improve_local(left_part, left_bound, top_vertex.x) + [top_vertex] + \
               self.__improve_local(right_part, top_vertex.x, right_bound)

    def __improve_positions(self, levels, index, prev):
        sorted_level = list(sorted(levels[index], lambda x: x.connectivity[prev - index]))
        for vertex in sorted_level:
            self.__compute_barycenter(vertex, levels[prev])
        return self.__improve_local(sorted_level, 0, len(sorted_level) * 2)

    def __prm_method(self, vertex):
        levels = self.__split_into_levels(vertex) #maybe it's better to use global levels list

        for i in range(0, len(levels)):
            levels[i][0].x = levels[i][0].width / 2

            for j in range(1, len(levels[i])):
                levels[i][j].x = levels[i][j - 1].x + self.__d2 + levels[i][j - 1].width / 2 + levels[i][j].width / 2

            for j in range(0, len(levels[i])):
                levels[i][j].connectivity = {}
                if i < len(levels) - 1:
                    levels[i][j].connectivity[-1] = self.__compute_connectivity(levels[i][j], levels[i - 1])
                if i > 0:
                    levels[i][j].connectivity[+1] = self.__compute_connectivity(levels[i][j], levels[i + 1])

            for i in range(1, len(levels)):
                levels = self.__improve_positions(levels, i, i - 1)
            for i in range(reversed(range(0, len(levels) - 1))):
                levels = self.__improve_positions(levels, i, i + 1)
            for i in range(1, len(levels)):
                levels = self.__improve_positions(levels, i, i - 1)

    def __layout_local(self, vertex):
        if len(self.__inc_graph.get_neighbours(vertex)) == 0:
            vertex.width = self.__leaf_width
            return

        for dst, edge in self.__inc_graph.get_neighbours(vertex):
            self.__layout_local(dst)

        self.__prm_method(vertex)

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
            for dst, edge in graph.get_inc_neighbours(src):
                self.__inc_graph.add_edge(self.__vertex_dict[src], self.__vertex_dict[dst], SpecEdge())
            for dst, edge in graph.get_adj_neighbours(src):
                self.__adj_graph.add_edge(self.__vertex_dict[src], self.__vertex_dict[dst], SpecEdge())

        self.__add_vertex.unregisterAll()
        self.__add_vertex.registerFunction(self.__vertex_set.add)
        self.__add_vertex.registerFunction(self.__inc_graph.add_vertex)
        self.__add_vertex.registerFunction(self.__adj_graph.add_vertex)

    def __layout(self):
        pass

    #we assume that inclusion graph is a rooted tree
    def draw(self, graph):
        self.__prepare_graph(graph)
        #we assume now there are now parent-child adjacency edges in the graph
        self.__assign_compound_layers()
        self.__normalize_graph()
        self.__determine_vertex_order()
        self.__layout_local(self.__root_vertex)
        self.__layout()