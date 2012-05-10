#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import Digraph
import CompoundDigraph
import DrawingFramework
from CallAggregator import CallAggregator
from copy import copy
from Tree import Tree

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


class CompoundLayer(list):
    def __init__(self):
        super(CompoundLayer, self).__init__()

    def cmp(self, other):
        for i in range(min(len(self), len(other))):
            if self[i] < other[i]:
                return -1
            elif self[i] > other[i]:
                return 1
        return len(other) - len(self)

    def __le__(self, other):
        return self.cmp(other) <= 0

    def __lt__(self, other):
        return self.cmp(other) < 0

    def __ge__(self, other):
        return self.cmp(other) >= 0

    def __gt__(self, other):
        return self.cmp(other) > 0

    def __eq__(self, other):
        return self.cmp(other) == 0

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
    def __init__(self, leaf_width=10, leaf_height = 10, d1=1, d2=1):
        self.__inc_tree = Tree()
        self.__inc_tree_nodes = {}
        self.__adj_graph = Digraph.Digraph()
        self.__relation_graph = Digraph.Digraph()
        self.__ordered_graph = Digraph.Digraph()
        self.__compound_layer_tree = Tree()
        self.__compound_layer_tree_nodes = {}
        self.__fake_vertices = set()
        self.__dummy_vertices = set()
        self.__adj_left = {}
        self.__adj_right = {}
        self.__order_service_graph = Digraph.Digraph()
        self.__order_index = {}
        self.__barycenter = {}
        self.__x = {}
        self.__y = {}
        self.__width = {}
        self.__leaf_width = leaf_width
        self.__leaf_height = leaf_height
        self.__left_top_x = 1
        self.__left_top_y = 1
        self.__d1 = d1
        self.__d2 = d2
        self.__drawer = DrawingFramework.DrawingFramework()

    def __compare_compound_layers(self, node1, node2):
        least_common_ancestor = self.__compound_layer_tree.get_least_common_ancestor(node1, node2)
        index1, ancestor1 = self.__compound_layer_tree.get_child_ancestor(least_common_ancestor, node1)
        index2, ancestor2 = self.__compound_layer_tree.get_child_ancestor(least_common_ancestor, node2)
        if ancestor1.data < ancestor2.data:
            return -1
        if ancestor1.data > ancestor2.data:
            return 1
        return 0

    def __derive_strict_relations(self, node):
        for dst, edge in self.__adj_graph.get_neighbours(node.origin):
            node1, node2 = self.__inc_tree.get_same_level_vertices(node, self.__inc_tree_nodes[dst])
            self.__relation_graph.add_edge(node1.origin, node2.origin, RelationEdge(Relation.LT))
        for index, child in node.children:
            self.__derive_strict_relations(child.origin)

    def __induce_graph(self, node_set):
        vertices = [node.origin for node in node_set]
        graph = Digraph.Digraph()
        graph.add_vertices(vertices)
        for src in vertices:
            for dst in vertices:
                if self.__relation_graph.has_edge(src, dst):
                    graph.add_edge(src, dst)
        return graph

    def __make_acyclic(self, induced_graph):
        ordered_vertices = {vertex: n for n, vertex in enumerate(
                                sorted(induced_graph.vertices,
                                       key=(lambda x: len(induced_graph.get_neighbours(x)))))}
        for src in induced_graph.vertices:
            for dst, edge in induced_graph.get_neighbours(src):
                if ordered_vertices[src] < ordered_vertices[dst]:
                    self.__relation_graph.invert_edge(src, dst)

    def __derive_all_relations(self, node_set):
        if len(node_set) == 0:
            return

        children = set()
        for node in node_set:
            for index, child in node.children:
                children.add(child)
        self.__derive_all_relations(children)

        for src_node in node_set:
            for dst_node in node_set:
                if self.__relation_graph.has_edge(src_node.origin, dst_node.origin):
                    src_parent_node = src_node.parent
                    dst_parent_node = dst_node.parent
                    if src_parent_node != dst_parent_node and \
                       not self.__relation_graph.has_edge(src_parent_node.origin, dst_parent_node.origin):
                        self.__relation_graph.add_edge(src_parent_node, dst_parent_node, RelationEdge(Relation.LE))

        self.__make_acyclic(self.__induce_graph(node_set))

    def __derive_relation_graph(self):
        self.__relation_graph = Digraph.Digraph()
        self.__relation_graph.add_vertices(self.__adj_graph.vertices)
        self.__derive_strict_relations(self.__inc_tree.root)
        self.__derive_all_relations(set(self.__inc_tree.root.children.values()))

    def __assign_compound_layers_r(self, vertex_set):
        if len(vertex_set) == 0:
            return

        layer = {vertex:0 for vertex in vertex_set}
        for vertex in vertex_set:
            if not self.__relation_graph.has_inverted_neighbours(vertex):
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

        children = set()
        for vertex in vertex_set:
            parent = self.__inc_tree_nodes[vertex].parent.origin
            new_node = self.__compound_layer_tree_nodes[parent].add_child(layer[vertex], layer[vertex])[0]
            self.__compound_layer_tree_nodes[vertex] = new_node
            for index, child in self.__inc_tree_nodes[vertex].children:
                children.add(child.origin)

        self.__assign_compound_layers_r(children)

    def __assign_compound_layers(self):
        self.__derive_relation_graph()

        self.__compound_layer_tree = Tree(root_data=1)
        self.__compound_layer_tree_nodes[self.__inc_tree.root.origin] = self.__compound_layer_tree.root
        self.__assign_compound_layers_r([self.__inc_tree.root.origin])

        for src, dst, edge in self.__adj_graph.edges:
            if self.__compare_compound_layers(self.__compound_layer_tree_nodes[src],
                                              self.__compound_layer_tree_nodes[dst]) > 0:
                self.__adj_graph.invert_edge(src, dst)
                self.__adj_graph.get_edge(dst, src).invert()

    def __add_vertex(self, parent_node, cl_index):
        vertex = Digraph.Vertex()
        self.__adj_graph.add_vertex(vertex)

        node = parent_node.add_child(len(parent_node.children))[0]
        node.origin = vertex
        self.__inc_tree_nodes[vertex] = node

        cl_node = self.__compound_layer_tree_nodes[parent_node.origin][cl_index]
        self.__compound_layer_tree_nodes[vertex] = cl_node

        return node

    def __create_fake_vertex_adj_chain(self, src, dst, inverted):
        src_cl_node = self.__compound_layer_tree_nodes[src]
        dst_cl_node = self.__compound_layer_tree_nodes[dst]
        prev_vertex = src
        for i in range(src_cl_node.data + 1, dst_cl_node.data):
            next_vertex = self.__add_vertex(src_cl_node.parent, i).origin
            self.__adj_graph.add_edge(prev_vertex, next_vertex, SpecEdge(inverted))
            prev_vertex = next_vertex
        self.__adj_graph.add_edge(prev_vertex, dst, SpecEdge(inverted))

    def __create_fake_vertex_inc_chain(self, inc_node, cl_node, adjustment):
        prev_node = inc_node

        prev_cl_node = cl_node
        indexes = []
        while prev_cl_node.level != prev_node.level:
            indexes.append(prev_cl_node.data)
            prev_cl_node = prev_cl_node.parent
        indexes.reverse()
        if indexes:
            indexes[-1] += adjustment

        top_node = None
        if cl_node.level > inc_node.level:
            top_node = self.__add_vertex(prev_node, indexes[0])
            self.__fake_vertices.add(top_node.origin)
            prev_node = top_node

        for i in range(1, cl_node.level - inc_node.level):
            prev_node = self.__add_vertex(prev_node, indexes[i])
            self.__fake_vertices.add(prev_node.origin)

        return prev_node, top_node

    def __normalize_graph(self):
        cl_common_ancestors = {}
        for src, dst, edge in self.__adj_graph.edges:
            src_cl_node = self.__compound_layer_tree_nodes[src]
            dst_cl_node = self.__compound_layer_tree_nodes[dst]
            key = (src_cl_node, dst_cl_node)
            cl_common_ancestors[key] = self.__compound_layer_tree.get_least_common_ancestor(src_cl_node, dst_cl_node)
        self.__compound_layer_tree.finished = False

        for src, dst, edge in self.__adj_graph.edges: #recall there are no parent-child adjacency edges!
            src_cl_node = self.__compound_layer_tree_nodes[src]
            dst_cl_node = self.__compound_layer_tree_nodes[dst]

            if src_cl_node.parent == dst_cl_node.parent and dst_cl_node.data - dst_cl_node.data == 1:
                continue

            inverted = edge.inverted
            self.__adj_graph.remove_edge(src, dst)

            src_node = self.__inc_tree_nodes[src]
            dst_node = self.__inc_tree_nodes[dst]
            src_top_node, dst_top_node = src_node, dst_node

            common_ancestor = self.__inc_tree.get_least_common_ancestor(src_node, dst_node)
            cl_common_ancestor = cl_common_ancestors[(src_cl_node, dst_cl_node)]
            fake_common_ancestor = self.__create_fake_vertex_inc_chain(common_ancestor, cl_common_ancestor, 0)[0]

            if src_cl_node.parent != cl_common_ancestor:
                src_neighbour_node, src_top_node = self.__create_fake_vertex_inc_chain(fake_common_ancestor, src_cl_node, 1)
                self.__adj_graph.add_edge(src_cl_node.origin, src_neighbour_node.origin, SpecEdge(inverted))
            if dst_cl_node.paren != cl_common_ancestor:
                dst_neighbour_node, dst_top_node = self.__create_fake_vertex_inc_chain(fake_common_ancestor, dst_cl_node, -1)
                self.__adj_graph.add_edge(dst_neighbour_node.origin, dst_cl_node.origin, SpecEdge(inverted))
            self.__create_fake_vertex_adj_chain(src_top_node, dst_top_node, inverted)

        self.__compound_layer_tree.finished = True

    def __get_adj_difference(self, vertex):
        return self.__adj_left[vertex] - self.__adj_right[vertex]

    def __split_into_levels(self, node):
        local_vertices = [child.origin for index, child in self.__inc_tree_nodes[node].children]

        levels = {}
        for vertex in local_vertices:
            levels.setdefault(self.__compound_layer_tree_nodes[vertex].data, []).append(vertex)

        return [value for key, value in levels.items()]

    def __count_level_neighbours(self, level):
        for vertex in level:
            self.__adj_left[vertex] = 0
            self.__adj_right[vertex] = 0
            for dst, edge in self.__order_service_graph.get_neighbours(vertex):
                vertex_parent_node = self.__inc_tree_nodes[vertex].parent
                dst_parent_node = self.__inc_tree_nodes[dst].parent
                if vertex_parent_node != dst_parent_node:
                    if self.__order_index[dst] < self.__order_index[vertex]:
                        self.__adj_left[vertex] += 1
                    else:
                        self.__adj_right[vertex] += 1

    def __minimize_closeness(self, level):
        splitted = {"left" : [], "middle" : [], "right" : []}
        for vertex in level:
            if self.__get_adj_difference(vertex) < 0:
                splitted["left"].append(vertex)
            elif self.__get_adj_difference(vertex) == 0:
                splitted["middle"].append(vertex)
            else:
                splitted["right"].append(vertex)
        return splitted

    def __create_dummies(self, level):
        dummies = []
        for src in level:
            for dst in level:
                if self.__order_service_graph.has_edge(src, dst):
                    vertex = Digraph.Vertex()
                    self.__adj_graph.add_vertex(vertex)
                    self.__adj_graph.add_edge(src, vertex)
                    self.__adj_graph.add_edge(dst, vertex)
                    self.__dummy_vertices.add(vertex)
                    dummies.append(vertex)
        return dummies

    def __compute_barycenter(self, src, level):
        mean = 0
        for dst in level:
            if self.__adj_graph.has_edge(src, dst) or self.__adj_graph.has_edge(dst, src):
                mean += self.__order_index[dst]
        self.__barycenter[src] = mean // len(level)

    def __compute_barycenter2(self, src, level):
        mean = 0
        for dst in level:
            if self.__adj_graph.has_edge(src, dst) or self.__adj_graph.has_edge(dst, src):
                mean += self.__x[dst]
        self.__barycenter[src] = mean // len(level)

    def __set_order_indexes(self, level):
        for index, vertex in enumerate(sorted(level, key=(lambda x: self.__barycenter[x]))):
            self.__order_index[vertex] = index

    def __barycentric_order(self, level1, level2):
        for vertex1 in level1:
            self.__compute_barycenter(vertex1, level2)
        for vertex2 in level2:
            self.__compute_barycenter(vertex2, level1)

    def __merge_lists(self, list1, list2, base_list):
        for vertex1 in list1:
            self.__compute_barycenter(vertex1, base_list)
        for vertex2 in list2:
            self.__compute_barycenter(vertex2, base_list)

        result = list1 + list2
        self.__set_order_indexes(result)

        return result

    def __remove_dummies(self, level):
        result = []
        for vertex in level:
            if vertex in self.__dummy_vertices:
                self.__adj_graph.remove_vertex(vertex)
            else:
                result.append(vertex)
        return result

    def __make_ordering_step(self, splitted, index, prev):
        dummies = self.__create_dummies(splitted[index]["middle"])
        merged = self.__merge_lists(splitted[prev]["middle"], dummies, splitted[index]["middle"])
        self.__barycentric_order(merged, splitted[index]["middle"])
        splitted[prev]["middle"] = self.__remove_dummies(merged)

        self.__set_order_indexes(splitted[prev]["middle"])
        self.__set_order_indexes(splitted[index]["middle"])

    def __order_local(self, node):
        if not node.children:
            return

        levels = self.__split_into_levels(node.origin)
        splitted = []
        for level in levels:
            self.__count_level_neighbours(level)
            for index, vertex in enumerate(sorted(level, key=self.__get_adj_difference)):
                self.__order_index[vertex] = n
            splitted.append(self.__minimize_closeness(level))

        for i in range(self.__order_iterations):
            init_dummies = self.__create_dummies(splitted[0]["middle"])
            if len(init_dummies) > 0:
                self.__barycentric_order(init_dummies, splitted[0]["middle"])
                self.__remove_dummies(init_dummies)

            for j in range(1, len(splitted)):
                self.__make_ordering_step(splitted, j, j - 1)

            init_dummies = self.__create_dummies(splitted[-1]["middle"])
            if len(init_dummies) > 0:
                self.__barycentric_order(init_dummies, splitted[-1]["middle"])
                self.__remove_dummies(init_dummies)

            for j in reversed(range(0, len(splitted) - 1)):
                self.__make_ordering_step(splitted, j, j + 1)

        for i in range(0, len(splitted)):
            for node in splitted[i]["middle"]:
                node.order_index += len(splitted[i]["left"])

    def __order_global(self, node):
        self.__order_local(node)
        for index, child in node.children:
            self.__order_global(child)

    #TODO: count edges for pairs of vertices?
    def __init_order_service_graph(self, node):
        for index, child in node.children:
            self.__init_order_service_graph(child)

        for dst, edge in self.__adj_graph.get_neighbours(node.origin):
            src_parent_node = self.__inc_tree_nodes[src].parent
            dst_parent_node = self.__inc_tree_nodes[dst].parent
            if src_parent_node != dst_parent_node:
                self.__order_service_graph.add_edge(src_parent_node, dst_parent_node)
                self.__order_service_graph.add_edge(dst_parent_node, src_parent_node)

    def __determine_vertex_order(self):
        self.__order_service_graph = Digraph.Digraph()
        self.__order_service_graph.add_vertices(self.__adj_graph.vertices)
        self.__init_order_service_graph(self.__inc_tree.root)

        self.__ordered_graph = Digraph.Digraph()
        self.__ordered_graph.add_vertices(self.__adj_graph.vertices)
        self.__order_global(self.__inc_tree.root)

    def __compute_connectivity(self, src, level):
        connectivity = 0
        for dst in level:
            if self.__adj_graph.has_edge(src, dst) or self.__adj_graph.has_edge(dst, src):
                connectivity += 1
        return connectivity

    def __split_level(self, level, top_vertex):
        left_part = []
        rigth_part = []
        for vertex in level:
            if self.__x[vertex] < self.__x[top_vertex]:
                left_part.append(vertex)
            else:
                rigth_part.append(vertex)
        return left_part, rigth_part

    def __improve_positions_r(self, level, left_bound, right_bound):
        if len(level) == 0:
            return []

        top_vertex = level.pop()
        left_part, right_part = self.__split_level(level, top_vertex)

        left_part_width = 0
        if len(left_part) > 0:
            left_part_width = sum(self.__width[vertex] + self.__d2 for vertex in left_part) - self.__d2
        right_part_width = 0
        if len(right_part) > 0:
            right_part_width = sum(self.__width[vertex] + self.__d2 for vertex in right_part) - self.__d2

        right_shift = min(self.__barycenter[top_vertex] - self.__x[top_vertex],
                          right_bound - right_part_width - self.__x[top_vertex] - self.__width[top_vertex] / 2)
        left_shift = min(self.__x[top_vertex] - self.__barycenter[top_vertex],
                         self.__x[top_vertex] - self.__width[top_vertex] / 2 - left_part_width - left_bound)

        if right_shift > 0:
            self.__x[top_vertex] += right_shift
            rightmost_x = self.__x[top_vertex] + self.__width[top_vertex] / 2
            for vertex in right_part:
                self.__x[vertex] = rightmost_x + self.__d2 + self.__width[vertex] / 2
                rightmost_x += self.__width[vertex]

        if left_shift > 0:
            self.__x[top_vertex] -= left_shift
            leftmost_x = self.__x[top_vertex] - self.__width[top_vertex] / 2
            for vertex in right_part:
                self.__x[vertex] = leftmost_x - self.__d2 - self.__width[vertex] / 2
                leftmost_x -= self.__width[vertex]

        return self.__improve_positions_r(left_part, left_bound, self.__x[top_vertex] - self.__width[top_vertex] - self.__d2) \
               + [top_vertex] + \
               self.__improve_positions_r(right_part, self.__x[top_vertex] + self.__width[top_vertex] + self.__d2, right_bound)

    def __improve_positions(self, levels, index, prev):
        sorted_level = list(sorted(levels[index], key=(lambda x: self.__connectivity[prev - index][x])))
        for vertex in sorted_level:
            self.__compute_barycenter2(vertex, levels[prev])
        return self.__improve_positions_r(sorted_level, -sys.maxsize, sys.maxsize)

    def __prm_method(self, vertex):
        if vertex in self.__dummy_vertices:
            self.__width[vertex] = 0
            return

        levels = self.__split_into_levels(vertex) #maybe it's better to use global levels list

        for i in range(0, len(levels)):
            levels[i] = list(sorted(levels[i], key=(lambda x: self.__order_index[x])))
            self.__x[levels[i][0]] = self.__width[levels[i][0]] / 2

            for j in range(1, len(levels[i])):
                self.__x[levels[i][j]] = self.__x[levels[i][j - 1]] + self.__width[levels[i][j - 1]] / 2 + self.__d2 + \
                                         self.__width[levels[i][j]] / 2

            for j in range(0, len(levels[i])):
                levels[i][j].connectivity = {}
                if i > 0:
                    self.__connectivity[-1][levels[i][j]] = self.__compute_connectivity(levels[i][j], levels[i - 1])
                if i < len(levels) - 1:
                    self.__connectivity[+1][levels[i][j]] = self.__compute_connectivity(levels[i][j], levels[i + 1])

        for i in range(1, len(levels)):
            levels[i] = self.__improve_positions(levels, i, i - 1)
        for i in reversed(range(0, len(levels) - 1)):
            levels[i] = self.__improve_positions(levels, i, i + 1)
        for i in range(1, len(levels)):
            levels[i] = self.__improve_positions(levels, i, i - 1)

        vertices = list(sum(levels, []))
        min_x = min(self.__x[vrtx] - self.__width[vrtx] / 2 for vrtx in vertices)
        max_x = max(self.__x[vrtx] + self.__width[vrtx] / 2 for vrtx in vertices)
        for vrtx in vertices:
            self.__x[vrtx] -= min_x - self.__d1
        self.__width[vertex] = max_x - min_x + 2 * self.__d1

    def __set_local_x_coords(self, vertex):
        if not self.__inc_graph.has_neighbours(vertex):
            if vertex.origin is None:
                vertex.width = 0
            else:
                vertex.width = self.__leaf_width
            return

        for dst, edge in self.__inc_graph.get_neighbours(vertex):
            self.__set_local_x_coords(dst)

        self.__prm_method(vertex)

    def __prepare_graph(self, graph):
        self.__inc_tree = Tree(digraph=graph.copy_inc_graph())
        self.__inc_tree_nodes = {}
        for node in self.__inc_tree.nodes:
            self.__inc_tree_nodes[node.origin] = node
        self.__adj_graph = graph.copy_adj_graph()

    def __restore_edge_directions(self):
        for src, dst, edge in self.__adj_graph.edges:
            if edge.inverted:
                edge.invert()
                self.__adj_graph.invert_edge(src, dst)

    def __build_compound_layer_tree(self, vertex, node):
        for dst, edge in self.__inc_graph.get_neighbours(vertex):
            index = self.__compound_layer[dst][-1]
            node.add_child(index)
            self.__compound_layer_tree_node[dst] = node[index]
            self.__build_compound_layer_tree(dst, node[index])

    def __set_y_coordinates_r(self, node):
        if len(node.children) == 0:
            node.height = self.__leaf_height
            return

        min_y = self.__d1
        max_y = min_y - self.__d2

        for value, child in node.children: #assume children in increasing compound layer order
            self.__set_y_coordinates_r(child)
            child.y = max_y + self.__d2 + child.height / 2
            max_y = max_y + self.__d2 + child.height
        node.height = max_y - min_y + 2 * self.__d1

    def __set_y_coordinates(self):
        self.__build_compound_layer_tree(self.__root_vertex, self.__compound_layer_tree)
        self.__set_y_coordinates_r(self.__compound_layer_tree)
        self.__compound_layer_tree.y = self.__compound_layer_tree.height / 2

    def __get_type(self, vertex):
        if vertex.origin is None:
            return DrawingFramework.EdgeType.ToDummy
        else:
            return DrawingFramework.EdgeType.ToReal

    def __layout(self, vertex, min_x, min_y):
        levels = self.__split_into_levels(vertex)
        for level in levels:
            for vrtx in level:
                node = self.__compound_layer_tree_node[vrtx]
                self.__drawer.draw_vertex(min_x + vrtx.x, min_y + node.y,
                                          vrtx.width, node.height)
                self.__layout(vrtx, min_x + vrtx.x - vrtx.width / 2, node.y - node.height / 2)

                for dst, edge in self.__adj_graph.get_neighbours(vrtx):
                    node_dst = self.__compound_layer_tree_node[dst]
                    self.__drawer.draw_edge(vrtx.x, node.y, vrtx.width, node.height,
                                            dst.x, dst.width, node_dst.y, node_dst.height,
                                            self.__get_type(dst))

    #we assume that inclusion graph is a rooted tree
    def draw(self, graph):
        self.__prepare_graph(graph)
        #we assume now there are no parent-child adjacency edges in the graph
        self.__assign_compound_layers()
        self.__normalize_graph()
        self.__determine_vertex_order()
        self.__restore_edge_directions()
        self.__set_local_x_coords(self.__root_vertex)
        self.__set_y_coordinates()
        self.__drawer.init()
        self.__layout(self.__root_vertex, 0, 0)
        self.__drawer.draw()

    def set_drawer_options(self, **kwargs):
        pass
