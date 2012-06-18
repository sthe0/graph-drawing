#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import Digraph
import DrawingFramework
from itertools import chain
from Tree import Tree
from RMQTree import RMQTree

class Relation:
    LT = 0
    LE = 1


class RelationEdge(Digraph.Edge):
    def __init__(self, relation):
        super(RelationEdge, self).__init__()
        self._relation = relation

    def update(self, relation):
        self._relation = max(self._relation, relation)

    def get_relation(self):
        return self._relation

    relation = property(get_relation)


class CompoundGraphDrawer(object):
    _order_iterations = 10

    #empty declarations
    #just declare object private variables
    #show variables interrelations
    def __init__(self, leaf_width=50, leaf_height=22, padding=5, header_height=15, margin=12):
        self._inc_tree = Tree()
        self._inc_tree_nodes = {}
        self._adj_graph = Digraph.Digraph()
        self._relation_graph = Digraph.Digraph()
        self._ordered_graph = Digraph.Digraph()
        self._compound_layer_tree = Tree()
        self._compound_layer_tree_nodes = {}
        self._inverted_edges = set()
        self._fake_vertices = set()
        self._dummy_vertices = set()
        self._adj_left = {}
        self._adj_right = {}
        self._order_service_graph = Digraph.Digraph()
        self._order_index = {}
        self._barycenter = {}
        self._x = {}
        self._y = {}
        self._width = {}
        self._height = {}
        self._leaf_width = leaf_width
        self._leaf_height = leaf_height
        self._left_top_x = 1
        self._left_top_y = 1
        self._header_height = header_height
        self._padding = padding
        self._margin = margin
        self._drawer = DrawingFramework.DrawingFramework()

    def _compare_compound_layers(self, node1, node2):
        least_common_ancestor = self._compound_layer_tree.get_least_common_ancestor(node1, node2)
        index1, ancestor1 = self._compound_layer_tree.get_child_ancestor(least_common_ancestor, node1)
        index2, ancestor2 = self._compound_layer_tree.get_child_ancestor(least_common_ancestor, node2)
        if ancestor1.data < ancestor2.data:
            return -1
        if ancestor1.data > ancestor2.data:
            return 1
        return 0

    def _half(self, x):
        return (x + 1) // 2

    def _derive_strict_relations(self, node):
        for dst, edge in self._adj_graph.get_neighbours(node.origin):
            node1, node2 = self._inc_tree.get_same_level_vertices(node, self._inc_tree_nodes[dst])
            self._relation_graph.add_edge(node1.origin, node2.origin, RelationEdge(Relation.LT))
        for index, child in node.children:
            self._derive_strict_relations(child)

    def _induce_graph(self, node_set):
        vertices = [node.origin for node in node_set]
        graph = Digraph.Digraph()
        graph.add_vertices(vertices)
        for src in vertices:
            for dst in vertices:
                if self._relation_graph.has_edge(src, dst):
                    graph.add_edge(src, dst)
        return graph

    def _make_acyclic(self, induced_graph):
        vertices = induced_graph.topological_sort()
        vertices.sort(key=(lambda x : len(induced_graph.get_neighbours(x))), reverse=True)
        vertex_indexes = {vertex : index for index, vertex in enumerate(vertices)}
        for src in vertices:
            for dst, edge in induced_graph.get_neighbours(src):
                if vertex_indexes[src] > vertex_indexes[dst]:
                    self._relation_graph.invert_edge(src, dst)

    def _derive_all_relations(self, node_set):
        if not node_set:
            return

        children = set()
        for node in node_set:
            for index, child in node.children:
                children.add(child)
        self._derive_all_relations(children)

        for src_node in node_set:
            for dst_node in node_set:
                if self._relation_graph.has_edge(src_node.origin, dst_node.origin):
                    src_parent_node = src_node.parent
                    dst_parent_node = dst_node.parent
                    if src_parent_node != dst_parent_node and \
                       not self._relation_graph.has_edge(src_parent_node.origin, dst_parent_node.origin):
                        self._relation_graph.add_edge(src_parent_node.origin,
                                                      dst_parent_node.origin,
                                                      RelationEdge(Relation.LE))

        self._make_acyclic(self._induce_graph(node_set))

    def _derive_relation_graph(self):
        self._relation_graph = Digraph.Digraph()
        self._relation_graph.add_vertices(self._adj_graph.vertices)
        self._derive_strict_relations(self._inc_tree.root)
        self._derive_all_relations(set(child for index, child in self._inc_tree.root.children))

	#TODO: Traverse vertices in the order of topological sort.
    def _assign_compound_layers_r(self, vertex_set):
        if len(vertex_set) == 0:
            return

        layer = {vertex:0 for vertex in vertex_set}
        for vertex in vertex_set:
            if not self._relation_graph.has_inverted_neighbours(vertex):
                layer[vertex] = 1
        while True:
            changed = False
            for src in vertex_set:
                if layer[src] >= 1:
                    for dst in vertex_set:
                        if self._relation_graph.has_edge(src, dst):
                            edge = self._relation_graph.get_edge(src, dst)
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
            parent = self._inc_tree_nodes[vertex].parent.origin
            new_node = self._compound_layer_tree_nodes[parent].add_child(layer[vertex], layer[vertex])[0]
            self._compound_layer_tree_nodes[vertex] = new_node
            for index, child in self._inc_tree_nodes[vertex].children:
                children.add(child.origin)

        self._assign_compound_layers_r(children)

    def _assign_compound_layers(self):
        self._derive_relation_graph()

        self._compound_layer_tree = Tree(root_data=1)
        self._compound_layer_tree_nodes[self._inc_tree.root.origin] = self._compound_layer_tree.root
        self._assign_compound_layers_r(set(child.origin for index, child in self._inc_tree.root.children))
        self._compound_layer_tree.finished = True
        self._inverted_edges = set()

        for src, dst, edge in self._adj_graph.edges:
            if self._compare_compound_layers(self._compound_layer_tree_nodes[src],
                                              self._compound_layer_tree_nodes[dst]) > 0:
                self._adj_graph.invert_edge(src, dst)
                self._inverted_edges.add(self._adj_graph.get_edge(dst, src))

    def _add_vertex(self, parent_node, cl_index):
        vertex = Digraph.Vertex()
        self._adj_graph.add_vertex(vertex)

        node = parent_node.add_child(len(parent_node.children))[0]
        node.origin = vertex
        self._inc_tree_nodes[vertex] = node

        cl_node = self._compound_layer_tree_nodes[parent_node.origin].add_child(cl_index)[0]
        cl_node.data = cl_index
        self._compound_layer_tree_nodes[vertex] = cl_node

        return node

    def _create_fake_vertex_adj_chain(self, src, dst, inverted):
        src_cl_node = self._compound_layer_tree_nodes[src]
        dst_cl_node = self._compound_layer_tree_nodes[dst]
        prev_vertex = src
        for i in range(src_cl_node.data + 1, dst_cl_node.data):
            next_vertex = self._add_vertex(src_cl_node.parent, i).origin
            self._add_fake_adj_edge(prev_vertex, next_vertex, inverted)
            prev_vertex = next_vertex
        self._add_fake_adj_edge(prev_vertex, dst, inverted)

    def _create_fake_vertex_inc_chain(self, inc_node, cl_node, adjustment):
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
            top_node = self._add_vertex(prev_node, indexes[0])
            self._fake_vertices.add(top_node.origin)
            prev_node = top_node

        for i in range(1, cl_node.level - inc_node.level):
            prev_node = self._add_vertex(prev_node, indexes[i])
            self._fake_vertices.add(prev_node.origin)

        return prev_node, top_node

    def _add_fake_adj_edge(self, src, dst, inverted):
        self._adj_graph.add_edge(src, dst)
        if inverted:
            self._inverted_edges.add(self._adj_graph.get_edge(src, dst))

    def _normalize_graph(self):
        cl_common_ancestors = {}
        common_ancestors = {}
        for src, dst, edge in self._adj_graph.edges:
            src_cl_node = self._compound_layer_tree_nodes[src]
            dst_cl_node = self._compound_layer_tree_nodes[dst]
            cl_key = (src_cl_node, dst_cl_node)
            cl_common_ancestors[cl_key] = self._compound_layer_tree.get_least_common_ancestor(src_cl_node, dst_cl_node)

            src_node = self._inc_tree_nodes[src]
            dst_node = self._inc_tree_nodes[dst]
            key = (src_node, dst_node)
            common_ancestors[key] = self._inc_tree.get_least_common_ancestor(src_node, dst_node)

        self._compound_layer_tree.finished = False
        self._inc_tree.finished = False

        for src, dst, edge in self._adj_graph.edges: #recall there are no parent-child adjacency edges!
            src_cl_node = self._compound_layer_tree_nodes[src]
            dst_cl_node = self._compound_layer_tree_nodes[dst]

            if src_cl_node.parent == dst_cl_node.parent and dst_cl_node.data - dst_cl_node.data == 1:
                continue

            inverted = edge in self._inverted_edges
            self._adj_graph.remove_edge(src, dst)

            src_node = self._inc_tree_nodes[src]
            dst_node = self._inc_tree_nodes[dst]
            src_top_node, dst_top_node = src_node, dst_node

            common_ancestor = common_ancestors[(src_node, dst_node)]
            cl_common_ancestor = cl_common_ancestors[(src_cl_node, dst_cl_node)]
            fake_common_ancestor = self._create_fake_vertex_inc_chain(common_ancestor, cl_common_ancestor, 0)[0]

            if src_cl_node.parent != cl_common_ancestor:
                src_neighbour_node, src_top_node = self._create_fake_vertex_inc_chain(fake_common_ancestor, src_cl_node, 1)
                self._add_fake_adj_edge(src_node.origin, src_neighbour_node.origin, inverted)
            if dst_cl_node.parent != cl_common_ancestor:
                dst_neighbour_node, dst_top_node = self._create_fake_vertex_inc_chain(fake_common_ancestor, dst_cl_node, -1)
                self._add_fake_adj_edge(dst_neighbour_node.origin, dst_node.origin, inverted)
            self._create_fake_vertex_adj_chain(src_top_node.origin, dst_top_node.origin, inverted)

        self._compound_layer_tree.finished = True
        self._inc_tree.finished = True

    def _get_adj_difference(self, vertex):
        return self._adj_right[vertex] - self._adj_left[vertex]

    def _split_into_levels(self, node):
        local_vertices = [child.origin for index, child in self._inc_tree_nodes[node].children]

        levels = {}
        for vertex in local_vertices:
            levels.setdefault(self._compound_layer_tree_nodes[vertex].data, []).append(vertex)

        return [value for key, value in levels.items()]

    def _count_level_neighbours(self, level):
        for vertex in level:
            self._adj_left[vertex] = 0
            self._adj_right[vertex] = 0
            for dst, edge in self._order_service_graph.get_neighbours(vertex):
                vertex_parent_node = self._inc_tree_nodes[vertex].parent
                dst_parent_node = self._inc_tree_nodes[dst].parent
                if vertex_parent_node != dst_parent_node:
                    if self._order_index[dst_parent_node.origin] < self._order_index[vertex_parent_node.origin]:
                        self._adj_left[vertex] += 1
                    else:
                        self._adj_right[vertex] += 1

    def _minimize_closeness(self, level):
        splitted = {"left" : [], "middle" : [], "right" : []}
        for vertex in level:
            if self._get_adj_difference(vertex) < 0:
                splitted["left"].append(vertex)
            elif self._get_adj_difference(vertex) == 0:
                splitted["middle"].append(vertex)
            else:
                splitted["right"].append(vertex)
        return splitted

    def _create_dummies(self, level):
        dummies = []
        edges = set()
        for src in level:
            for dst in level:
                if self._order_service_graph.has_edge(src, dst):
                    if (dst, src) in edges:
                        continue
                    edges.add((src, dst))
                    vertex = Digraph.Vertex()
                    self._adj_graph.add_vertex(vertex)
                    self._adj_graph.add_edge(src, vertex)
                    self._adj_graph.add_edge(dst, vertex)
                    self._dummy_vertices.add(vertex)
                    self._order_index[vertex] = len(dummies)
                    dummies.append(vertex)
        return dummies

    def _compute_barycenter(self, src, level):
        mean = 0
        neighbours = 0
        for index, dst in enumerate(level):
            if self._adj_graph.has_edge(src, dst) or self._adj_graph.has_edge(dst, src):
                mean += index
                neighbours += 1
        self._barycenter[src] = mean / max(neighbours, 1)

    def _compute_barycenter2(self, src, level):
        mean = 0
        neighbours = 0
        for dst in level:
            if self._adj_graph.has_edge(src, dst) or self._adj_graph.has_edge(dst, src):
                mean += self._x[dst]
                neighbours += 1
        self._barycenter[src] = mean // max(neighbours, 1)

    def _reorder(self, level):
        level.sort(key=(lambda x: self._barycenter[x]))

    def _barycentric_order(self, level1, level2):
        for vertex1 in level1:
            self._compute_barycenter(vertex1, level2)
        for vertex2 in level2:
            self._compute_barycenter(vertex2, level1)

	#TODO: compute barycenters for _sum_ of lists?
    def _merge_lists(self, list1, list2, base_list):
        for vertex1 in list1:
            self._compute_barycenter(vertex1, base_list)
        for vertex2 in list2:
            self._compute_barycenter(vertex2, base_list)

        result = list1 + list2
        self._reorder(result)

        return result

    def _remove_dummies(self, level):
        result = []
        for vertex in level:
            if vertex in self._dummy_vertices:
                del self._order_index[vertex]
                self._adj_graph.remove_vertex(vertex)
                self._dummy_vertices.remove(vertex)
            else:
                result.append(vertex)
        return result

    def _make_ordering_step(self, splitted, index, prev):
        dummies = self._create_dummies(splitted[index]["middle"])
        merged = self._merge_lists(splitted[prev]["middle"], dummies, splitted[index]["middle"])
        self._barycentric_order(merged, splitted[index]["middle"])
        splitted[prev]["middle"] = self._remove_dummies(merged)

        self._reorder(splitted[prev]["middle"])
        self._reorder(splitted[index]["middle"])

    def _order_local(self, node):
        if not node.children:
            return

        levels = self._split_into_levels(node.origin)
        splitted = []
        for level in levels:
            self._count_level_neighbours(level)
            splitted.append(self._minimize_closeness(level))

        for i in range(self._order_iterations):
            init_dummies = self._create_dummies(splitted[0]["middle"])
            if len(init_dummies) > 0:
                self._barycentric_order(init_dummies, splitted[0]["middle"])
                self._remove_dummies(init_dummies)
                self._reorder(splitted[0]["middle"])

            for j in range(1, len(splitted)):
                self._make_ordering_step(splitted, j, j - 1)

            init_dummies = self._create_dummies(splitted[-1]["middle"])
            if len(init_dummies) > 0:
                self._barycentric_order(init_dummies, splitted[-1]["middle"])
                self._remove_dummies(init_dummies)
                self._reorder(splitted[-1]["middle"])

            for j in reversed(range(0, len(splitted) - 1)):
                self._make_ordering_step(splitted, j, j + 1)

        for i in range(0, len(splitted)):
            for index, vertex in enumerate(splitted[i]["left"] + splitted[i]["middle"] + splitted[i]["right"]):
                self._order_index[vertex] = index

    def _order_global(self, node):
        self._order_local(node)
        for index, child in node.children:
            self._order_global(child)

    #TODO: count edges for pairs of vertices?
    def __init__order_service_graph(self, node):
        for index, child in node.children:
            self.__init__order_service_graph(child)

        for dst, edge in self._adj_graph.get_neighbours(node.origin):
            self._order_service_graph.add_edge(node.origin, dst)
            self._order_service_graph.add_edge(dst, node.origin)
            src_parent_node = self._inc_tree_nodes[node.origin].parent
            dst_parent_node = self._inc_tree_nodes[dst].parent
            if src_parent_node != dst_parent_node:
                self._order_service_graph.add_edge(src_parent_node.origin, dst_parent_node.origin)
                self._order_service_graph.add_edge(dst_parent_node.origin, src_parent_node.origin)

    def _determine_vertex_order(self):
        self._order_service_graph = Digraph.Digraph()
        self._order_service_graph.add_vertices(self._adj_graph.vertices)
        self.__init__order_service_graph(self._inc_tree.root)

        self._ordered_graph = Digraph.Digraph()
        self._ordered_graph.add_vertices(self._adj_graph.vertices)
        self._order_global(self._inc_tree.root)

    def _compute_connectivity(self, src, level):
        connectivity = 0
        for dst in level:
            if self._adj_graph.has_edge(src, dst) or self._adj_graph.has_edge(dst, src):
                connectivity += 1
        return connectivity

    def _split_level(self, level, top_vertex):
        left_part = []
        rigth_part = []
        for vertex in level:
            if self._order_index[vertex] < self._order_index[top_vertex]:
                left_part.append(vertex)
            elif self._order_index[vertex] > self._order_index[top_vertex]:
                rigth_part.append(vertex)
        return left_part, rigth_part

	#TODO: get top_vertex according to both connectivity and the index of vertices
    def _improve_positions_r(self, level, rmqtree, l, r, left_bound, right_bound):
        if l > r:
            return

        priority, index = rmqtree.rmax(l, r)
        top_vertex = level[index]

        left_part_width = 0
        if l < index:
            left_part_width = sum(self._width[vertex] + self._margin for vertex in level[l:index])
        right_part_width = 0
        if r > index:
            right_part_width = sum(self._margin + self._width[vertex] for vertex in level[index+1:r+1])

        right_shift = min(self._barycenter[top_vertex] - self._x[top_vertex],
                          right_bound - right_part_width - self._x[top_vertex] - self._half(self._width[top_vertex]))
        left_shift = min(self._x[top_vertex] - self._barycenter[top_vertex],
                         self._x[top_vertex] - self._half(self._width[top_vertex]) - left_part_width - left_bound)

        if right_shift > 0:
            self._x[top_vertex] += right_shift

        if left_shift > 0:
            self._x[top_vertex] -= left_shift

        self._improve_positions_r(level,
                                  rmqtree,
                                  l,
                                  index - 1,
                                  left_bound,
                                  self._x[top_vertex] - self._half(self._width[top_vertex]) - self._margin)
        self._improve_positions_r(level,
                                  rmqtree,
                                  index+1,
                                  r,
                                  self._x[top_vertex] + self._half(self._width[top_vertex]) + self._margin,
                                  right_bound)

    def _improve_positions(self, levels, index, prev):
        priorities = []
        for n, vertex in enumerate(levels[index]):
            self._compute_barycenter2(vertex, levels[prev])
            priorities.append((self._connectivity[vertex][prev - index], n))
        rmqtree = RMQTree(priorities)
        return self._improve_positions_r(levels[index], rmqtree, 0, len(levels[index]) - 1, -sys.maxsize, sys.maxsize)

	#TODO: set highes connectivity value to dummy vertices
    def _prm_method(self, vertex):
        if not self._inc_tree_nodes[vertex].children and vertex not in self._fake_vertices:
            self._width[vertex] = self._leaf_width
            return
        if vertex in self._fake_vertices:
            for index, child in self._inc_tree_nodes[vertex].children:
                self._x[child.origin] = 0
            self._width[vertex] = 0
            return

        levels = self._split_into_levels(vertex) #maybe it's better to use global levels list

        for i in range(0, len(levels)):
            levels[i] = list(sorted(levels[i], key=(lambda x: self._order_index[x])))
            self._x[levels[i][0]] = self._half(self._width[levels[i][0]])

            for j in range(1, len(levels[i])):
                self._x[levels[i][j]] = self._x[levels[i][j - 1]] + self._half(self._width[levels[i][j - 1]]) + \
                                        self._margin + self._half(self._width[levels[i][j]])

            for j in range(0, len(levels[i])):
                self._connectivity[levels[i][j]] = {}
                if i > 0:
                    self._connectivity[levels[i][j]][-1] = self._compute_connectivity(levels[i][j], levels[i - 1])
                if i < len(levels) - 1:
                    self._connectivity[levels[i][j]][+1] = self._compute_connectivity(levels[i][j], levels[i + 1])

        for i in range(1, len(levels)):
            self._improve_positions(levels, i, i - 1)
        for i in reversed(range(0, len(levels) - 1)):
            self._improve_positions(levels, i, i + 1)
        for i in range(1, len(levels)):
            self._improve_positions(levels, i, i - 1)

        vertices = list(sum(levels, []))
        min_x = min(self._x[vrtx] - self._half(self._width[vrtx]) for vrtx in vertices)
        max_x = max(self._x[vrtx] + self._half(self._width[vrtx]) for vrtx in vertices)
        for vertex_ in vertices:
            self._x[vertex_] -= min_x - self._padding
        self._width[vertex] = max_x - min_x + 2 * self._padding

    def _set_local_x_coordinates(self, node):
        for index, child in node.children:
            self._set_local_x_coordinates(child)

        self._prm_method(node.origin)

    def _prepare_graph(self, graph):
        self._connectivity = {}
        self._fake_vertices = set()
        self._dummy_vertices = set()
        self._inc_tree = Tree(digraph=graph.copy_inc_graph())
        self._inc_tree_nodes = {}
        for node in self._inc_tree.nodes:
            self._inc_tree_nodes[node.origin] = node
        self._adj_graph = graph.copy_adj_graph()

    def _restore_edge_directions(self):
        for src, dst, edge in self._adj_graph.edges:
            if edge in self._inverted_edges:
                self._adj_graph.invert_edge(src, dst)

    def _set_fake_compound_layer_nodes(self):
        for node in self._compound_layer_tree.nodes:
            node.fake = True
        for vertex in self._adj_graph.vertices:
            if vertex not in self._fake_vertices:
                self._compound_layer_tree_nodes[vertex].fake = False

    def _set_y_coordinates(self, cl_node, min_y):
        if not cl_node.children:
            if not cl_node.fake:
                self._height[cl_node] = self._leaf_height + self._header_height
            else:
                self._height[cl_node] = 0
            return

        min_y += self._padding
        max_y = min_y - self._margin + self._header_height

        for index, cl_child in sorted(cl_node.children, key=(lambda pair: pair[0])):
            self._set_y_coordinates(cl_child, max_y + self._margin)
            self._y[cl_child] = max_y + self._margin + self._half(self._height[cl_child])
            max_y += self._margin + self._height[cl_child]

        self._height[cl_node] = max_y - min_y + 2 * self._padding

    def _get_type(self, vertex):
        if vertex in self._fake_vertices:
            return DrawingFramework.EdgeType.ToDummy
        else:
            return DrawingFramework.EdgeType.ToReal

    def _set_coordinates(self, node, min_x):
        levels = self._split_into_levels(node.origin)
        for level in levels:
            for vertex in level:
                vertex_node = self._inc_tree_nodes[vertex]
                self._x[vertex] += min_x
                self._y[vertex] = self._y[self._compound_layer_tree_nodes[vertex]]
                self._height[vertex] = self._height[self._compound_layer_tree_nodes[vertex]]
                self._set_coordinates(vertex_node, self._x[vertex] - self._half(self._width[vertex]))

    def _determine_bounds(self, node):
        min_y = sys.maxsize
        max_y = -sys.maxsize

        for index, child in node.children:
            child_min_y, child_max_y = self._determine_bounds(child)
            min_y = min(min_y, child_min_y)
            max_y = max(max_y, child_max_y)

        src = node.origin
        for dst, neighbour in chain(self._adj_graph.get_neighbours(src), self._adj_graph.get_inverted_neighbours(src)):
            src_cl_node = self._compound_layer_tree_nodes[src]
            dst_cl_node = self._compound_layer_tree_nodes[dst]
            if dst_cl_node.data < src_cl_node.data:
                min_y = min(min_y, self._y[src] - self._height[src] / 2)
            elif dst_cl_node.data > src_cl_node.data:
                max_y = max(max_y, self._y[src] + self._height[src] / 2)

        return min_y, max_y

    def _layout_fake_vertex(self, vertex):
        node = self._inc_tree_nodes[vertex]
        if node.parent.origin in self._fake_vertices:
            return

        min_y, max_y = self._determine_bounds(node)
        self._drawer.draw_fake_vertex(self._x[vertex], min_y, max_y)

    def _layout_vertices(self):
        for vertex in self._adj_graph.vertices:
            if vertex in self._fake_vertices:
                self._layout_fake_vertex(vertex)
            else:
                self._drawer.draw_vertex(self._x[vertex],
                                         self._y[vertex],
                                         self._width[vertex],
                                         self._height[vertex],
                                         self._header_height,
                                         str(vertex.id))

    def _layout_edges(self):
        for src, dst, edge in self._adj_graph.edges:
            self._drawer.draw_edge(self._x[src], self._y[src], self._width[src], self._height[src],
                                   self._x[dst], self._y[dst], self._width[dst], self._height[dst],
                                   self._get_type(dst))

    #we assume that inclusion graph is a rooted tree
    def draw(self, graph):
        min_x = 5
        min_y = 5
        self._prepare_graph(graph)
        #we assume now there are no parent-child adjacency edges in the graph
        self._assign_compound_layers()
        self._normalize_graph()
        self._determine_vertex_order()
        self._restore_edge_directions()
        self._set_local_x_coordinates(self._inc_tree.root)
        self._set_fake_compound_layer_nodes()
        self._set_y_coordinates(self._compound_layer_tree.root, min_x)
        self._set_coordinates(self._inc_tree.root, min_y)
        self._x[self._inc_tree.root.origin] = self._half(self._width[self._inc_tree.root.origin]) + min_x
        self._y[self._inc_tree.root.origin] = self._half(self._height[self._compound_layer_tree.root]) + min_y
        self._height[self._inc_tree.root.origin] = self._height[self._compound_layer_tree.root]
        self._drawer.init()
        self._layout_vertices()
        self._layout_edges()
        self._drawer.draw()

    def set_drawer_options(self, **kwargs):
        pass
