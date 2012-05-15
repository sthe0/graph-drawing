#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

class TreeNode(object):
    def __init__(self, tree, parent=None, data=None, level=0):
        self._tree = tree
        self._parent = parent
        self._level = level
        self._children = {}
        self.data = data

    def get_child(self, index):
        return self._children[index]

    def add_child(self, index, data=None):
        if self._tree.finished:
            raise RuntimeError("Adding vertex to a finished tree.")

        if index in self._children:
            return self._children[index], False

        self._children[index] = TreeNode(tree=self._tree, data=data, level=self._level + 1, parent=self)
        self._tree._nodes.add(self._children[index])
        return self._children[index], True

    def remove_child(self, index):
        if self._tree.finished:
            raise RuntimeError("Removing vertex from a finished tree.")

        self._tree._nodes.remove(self._children[index])
        del self._children[index]

    def __getitem__(self, index):
        return self.get_child(index)

    def __delitem__(self, index):
        self.remove_child(index)

    def get_parent(self):
        return self._parent

    def get_children(self):
        return self._children.items()

    def get_level(self):
        return self._level

    def get_tree(self):
        return self._tree

    parent = property(get_parent)
    children = property(get_children)
    level = property(get_level)
    tree = property(get_tree)

class Tree(object):
    def __init__(self, root_data=None, digraph=None):
        self._finished = False
        self._tin = {}
        self._tout = {}
        self._uplinks = {}
        self._time = 0
        if digraph is not None:
            if not digraph.is_tree():
                raise ValueError("Specified digraph is not a tree.")
            root = digraph.topological_sort()[0]
            self._root = TreeNode(tree=self, parent=None, level=0, data=None)
            self._root.origin = root
            self._nodes = {self._root}
            self._init_from_digraph_r(self._root, root, digraph)
            self.set_finished(True)
        else:
            self._root = TreeNode(tree=self, parent=None, level=0, data=root_data)
            self._nodes = {self._root}

    def _init_from_digraph_r(self, node, vertex, digraph):
        for index, (dst, edge) in enumerate(digraph.get_neighbours(vertex)):
            node.add_child(index)
            node[index].origin = dst
            self._init_from_digraph_r(node[index], dst, digraph)

    def _finish_r(self, node):
        self._tin[node] = self._time
        self._time += 1
        for index, child in node.children:
            self._uplinks[child] = [child] + self._uplinks[node]
            self._finish_r(child)
        self._tout[node] = self._time
        self._time += 1

    def set_finished(self, value):
        if self._finished and value:
            return
        if not value:
            self._finished = value
            return
        self._time = 0
        self._uplinks = {self._root : [self._root]}
        self._finish_r(self._root)
        self._finished = True

    def get_finished(self):
        return self._finished

    def is_ancestor_of(self, node1, node2):
        if not self._finished:
            raise RuntimeError("Query on an unfinished tree.")
        return self._tin[node1] < self._tin[node2] and self._tout[node2] < self._tout[node1]

    def is_descendant_of(self, node1, node2):
        if not self._finished:
            raise RuntimeError("Query on an unfinished tree.")
        return self._tin[node1] > self._tin[node2] and self._tout[node2] > self._tout[node1]

    def get_least_common_ancestor(self, node1, node2):
        if not self._finished:
            raise RuntimeError("Query on an unfinished tree.")

        l, r = 0, len(self._uplinks[node2]) - 1
        while l < r:
            m = (l + r) // 2
            if self.is_ancestor_of(self._uplinks[node2][m], node1):
                r = m
            else:
                l = m + 1
        return self._uplinks[node2][l]

    def get_same_level_vertices(self, node1, node2):
        if node1.level < node2.level:
            return node1, self._uplinks[node2][node2.level - node1.level]
        else:
            return self._uplinks[node1][node1.level - node2.level], node2

    def get_child_ancestor(self, node1, node2):
        if not self.is_ancestor_of(node1, node2):
            raise ValueError("Second argument must be a descendant of first one.")

        for index, child in node1.children:
            if self.is_ancestor_of(child, node2) or child == node2:
                return index, child

    def get_root(self):
        return self._root

    def get_nodes(self):
        return copy.copy(self._nodes)

    root = property(get_root)
    nodes = property(get_nodes)
    finished = property(get_finished, set_finished)
