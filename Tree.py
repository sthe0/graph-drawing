#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

class TreeNode(object):
    def __init__(self, tree, parent=None, data=None, level=0):
        self.__tree = tree
        self.__parent = parent
        self.__level = level
        self.__children = {}
        self.data = data

    def get_child(self, index):
        return self.__children[index]

    def add_child(self, index, data=None):
        if self.__tree.finished:
            raise RuntimeError("Adding vertex to a finished tree.")

        if index in self.__children:
            return self.__children[index], False

        self.__children[index] = TreeNode(tree=self.__tree, data=data, level=self.__level + 1, parent=self)
        return self.__children[index], True

    def remove_child(self, index):
        if self.__tree.finished:
            raise RuntimeError("Removing vertex from a finished tree.")

        del self.__children[index]

    def __getitem__(self, index):
        return self.get_child(index)

    def __setitem__(self, index, data):
        return self.add_child(index, data)

    def __delitem__(self, index):
        self.remove_child(index)

    def get_parent(self):
        return self.__parent

    def get_children(self):
        return copy.copy(self.__children.items())

    def get_level(self):
        return self.__level

    def get_tree(self):
        return self.__tree

    parent = property(get_parent)
    children = property(get_children)
    level = property(get_level)
    tree = property(get_tree)

class Tree(object):
    def __init__(self, root_data=None, digraph=None):
        self.__finished = False
        self.__tin = {}
        self.__tout = {}
        self.__uplinks = {}
        self.__time = 0
        if digraph is not None:
            if not digraph.is_tree():
                raise ValueError("Specified digraph is not a tree.")
            root = digraph.topological_sort()[0]
            self.__root = TreeNode(tree=self, parent=None, level=0, data=root.data)
            self.__root.origin = root
            self.__init_from_digraph_r(self.__root, root, digraph)
            self.set_finished(True)
        else:
            self.__root = TreeNode(tree=self, parent=None, level=0, data=root_data)
            self.__nodes = {self.__root}

    def __init_from_digraph_r(self, node, vertex, digraph):
        for index, (dst, edge) in enumerate(digraph.get_neighbours(vertex)):
            node[index] = dst.data
            node[index].origin = dst
            self.__init_from_digraph_r(node[index], dst, digraph)

    def __finish_r(self, node):
        self.__tin[node] = self.__time
        self.__time += 1
        for index, child in node.children:
            self.__uplinks[child] = [child] + self.__uplinks[node]
            self.__finish_r(child)
        self.__tout[node] = self.__time
        self.__time += 1

    def set_finished(self, value):
        if self.__finished and value:
            return
        if not value:
            self.__finished = value
            return
        self.__time = 0
        self.__uplinks[] = {self.__root : [self.__root]}
        self.__finish_r(self.__root)
        self.__finished = True

    def get_finished(self):
        return self.__finished

    def is_ancestor_of(self, node1, node2):
        if not self.__finished:
            raise RuntimeError("Query on an unfinished tree.")
        return self.__tin[node1] < self.__tin[node2] and self.__tout[node2] < self.__tout[node1]

    def is_descendant_of(self, node1, node2):
        if not self.__finished:
            raise RuntimeError("Query on an unfinished tree.")
        return self.__tin[node1] > self.__tin[node2] and self.__tout[node2] > self.__tout[node1]

    def get_least_common_ancestor(self, node1, node2):
        if not self.__finished:
            raise RuntimeError("Query on an unfinished tree.")

        l, r = 0, len(self.__uplinks[node2]) - 1
        while l < r:
            m = (l + r) // 2
            if self.is_ancestor_of(self.__uplinks[node2][m], node1):
                r = m
            else:
                l = m + 1
        return self.__uplinks[node2][l]

    def get_same_level_vertices(self, node1, node2):
        if node1.level < node2.level:
            return node1, self.__uplinks[node2][node2.level - node1.level]
        else:
            return self.__uplinks[node1][node1.level - node2.level], node2

    def get_child_ancestor(self, node1, node2):
        if not self.is_ancestor_of(node1, node2):
            raise ValueError("Second argument must be a descendant of first one.")

        for index, child in node1.children:
            if self.is_ancestor_of(child, node2):
                return index, child

    def get_root(self):
        return self.__root

    def get_nodes(self):
        return copy.copy(self.__nodes)

    root = property(get_root)
    nodes = property(get_nodes)
    finished = property(get_finished, set_finished)
