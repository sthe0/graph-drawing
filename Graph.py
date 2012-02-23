#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Edge(object):
    pass


class Vertex(object):
    pass


class AdjacencyEdge(Edge):
    pass


class InclusionEdge(Edge):
    pass


class EdgeCollection(object):
    def __init__(self, edge_class, vertex_count):
        self.__container = [{} for _ in range(vertex_count)]
        self.__edge_class = edge_class

    def add_edge(self, x, y, args=()):
        self.__container[x][y] = self.__edge_class(*args)

    def remove_edge(self, x, y):
        del self.__container[x][y]

    def edge_exists(self, x, y):
        return y in self.__container[x]

    def get_edge(self, x, y):
        return self.__container[x][y]

    def invert_edge(self, x, y):
        tmp = self.__container[y][x] if self.edge_exists(y, x) else None
        self.__container[y][x] = self.__container[x][y]
        if tmp is not None: self.__container[x][y] = tmp


class Digraph(object):
    def __init__(self, vertex_count, vertex_class=Vertex, edge_class=Edge):
        self.__vertex_class = vertex_class
        self.__edge_class = edge_class
        self.__vertexes = tuple([self.__vertex_class() for _ in range(vertex_count)])
        self.__edges = EdgeCollection(edge_class, vertex_count)

    def get_edge_class(self):
        return self.__edge_class

    def get_vertex_class(self):
        return self.__vertex_class

    def get_edges(self):
        return self.__edges

    def get_vertexes(self):
        return self.__vertexes

    edge_class = property(get_edge_class)
    vertex_class = property(get_vertex_class)
    edges = property(get_edges)
    vertexes = property(get_vertexes)


class CompoundDigraph(Digraph):
    def __init__(self, vertex_count, vertex_class=Vertex, edge_class=Edge, inc_edge_class=Edge):
        super(CompoundDigraph, self).__init__(vertex_count, vertex_class, edge_class)
        self.__inc_edges = EdgeCollection(inc_edge_class, vertex_count)

    def get_inc_edges(self):
        return self.__inc_edges

    inc_edges = property(get_inc_edges)



x = CompoundDigraph(5)
x.edges.add_edge(2,3)
x.inc_edges.add_edge(1,2)
