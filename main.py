#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import contextlib
import Digraph
import CompoundDigraph
import CompoundGraphDrawer


def read_graph(filename):
    graph = CompoundDigraph.CompoundDiraph()
    f = open(filename)

    vertex_count = int(f.readline())

    for i in range(vertex_count):
        graph.add_vertex(IndexedVertex(i))

    for line in f:
        row = tuple(map(int, line.rstrip("\n").split()))
        edge = Digraph.Edge(IndexedVertex(row[0]), IndexedVertex(row[1]))
        if row[2] == 0:
            graph.add_adj_edge(row[0], row[1], edge)
        elif row[2] == 1:
            graph.add_inc_edge(row[0], row[1], edge)
    return graph


def main(argv=None):
    if argv is None:
        argv = sys.argv

    graph = read_graph("resource/graph.txt")
    CompoundGraphDrawer.CompoundGraphDrawer().draw(graph)

    return 0


if __name__ == "__main__":
    sys.exit(main())
