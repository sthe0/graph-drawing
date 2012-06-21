#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import contextlib
from Digraph import *
import CompoundDigraph
from CompoundGraphDrawer import *


def read_graph(filename):
    graph = CompoundDigraph.CompoundDigraph()
    with contextlib.closing(open(filename)) as f:
        vertex_count = int(f.readline())

        for i in range(vertex_count):
            graph.add_vertex(Vertex(i))

        for line in f:
            row = tuple(map(int, line.rstrip("\n").split()))
            if row[2] == 0:
                graph.add_adj_edge(Vertex(row[0]), Vertex(row[1]))
            elif row[2] == 1:
                graph.add_inc_edge(Vertex(row[0]), Vertex(row[1]))
    return graph


def main():
    graph = read_graph("resource/graph0.txt")
    CompoundGraphDrawer().draw(graph)

    return 0


if __name__ == "__main__":
    sys.exit(main())
