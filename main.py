#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import contextlib
import Graph
import LayoutAlgorithm


def read_graph(filename):
    f = open(filename)
    vertex_count = int(f.getline())
    graph = Graph.CompoundDigraph(vertex_count)
    for line in f:
        row = line.rstrip("\n").split()
        if row[2] == 0:
            graph.edges.add_edge(row[0], row[1])
        elif row[2] == 1:
            graph.inc_edges.add_edge(row[0], row[1])
    return graph

def main(argv=None):
    if argv is None:
        argv = sys.argv

    graph = read_graph(Graph.CompoundDigraph(vertex_count))
    LayoutAlgorithm.lay_out(graph)

    return 0


if __name__ == "__main__":
    sys.exit(main())
