#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tkinter import *


class EdgeType(object):
    ToReal = 0,
    ToDummy = 1


class DrawingFramework(object):
    def __init__(self):
        self.__canvas_parameters = {"background" : "White",
                                    "width" : 600,
                                    "height" : 600}
        self.__vertex_parameters = {"fill" : "White",
                                    "width" : 1,
                                    "outline" : "Black"}
        self.__edge_parameters = {"fill" : "Red",
                                  "width" : 1}
        self.__root = None
        self.__canvas = None
        self.__initialized = False

    # parameters:
    #
    # width - width (in pixels)
    # height - height (in pixels)
    # background - background color
    def set_canvas_parameters(self, **kwargs):
        self.__canvas_parameters = kwargs

    # parameters:
    #
    # fill - background color
    # width - border width (in pixels)
    # outline - border color
    def set_vertex_parameters(self, **kwargs):
        self.__vertex_parameters = kwargs

    # parameters:
    # width - line width
    # fill - line color
    def set_edge_parameters(self, **kwargs):
        self.__edge_parameters = kwargs

    def init(self):
        self.__root = Tk()
        self.__canvas = Canvas(self.__root, self.__canvas_parameters)
        self.__canvas.pack()
        self.__initialized = True

    def draw_vertex(self, x, y, width, height):
        x *= 10
        y *= 10
        width *= 10
        height *= 10
        coordinates = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
        self.__canvas.create_rectangle(coordinates, self.__vertex_parameters)

    def __draw_arrow(self, x1, y1, x2, y2):
        pass

    def draw_edge(self, x1, y1, width1, height1, x2, y2, width2, height2, type):
        if y1 < y2:
            src_y = y1 + height1 / 2
            dst_y = y2 - height2 / 2
        else:
            src_y = y1 - height1 / 2
            dst_y = y2 + height2  /2
        self.__canvas.create_line(x1, src_y, x2, dst_y, self.__edge_parameters)

        if type == EdgeType.ToReal:
            self.__draw_arrow(x1, src_y, x2, dst_y)

    def draw(self):
        self.__root.mainloop()
        self.__initialized = False
