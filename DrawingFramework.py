#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tkinter import *


class EdgeType(object):
    ToReal = 0,
    ToDummy = 1


class DrawingFramework(object):
    def __init__(self):
        self._canvas_options = {"background": "White",
                                "width": 1200,
                                "height": 700}
        self._vertex_options = {"fill": "White",
                                "width": 1,
                                "outline": "Black"}
        self._edge_options = {"fill": "Red",
                              "width": 1}
        self._header_options = {"fill": "LightSeaGreen",
                                "width": 0}
        self._font = ("Helvetica", "6")
        self._root = None
        self._canvas = None
        self._initialized = False

    # options:
    #
    # width - width (in pixels)
    # height - height (in pixels)
    # background - background color
    def set_canvas_options(self, **kwargs):
        self._canvas_options = kwargs

    # options:
    #
    # fill - background color
    # width - border width (in pixels)
    # outline - border color
    def set_vertex_options(self, **kwargs):
        self._vertex_options = kwargs

    # options:
    # width - line width
    # fill - line color
    def set_edge_options(self, **kwargs):
        self._edge_options = kwargs

    def init(self):
        self._root = Tk()
        scrollbar_y = Scrollbar(self._root)
        scrollbar_y.pack(side=RIGHT, fill=Y)
        scrollbar_x = Scrollbar(self._root, orient=HORIZONTAL)
        scrollbar_x.pack(side=BOTTOM, fill=X)
        self._canvas = Canvas(self._root, self._canvas_options)
        self._canvas.configure(scrollregion=self._canvas.bbox(ALL))
        self._canvas.config(yscrollcommand=scrollbar_y.set)
        self._canvas.config(xscrollcommand=scrollbar_x.set)
        scrollbar_y.config(command=self._canvas.yview)
        scrollbar_x.config(command=self._canvas.xview)
        self._canvas.pack()
        #        scrollbar = Scrollbar(self._canvas)
        #        scrollbar.pack(side=RIGHT, fill=X)
        self._initialized = True

    def draw_vertex(self, x, y, width, height, header_height=0, text=None):
        left_x = x - width / 2
        top_y = y - height / 2

        vertex_coordinates = (left_x, top_y, left_x + width, top_y + height)
        self._canvas.create_rectangle(vertex_coordinates, self._vertex_options)

        if header_height > 0:
            header_coordinates = (left_x + 1, top_y + 1, left_x + width, top_y + header_height)
            self._canvas.create_rectangle(header_coordinates, self._header_options)
            if text is not None:
                self._canvas.create_text(left_x + width / 2, top_y + header_height / 2,
                    font=self._font, text=text)

    def draw_fake_vertex(self, x, min_y, max_y):
        self._canvas.create_line(x, min_y, x, max_y, self._edge_options)

    def draw_edge(self, x1, y1, width1, height1, x2, y2, width2, height2, type):
        if y1 < y2:
            src_y = y1 + height1 / 2
            dst_y = y2 - height2 / 2
        else:
            src_y = y1 - height1 / 2
            dst_y = y2 + height2 / 2

        if type == EdgeType.ToReal:
            self._canvas.create_line(x1, src_y, x2, dst_y, self._edge_options, arrow=LAST,
                arrowshape=(8, 9, 3))
        else:
            self._canvas.create_line(x1, src_y, x2, dst_y, self._edge_options)

    def get_center(self):
        return self._canvas_options["width"] / 2, self._canvas_options["height"] / 2

    def draw(self):
        self._root.mainloop()
        self._initialized = False

