#!/usr/bin/env python
# -*- coding: utf-8 -*-

class CallAggregator(object):
    def __init__(self, main_function, aggregated_functions=(), kwargs={}, result_aggregator=None):
        self.__main_function = main_function
        self.__aggregated_functions = set(aggregated_functions)
        self.__result_aggregator = result_aggregator
        self.__static_args = args
        self.__static_kwargs = kwargs

    def registerFunction(self, function):
        self.__aggregated_functions.add(function)

    def unregisterFunction(self, function):
        self.__aggregated_functions.remove(function)

    def __call__(self, *args, **kwargs):
        result = set()
        kwargs.update(self.__static_kwargs)
        for function in self.__aggregated_functions:
            set.add(function(*args, **kwargs))
        return self.__result_aggregator(result) if self.__result_aggregator else result
