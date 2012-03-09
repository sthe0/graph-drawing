#!/usr/bin/env python
# -*- coding: utf-8 -*-

class CallAggregator(object):
    def __init__(self, aggregated_functions=(), kwargs=None, result_aggregator=None):
        self.__aggregated_functions = set(aggregated_functions)
        self.__result_aggregator = result_aggregator
        self.__static_kwargs = {} if kwargs is None else kwargs

    def registerFunction(self, function):
        self.__aggregated_functions.add(function)

    def unregisterFunction(self, function):
        self.__aggregated_functions.remove(function)

    def unregisterAll(self):
        self.__aggregated_functions.clear()

    def __call__(self, *args, **kwargs):
        result = set()
        kwargs.update(self.__static_kwargs)
        for function in self.__aggregated_functions:
            one_result = function(*args, **kwargs)
            if one_result is not None:
                set.add(one_result)
        return self.__result_aggregator(result) if self.__result_aggregator is not None else result
