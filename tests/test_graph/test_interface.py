import unittest

import networkx as nx

from graphrole.graph.graph import Graph
from graphrole.graph.interface import get_interface


class TestGetInterface(unittest.TestCase):

    """ Unit tests for get_interface() """
    
    def test_get_interface(self):
        # test with object from supported library
        obj = nx.Graph()
        graph_interface = get_interface(obj)
        self.assertIsNotNone(graph_interface)
        self.assertIsInstance(graph_interface, Graph)
        # test with object not from supported library
        obj = str
        graph_interface = get_interface(obj)
        self.assertIsNone(graph_interface)
        # test with object with no module property
        obj = 'str'
        graph_interface = get_interface(obj)
        self.assertIsNone(graph_interface)
