import unittest

from graphrole.graph.graph import AdjacencyDictGraph

# pylint: disable=protected-access

class TestAdjacencyDictGraph(unittest.TestCase):

    """ Unit tests for AdjacencyDictGraph """

    test_graph_table = {
        'disjoint nodes': {
            'edges':        [(0, 1), (2, 3)],
            'adjacency':    {0: {1}, 1: {0}, 2: {3}, 3: {2}},
            'components':   [{0, 1}, {2, 3}]
        },
        'cycle': {
            'edges':        [(0, 1), (1, 2), (2, 0)],
            'adjacency':    {0: {1, 2}, 1: {0, 2}, 2: {0, 1}},
            'components':   [{0, 1, 2}]
        },
        'two components': {
            'edges':        [(0, 7), (0, 8), (8, 2), (8, 5), (1, 3), (6, 2), (6, 4)],
            'adjacency':    {0: {7, 8}, 1: {3}, 2: {6, 8}, 3: {1}, 4: {6},
                             5: {8}, 6: {2, 4}, 7: {0}, 8: {0, 2, 5}},
            'components':   [{0, 2, 4, 5, 6, 7, 8}, {1, 3}]
        },
        'dangling': {
            'edges':        [(0, 0), (1, 2)],
            'adjacency':    {0: {0}, 1: {2}, 2: {1}},
            'components':   [{0}, {1, 2}]
        }
    }

    def test_constructor(self):
        for test_name, test in self.test_graph_table.items():
            graph = AdjacencyDictGraph(test['edges'])
            self.assertDictEqual(graph.adj_dict, test['adjacency'], test_name)

    def test_get_connected_components(self):
        for test_name, test in self.test_graph_table.items():
            components = AdjacencyDictGraph(test['edges']).get_connected_components()
            expected = test['components']
            for component in components:
                self.assertIn(component, expected, test_name)

    def test__dfs(self):
        for test_name, test in self.test_graph_table.items():
            graph = AdjacencyDictGraph(test['edges'])
            for component in test['components']:
                for node in component:
                    result_component = graph._dfs(node)
                    self.assertSetEqual(result_component, component, test_name)
