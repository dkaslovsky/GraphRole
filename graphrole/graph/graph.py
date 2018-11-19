from collections import defaultdict
from typing import Dict, Iterable, Iterator, Optional, Set

from graphrole.graph.interface import Edge, NodeName


class AdjacencyDictGraph:

    """
    Graph implementation using a mapping of node to the set of its neighbors to provide
    lightweight functionality for needed graph calculations
    """

    def __init__(self, edges: Iterable[Edge]) -> None:
        self.edges = edges
        self._build_adjacency_dict()

    def get_connected_components(self) -> Iterator[Set[NodeName]]:
        """
        Generate connected components represented as sets of nodes
        """
        # maintain set of all nodes already accounted for
        visited = set()
        # run dfs over all nodes not previously visited
        for node in self.adj_dict.keys():
            if node not in visited:
                component = self._dfs(node)
                visited.update(component)
                yield component

    def _build_adjacency_dict(self) -> Dict[NodeName, Set[NodeName]]:
        """
        Construct adjacency dict mapping node to a set of its neighbor nodes
        """
        adj = defaultdict(set)
        for (node1, node2) in self.edges:
            adj[node1].add(node2)
            adj[node2].add(node1)
        self.adj_dict = dict(adj)

    def _dfs(self, node: NodeName, visited: Optional[Set[NodeName]] = None) -> Set[NodeName]:
        """
        Run recursive depth first search starting from node and
        return set of all visited nodes
        :param node: node at which to start search
        :param visited: set of all nodes visited, initially should be None
        """
        if not visited:
            visited = set()
        visited.add(node)
        next_level_unvisited = self.adj_dict[node] - visited
        for nbr in next_level_unvisited:
            self._dfs(nbr, visited)
        return visited
