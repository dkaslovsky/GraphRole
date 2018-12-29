from collections import defaultdict
from typing import Dict, Iterable, Iterator, Set

from graphrole.types import Edge, Node


class AdjacencyDictGraph:

    """
    Graph implementation using a mapping of node to the set of its neighbors to provide
    lightweight functionality for needed graph calculations
    """

    def __init__(self, edges: Iterable[Edge]) -> None:
        self.edges = edges
        self.adj_dict = self._build_adjacency_dict()

    def get_connected_components(self) -> Iterator[Set[Node]]:
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

    def _build_adjacency_dict(self) -> Dict[Node, Set[Node]]:
        """
        Construct adjacency dict mapping node to a set of its neighbor nodes
        """
        adj = defaultdict(set)
        for (node1, node2) in self.edges:
            adj[node1].add(node2)
            adj[node2].add(node1)
        return dict(adj)

    def _dfs(self, node: Node) -> Set[Node]:
        """
        Run depth first search starting from node and return set of all visited nodes
        :param node: node at which to start search
        """
        visited = set()
        # use a list as a stack; pop() retrieves last element ensuring LIFO
        node_stack = [node]
        while node_stack:
            cur_node = node_stack.pop()
            if cur_node in visited:
                continue
            visited.add(cur_node)
            # add cur_node's non-visited neighors to the stack
            neighbors_to_visit = self.adj_dict[cur_node] - visited
            node_stack.extend(neighbors_to_visit)
        return visited
