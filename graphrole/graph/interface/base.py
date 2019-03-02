from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd

from graphrole.types import Node


class BaseGraphInterface(ABC):

    """
    Abstract class to define the interface used to interact with various graph libraries
    """

    def get_neighborhood_features(self) -> pd.DataFrame:
        """
        Return neighborhood features (local + egonet) for each node in the graph
        """
        local = self._get_local_features()
        ego = self._get_egonet_features()
        features = (pd.concat([local, ego], axis=1, sort=True)
                    .sort_index())
        return features

    @abstractmethod
    def get_num_edges(self) -> int:
        """
        Return number of edges in the graph
        """
        pass

    @abstractmethod
    def get_nodes(self) -> Iterable[Node]:
        """
        Return iterable of nodes in the graph
        """
        pass

    @abstractmethod
    def get_neighbors(self, node: Node) -> Iterable[Node]:
        """
        Return iterable of neighbors of specified node
        """
        pass

    @abstractmethod
    def _get_local_features(self) -> pd.DataFrame:
        """
        Return local features for each node in the graph
        """
        pass

    @abstractmethod
    def _get_egonet_features(self) -> pd.DataFrame:
        """
        Return egonet features for each node in the graph
        """
        pass