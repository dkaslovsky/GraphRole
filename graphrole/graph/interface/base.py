from abc import ABC, abstractmethod
from typing import Iterable, List

import pandas as pd

from graphrole.types import Node


class BaseGraphInterface(ABC):

    """
    Abstract class to define the interface used to interact with various graph libraries
    """

    # prefix appended to attribute names to form the resulting feature name
    attribute_feature_prefix = 'attribute'

    def get_neighborhood_features(self) -> pd.DataFrame:
        """
        Return neighborhood features (local + egonet) for each node in the graph
        """
        local = self._get_local_features()
        ego = self._get_egonet_features()
        features = (pd.concat([local, ego], axis=1, sort=True)
                    .sort_index())
        return features

    def _set_attribute_kwargs(self, **kwargs) -> None:
        """
        Parses kwargs to set attribute parameters
        :kwarg attributes: boolean indicating whether to use node attributes as features
        :kwarg attributes_include: include list of node attributes for features
          (all attributes are used if not specified)
        :kwarg attributes_exclude: exclude list of node attributes for features
          (overrides attributes_include in cases of conflict)
        """
        self._attrs: bool = kwargs.get('attributes', False)
        self._attrs_include: List[str] = kwargs.get('attributes_include', [])
        self._attrs_exclude: List[str] = kwargs.get('attributes_exclude', [])

    @classmethod
    def _attribute_feature_name(cls, attr_name: str) -> str:
        """
        Forms a name used for attribute features by appending attribute_prefix to the attribute's
        name, avoiding any potential collision with predefined features
        :param attr_name: attribute name to prepend with prefix
        """
        return f'{cls.attribute_feature_prefix}_{attr_name}'

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
