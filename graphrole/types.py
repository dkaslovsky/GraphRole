""" Define common types used in graphrole """

from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd


VectorLike = Union[np.array, pd.Series]
MatrixLike = Union[pd.DataFrame, np.ndarray]
DataFrameLike = Union[pd.DataFrame, pd.Series]

# node of a graph is labeled by a string or an int
Node = Union[int, str]
# edge of a graph is a 2-tuple of Nodes
Edge = Tuple[Node, Node]

# returned by pd.DataFrame.to_dict()
DataFrameDict = Dict[str, Dict[Node, float]]

FactorTuple = Tuple[np.ndarray, np.ndarray]
