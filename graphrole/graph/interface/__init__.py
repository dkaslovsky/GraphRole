from typing import List, TypeVar

from graphrole.graph.interface.base import BaseGraphInterface
from graphrole.graph.interface.networkx import NetworkxInterface

# IgraphInterface should not be imported if igraph is not installed
try:
    from graphrole.graph.interface.igraph import IgraphInterface
except ImportError:
    pass

INTERFACES = {
    'networkx': NetworkxInterface,
    # lazy eval in case IgraphInterface was not imported
    # pylint: disable=unnecessary-lambda
    'igraph': lambda G, **kwargs: IgraphInterface(G, **kwargs),
}


# define types
GraphInterfaceType = TypeVar('GraphInterfaceType', bound=BaseGraphInterface)
GraphLibInstance = TypeVar('GraphLibInstance', *list(INTERFACES.keys()))


def get_supported_graph_libraries() -> List[str]:
    """
    Return list of supported graph libraries
    """
    return list(INTERFACES.keys())


# NOTE: There are many ways of determining the module/package/type
#       of an object: inspect, type, isinstance.  Here we access
#       the __module__ property directly since isinstance returns
#       a bool which will not facilitate a dict lookup, type
#       does not support inheritance, and inspect.getmodule returns
#       a module ojbect that is specific to the subclass.  There is
#       likely a better approach and this should be futher investigated.
def get_interface(G: GraphLibInstance) -> GraphInterfaceType:
    """
    Return subclass of Graph initialized with G
    :param G: graph object from a supported graph libraries
    """
    try:
        module = G.__module__
        package = module.split('.')[0]
    except (AttributeError, IndexError):
        return None
    try:
        graph_interface = INTERFACES[package]
    except KeyError:
        return None
    return graph_interface
