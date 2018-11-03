from typing import Any

from graphrole.graph import graph

INTERFACES = {
    'networkx': graph.NetworkxGraph
}


# NOTE: There are many ways of determining the module/package/type
#       of an object: inspect, type, isinstance.  Here we access
#       the __module__ property directly since isinstance returns
#       a bool which will not facilitate a dict lookup, type
#       does not support inheritance, and inspect.getmodule returns
#       a module ojbect that is specific to the subclass.  There is
#       likely a better approach and this should be futher investigated.
def get_interface(G: Any) -> graph.Graph:
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
        interface = INTERFACES[package]
    except KeyError:
        return None
    return interface(G)
