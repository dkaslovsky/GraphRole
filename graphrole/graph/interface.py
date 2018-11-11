from typing import List, NewType

import graphrole.graph.graph as g

INTERFACES = {
    'networkx': g.NetworkxGraph,
}


# Once more than one interface is supported, we can define a TypeVar
# constrained to the interfaces.  For now this will have to be a NewType.
#from typing import TypeVar
#GraphLib = TypeVar('GraphLib', *list(INTERFACES.keys()))
GraphLibInstance = NewType('GraphLibInstance', g.NetworkxGraph)


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
def get_interface(G: GraphLibInstance) -> g.GraphInterface:
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
