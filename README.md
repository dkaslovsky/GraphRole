# GraphRole

Automatic feature extraction and node role assignment for transfer learning on graphs; based on the ReFeX/RolX algorithms [1, 2] of Henderson, et al.

### Overview
This package is still a work in progress.  It has only been tested with Python 3.7 at this point.
Also worth noting is that `requirements.txt` will likely be changing as development continues and
this project may adopt the `pipenv` paradigm.

Remaining features to add include:
* Sense-making (model explanation) to the role extraction module
* Tests for the role extraction module
* Support for reading graph from text file
* Support directed and weighted graphs
* Improve example.py
* Add standard packaging/setup/CI capabilities


### Graph Interfaces
An interface for graph data structures is provided in the `graphrole.graph.interface` module.  Implementations for `networkx` and `igraph` are included.

To add an implementation of an additional graph library or data structure:
1. Subclass the `BaseGraphInterface` ABC in `graphrole.graph.interface.base.py` and implement the required methods
1. Update the `INTERFACES` dict in `graphrole.graph.interface.__init__.py` to make the new subclass discoverable
1. Add tests by trivially implementing a `setUpClass()` classmethod of a subclass of `BaseGraphInterfaceTest.BaseGraphInterfaceTestCases` in the `tests.test_graph.test_interface.py` module

### References
[1] Henderson, et al. [It’s Who You Know: Graph Mining Using Recursive Structural Features](http://www.cs.cmu.edu/~leili/pubs/henderson-kdd2011.pdf).

[2] Henderson, et al. [RolX: Structural Role Extraction & Mining in Large Graphs](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46591.pdf).
