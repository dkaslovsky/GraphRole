# GraphRole

[![Build Status](https://travis-ci.org/dkaslovsky/GraphRole.svg?branch=master)](https://travis-ci.org/dkaslovsky/GraphRole)
[![Coverage Status](https://coveralls.io/repos/github/dkaslovsky/GraphRole/badge.svg?branch=master)](https://coveralls.io/github/dkaslovsky/GraphRole?branch=master)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

Automatic feature extraction and node role assignment for transfer learning on graphs; based on the ReFeX/RolX algorithms [1, 2] of Henderson, et al.

### Overview
A fundamental problem for learning on graphs is extracting meaningful features.  `GraphRole` provides the `RecursiveFeatureExtractor` class to automate this process by extracting recursive features capturing local and neighborhood ("regional") structural properties of a given graph.  The specific implementation follows that of the ReFeX algorithm [1].  Node features (e.g., degree) and ego-net features (e.g., neighbors, number of internal vs. external edges) are extracted and then recursively aggregated over each node's neighbors' features until no additional information is encoded.  As is shown in [1], these recursive, "regional" features facilitate node classification and perform quite well in transfer learning tasks.

`GraphRole` also provides the `extract_role_factors` function for node role assignment (a form of classification).  Different nodes play different structural roles in a graph, and using recursive regional features, these roles can be identified and assigned to collections of nodes.  As they are structural in nature, node roles differ from and are often more intuitive than the commonly used communities of nodes.  In particular, roles can generalize across graphs whereas the notion of communities cannot [2].  Identification and assignment of node roles has been shown to facilitate many graph learning tasks.

Please see [1, 2] for more technical details.

### Installation
This package is not yet hosted on PyPI.  To install from source, clone this repo and run the `setup.py` script:
```
$ git clone https://github.com/dkaslovsky/GraphRole.git
$ cd GraphRole
$ python setup.py install
```

### Example Usage
Examples go here.


### Graph Interfaces
An interface for graph data structures is provided in the `graphrole.graph.interface` module.  Implementations for `networkx` and `igraph` are included.

The `igraph` package is not included in `requirements.txt` and thus will need to be manually installed
if desired.  This is due to additional installation requirements beyond `pip install python-igraph`; see
the [igraph documentation](https://igraph.org/python/#pyinstall) for more details.  Note that all tests
that require `igraph` will be skipped if it is not installed.

To add an implementation of an additional graph library or data structure:
1. Subclass the `BaseGraphInterface` ABC in `graphrole.graph.interface.base.py` and implement the required methods
1. Update the `INTERFACES` dict in `graphrole.graph.interface.__init__.py` to make the new subclass discoverable
1. Add tests by trivially implementing a `setUpClass()` classmethod of a subclass of `BaseGraphInterfaceTest.BaseGraphInterfaceTestCases` in the `tests.test_graph.test_interface.py` module
1. If desired, a similar procedure allows the feature extraction tests to be run using the added interface
by again trivially implementing a `setUpClass()` classmethod of a subclass of `BaseRecursiveFeatureExtractorTest.TestCases` in the `tests.test_features.test_extract.py` module

### W.I.P. Disclaimer
This package is still a work in progress.
Worth noting is that `requirements.txt` will likely be changing as development continues and this project may adopt the `pipenv` paradigm.

Features yet to be implemented include:
* Sense-making (model explanation) to the role extraction module
* Support directed and weighted graphs
* Support for reading graph from text file

### Tests
To run tests:
```
$ python -m unittest discover -v
```
As noted above, the tests for the `igraph` interface are skipped when `igraph` is not installed.  Because this package is intentionally not required, the  test coverage reported above is much lower than when `igraph` is installed and its interface tests are not skipped (__96% coverage__ to date).

### References
[1] Henderson, et al. [It’s Who You Know: Graph Mining Using Recursive Structural Features](http://www.cs.cmu.edu/~leili/pubs/henderson-kdd2011.pdf).

[2] Henderson, et al. [RolX: Structural Role Extraction & Mining in Large Graphs](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46591.pdf).
