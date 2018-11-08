# GraphRole

Much more to come...

So Far
- Recursive Feature Extraction (ReFeX) for Graphs
- See: [henderson-kdd2011.pdf](http://www.cs.cmu.edu/~leili/pubs/henderson-kdd2011.pdf)
- Interface for Networkx graphs; to add support for any other graph library:
  - Subclass Graph ABC in `graphrole.graph.graph.py`
  - Update dict of supported interfaces in `graphrole.graph.interface.py`
  - Add tests via trivial subclass in `tests.test_graph.test_graph.py`
