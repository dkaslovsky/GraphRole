import networkx as nx
import pandas as pd

# TODO: directed and weighted versions of these functions
# TODO: provide networkx abstraction


def get_local_features(G: nx.Graph) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        dict(G.degree),
        orient='index',
        columns=['degree']
    )


def get_egonet_features(G: nx.Graph) -> pd.DataFrame:
    egonet_features = {}
    for node in G.nodes:
        ego = nx.ego_graph(G, node, radius=1)
        features = {
            'n_internal_edges': len(ego.edges),
            'n_external_edges': len(list(nx.edge_boundary(G, ego.nodes)))
        }
        egonet_features[node] = features        
    return pd.DataFrame.from_dict(egonet_features, orient='index')


def get_neighborhood_features(G: nx.Graph) -> pd.DataFrame:
    local = get_local_features(G)
    ego = get_egonet_features(G)
    df = pd.concat([local, ego], axis=1)
    df.index.rename('node', inplace=True)
    return df.sort_index()
