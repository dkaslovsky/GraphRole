import matplotlib.pyplot as plt
import networkx as nx

from graphrole.features.extract import RecursiveFeatureExtractor
from graphrole.roles.roles import extract_role_factors


COLORS = [
    'red',
    'blue',
    'green',
]


if __name__ == '__main__':

    G = nx.florentine_families_graph()

    rfe = RecursiveFeatureExtractor(G)
    features = rfe.extract_features()

    print(f'Generation count = {rfe.generation_count}')
    print(features)

    n_roles = 3
    verbose = True
    features_sum_normalized = features.divide(features.sum(axis=0), axis=1)
    Gf, Ff = extract_role_factors(features_sum_normalized, n_roles=n_roles, verbose=verbose)

    node_roles = Gf.idxmax(axis=1)
    print(node_roles)

    role_colors = {f'role_{i}': COLORS[i] for i in range(Gf.shape[1])}
    node_colors = [role_colors[node_roles[node]] for node in G.nodes]

    plt.figure()
    nx.draw(G, with_labels=True, node_color=node_colors)
    plt.show()
