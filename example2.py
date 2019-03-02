import matplotlib.pyplot as plt
import networkx as nx

from graphrole.features.extract import RecursiveFeatureExtractor
from graphrole.roles.roles import extract_role_factors

#plt.ion()

COLORS = [
    'red',
    'blue',
    'green',
    'cyan',
    'magenta',
    'black',
    'white',
]


if __name__ == '__main__':

    #G = nx.florentine_families_graph()
    G = nx.karate_club_graph()

    rfe = RecursiveFeatureExtractor(G)
    features = rfe.extract_features()

    print(f'Generation count = {rfe.generation_count}')
    print(features)

    n_roles = None
    verbose = True
    #features = features.divide(features.sum(axis=0), axis=1)
    Gf, Ff = extract_role_factors(features, n_roles=n_roles, verbose=verbose)

    node_roles = Gf.idxmax(axis=1)
    print(node_roles)

    all_roles = sorted(list(set(node_roles.values)))
    print(all_roles)

    #role_colors = {f'role_{i}': COLORS[i] for i in range(Gf.shape[1])}
    role_colors = {role: COLORS[i] for i, role in enumerate(all_roles)}
    node_colors = [role_colors[node_roles[node]] for node in G.nodes]

    plt.figure()
    nx.draw(G, with_labels=True, node_color=node_colors)
    plt.show()
