import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from graphrole import RecursiveFeatureExtractor, RoleExtractor


# pylint: disable=invalid-name

if __name__ == '__main__':

    # load the well known karate_club_graph from Networkx
    G = nx.karate_club_graph()

    # extract features
    feature_extractor = RecursiveFeatureExtractor(G)
    features = feature_extractor.extract_features()
    print(f'\nFeatures extracted from {feature_extractor.generation_count} recursive generations:')
    print(features)

    # assign node roles
    role_extractor = RoleExtractor(n_roles=None)
    role_extractor.extract_role_factors(features)
    node_roles = role_extractor.roles
    print('\nNode role assignments:')
    pprint(node_roles)
    print('\nNode role membership by percentage:')
    print(role_extractor.role_percentage.round(2))

    # build color palette for plotting
    unique_roles = sorted(set(node_roles.values()))
    color_map = sns.color_palette('Paired', n_colors=len(unique_roles))
    # map roles to colors
    role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
    # build list of colors for all nodes in G
    node_colors = [role_colors[node_roles[node]] for node in G.nodes]

    # plot graph
    plt.figure()
    with warnings.catch_warnings():
        # catch matplotlib deprecation warning
        warnings.simplefilter('ignore')
        nx.draw(
            G,
            pos=nx.spring_layout(G, seed=42),
            with_labels=True,
            node_color=node_colors,
        )
    plt.show()
