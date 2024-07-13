import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# from examples.graph import load_example_graph
from data import load_nx_karate_club_graph
from graphrole import RecursiveFeatureExtractor, RoleExtractor

# pylint: disable=invalid-name

if __name__ == '__main__':

    # This example script demonstrates GraphRole's main functionality. It matches the examples.ipynb notebook.
    #
    # Notes:
    # - The example workflow was built prior to GraphRole's ability to use edge weights in its feature calculations.
    #   To keep consistent results, the example graph is loaded without edge weights. To include edge weights
    #   when loading the graph, change the weighted kwarg to True. GraphRole will automatically include
    #   these weights in its feature calculations and results will therefore differ from below.
    #
    # - The role calculations are slightly sensitive to the versions of scipy and scikit-learn used
    #   for matrix factorization (NMF) and clustering (K-Means). The results below are obtained using
    #   scipy==1.10.1 and scikit-learn==1.2.2, matching the original implementation of GraphRole. More
    #   granular insight into previous version requirements can be found in the git history of
    #   GraphRole's requirements.txt file.

    # load the well known karate_club_graph from Networkx
    G = load_nx_karate_club_graph(weighted=False)

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
