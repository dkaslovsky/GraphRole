from typing import Optional, Tuple

import pandas as pd
from sklearn.decomposition import NMF


def extract_role_factors(
    features: pd.DataFrame,
    n_roles: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return node-role and role-feature factor DataFrames
    :param features: DataFrame of features
    :param n_roles: optional number of roles to extract
    """
    n_roles = n_roles if n_roles is not None else get_num_roles(features)

    nmf = NMF(n_components=n_roles, solver='mu')
    role_labels = [f'role_{i}' for i in range(n_roles)]

    node_role_array = nmf.fit_transform(features)
    role_feature_array = nmf.components_

    node_role_df = pd.DataFrame(
        node_role_array,
        index=features.index,
        columns=role_labels
    )
    role_feature_df = pd.DataFrame(
        role_feature_array,
        index=role_labels,
        columns=features.columns
    )
    return node_role_df, role_feature_df


def get_num_roles(features: pd.DataFrame) -> int:
    """
    Return estimate of number of node roles present based on node features
    :param features: DataFrame of features
    """
    return 2
