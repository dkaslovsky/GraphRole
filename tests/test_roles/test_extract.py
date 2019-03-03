import unittest

import numpy as np
import pandas as pd

from graphrole.roles.extract import RoleExtractor

np.random.seed(0)


# pylint: disable=protected-access

class TestRoles(unittest.TestCase):

    """ Unit tests for role extraction """

    def setUp(self):
        self.re = RoleExtractor()
        self.n_nodes, self.n_features = 20, 30
        feature_names = [f'feature{i+1}' for i in range(self.n_features)]
        node_names = range(self.n_nodes)
        data = np.random.rand(self.n_nodes, self.n_features)
        self.features = pd.DataFrame(data, columns=feature_names, index=node_names)

    def test_init(self):
        # default args
        self.assertIsNone(self.re.n_roles)
        self.assertTupleEqual((self.re.min_roles, self.re.max_roles), self.re.N_ROLE_RANGE)
        self.assertTupleEqual((self.re.min_bits, self.re.max_bits), self.re.N_BIT_RANGE)
        # specify n_roles
        self.re = RoleExtractor(n_roles=5)
        self.assertEqual(self.re.n_roles, 5)
        # specify ranges
        self.re = RoleExtractor(n_role_range=(3, 5), n_bit_range=(2, 6))
        self.assertTupleEqual((self.re.min_roles, self.re.max_roles), (3, 5))
        self.assertTupleEqual((self.re.min_bits, self.re.max_bits), (2, 6))

    def test_extract_role_factors(self):
        # test with fixed n_roles; model selection tests will test n_roles=None
        for n_roles in range(2, 6):
            self.re = RoleExtractor(n_roles=n_roles)
            self.re.extract_role_factors(self.features)

            expected_features = set(self.features.columns)
            expected_nodes = set(self.features.index)
            expected_roles = {f'role_{i}' for i in range(n_roles)}

            self.assertEqual(self.re.node_role_factor.shape[1], n_roles)
            self.assertEqual(self.re.role_feature_factor.shape[0], n_roles)
            self.assertSetEqual(set(self.re.node_role_factor.index), expected_nodes)
            self.assertSetEqual(set(self.re.node_role_factor.columns), expected_roles)
            self.assertSetEqual(set(self.re.role_feature_factor.index), expected_roles)
            self.assertSetEqual(set(self.re.role_feature_factor.columns), expected_features)

    def test_roles(self):
        # return None when extract_role_factors has not yet been called
        roles = self.re.roles
        role_pct = self.re.role_percentage
        self.assertIsNone(roles)
        self.assertIsNone(role_pct)

        # extract role factors so roles and role_percentage should be populated
        n_roles = 3
        role_names = {f'role_{i}' for i in range(n_roles)}
        self.re = RoleExtractor(n_roles=n_roles)
        self.re.extract_role_factors(self.features)
        # test roles
        roles = self.re.roles
        self.assertSetEqual(set(roles.keys()), set(self.features.index))
        self.assertTrue(set(roles.values()).issubset(role_names))
        # test role_percentage
        role_pct = self.re.role_percentage
        self.assertSetEqual(set(role_pct.index), set(self.features.index))
        self.assertSetEqual(set(role_pct.columns), role_names)
        self.assertTrue(np.allclose(role_pct.sum(axis=1).values, np.ones((role_pct.shape[0], 1))))

    def test_explain(self):
        with self.assertRaises(NotImplementedError):
            self.re.explain()

    def test__select_model(self):
        self.re = RoleExtractor(n_role_range=(2, 5), n_bit_range=(2, 5))
        model = self.re._select_model(self.features)
        G, F = model
        expected_n_roles = 2  # model selection selects 2 for the generated data
        n_roles = G.shape[1]
        self.assertEqual(G.shape[1], F.shape[0])
        self.assertEqual(n_roles, expected_n_roles)

    def test__get_encoded_role_factors(self):
        min_shape = min(self.features.shape)
        for n_roles in range(2, 4):
            total_values = n_roles * min_shape
            for n_bits in range(1, 6):
                expected_uniq_vals = 2**n_bits
                if expected_uniq_vals <= total_values:
                    # enough samples exist to get role factors
                    G, F = self.re._get_encoded_role_factors(self.features, n_roles, n_bits)
                    uniq_vals_g = len(np.unique(G))
                    uniq_vals_f = len(np.unique(F))
                    self.assertTupleEqual(G.shape, (self.n_nodes, n_roles))
                    self.assertLessEqual(uniq_vals_g, expected_uniq_vals)
                    self.assertTupleEqual(F.shape, (n_roles, self.n_features))
                    self.assertLessEqual(uniq_vals_f, expected_uniq_vals)
                else:
                    with self.assertRaises(ValueError):
                        G, F = self.re._get_encoded_role_factors(self.features, n_roles, n_bits)

    def test__rescale_costs(self):
        costs = np.full((3, 3), np.nan)
        costs[1, 1] = np.random.rand()
        costs[2, :] = np.random.rand(1, costs.shape[1])
        rescaled_costs = self.re._rescale_costs(costs)
        # first row is still all nans
        self.assertTrue(np.isnan(rescaled_costs[0, :]).all())
        # second row is nans with 1.0 in position 1
        self.assertTrue(np.isnan(rescaled_costs[1, 0]))
        self.assertTrue(np.isnan(rescaled_costs[1, 2]))
        self.assertAlmostEqual(rescaled_costs[1, 1], 1.0)
        # third row has norm 1.0
        self.assertAlmostEqual(np.linalg.norm(rescaled_costs[2, :]), 1.0)
