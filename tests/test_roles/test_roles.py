import unittest

import numpy as np
import pandas as pd

import graphrole.roles.roles as rl


np.random.seed(0)

class TestRoles(unittest.TestCase):

    """ Unit tests for role extraction """

    def setUp(self):
        self.n_nodes, self.n_features = 20, 30
        feature_names = [f'feature{i+1}' for i in range(self.n_features)]
        node_names = range(self.n_nodes)
        data = np.random.rand(self.n_nodes, self.n_features)
        self.features = pd.DataFrame(data, columns=feature_names, index=node_names)

    def test_extract_role_factors(self):
        G, F = rl.extract_role_factors(self.features)
        n_roles = G.shape[1]
        self.assertEqual(G.shape[1], F.shape[0])
        expected_features = set(self.features.columns)
        expected_nodes = set(self.features.index)
        expected_roles = {f'role_{i}' for i in range(n_roles)}
        self.assertSetEqual(set(G.index), expected_nodes)
        self.assertSetEqual(set(G.columns), expected_roles)
        self.assertSetEqual(set(F.index), expected_roles)
        self.assertSetEqual(set(F.columns), expected_features)
    
    def test_get_role_ndarrays(self):
        table = {
            'specified n_roles': {
                'specified_n_roles': 4,
                'expected_n_roles': 4,
            },
            'unspecified n_roles': {
                'specified_n_roles': None,
                'expected_n_roles': 2,  # the model selection selects 2 for the generated data
            },
        }
        for test_name, test in table.items():
            specified_n_roles = test['specified_n_roles']
            expected_n_roles = test['expected_n_roles']
            G, F = rl.get_role_ndarrays(self.features, n_roles=specified_n_roles, verbose=False)
            n_roles = G.shape[1]
            self.assertEqual(G.shape[1], F.shape[0], test_name)
            self.assertEqual(n_roles, expected_n_roles, test_name)
    
    def test_get_role_factors(self):
        min_shape = min(self.features.shape)
        for n_roles in range(2, 4):
            for n_bits in range(1, 7):
                expected_uniq_vals = 2**n_bits
                # ValueError is raised if not enough samples exist to encode with n_bits
                if expected_uniq_vals > n_roles * min_shape:
                    with self.assertRaises(ValueError):
                        G, F = rl.get_role_factors(self.features, n_roles, n_bits)
                    continue
                # enough samples exist to get role factors
                G, F = rl.get_role_factors(self.features, n_roles, n_bits)
                uniq_vals_g = len(np.unique(G))
                uniq_vals_f = len(np.unique(F))
                self.assertTupleEqual(G.shape, (self.n_nodes, n_roles))
                self.assertLessEqual(uniq_vals_g, expected_uniq_vals)
                self.assertTupleEqual(F.shape, (n_roles, self.n_features))
                self.assertLessEqual(uniq_vals_f, expected_uniq_vals)

    def test_get_nmf_factors(self):
        for n_roles in range(2, 4):
            G, F = rl.get_nmf_factors(self.features.values, n_roles)
            self.assertTupleEqual(G.shape, (self.n_nodes, n_roles))
            self.assertTupleEqual(F.shape, (n_roles, self.n_features))

    def test_encode(self):
        for n_bins in range(1, 7):
            data = self.features.values
            encoded = rl.encode(data, n_bins)
            uniq_vals = np.unique(encoded)
            self.assertLessEqual(len(uniq_vals), n_bins)

    def test_get_description_length(self):
        for n_roles in range(2, 4):
            for n_bits in range(1, 6):
                code = (
                    np.random.rand(self.n_nodes, n_roles),
                    np.random.rand(n_roles, self.n_features)
                )
                cost_tuple = rl.get_description_length(self.features, code, n_bits)
                self.assertEqual(len(cost_tuple), 3)
                (tot_cost, enc_cost, err_cost) = cost_tuple
                self.assertGreaterEqual(tot_cost, 0)
                self.assertGreaterEqual(enc_cost, 0)
                self.assertGreaterEqual(err_cost, 0)
                self.assertEqual(tot_cost, enc_cost + err_cost)

    def test_get_error_cost(self):
        original = self.features.values
        # assert positive error cost
        approx = abs(original - np.random.randn(*original.shape))
        err = rl.get_error_cost(original, approx)
        self.assertGreater(err, 0)
        # assert zero error cost for same input
        approx = original
        err = rl.get_error_cost(original, approx)
        self.assertEqual(err, 0)
