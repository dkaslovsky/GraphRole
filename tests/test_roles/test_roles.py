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

    def test_get_role_factors(self):
        for n_roles in range(2, 4):
            for n_bits in range(1, 7):
                expected_uniq_vals = 2**n_bits
                G, F = rl.get_role_factors(self.features, n_roles, n_bits)
                uniq_vals_g = len(np.unique(G))
                uniq_vals_F = len(np.unique(F))
                self.assertTupleEqual(G.shape, (self.n_nodes, n_roles))
                self.assertEqual(uniq_vals_g, expected_uniq_vals)
                self.assertTupleEqual(F.shape, (n_roles, self.n_features))
                self.assertEqual(uniq_vals_F, expected_uniq_vals)

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
            self.assertEqual(len(uniq_vals), n_bins)

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
