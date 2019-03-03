import unittest

import numpy as np

import graphrole.roles.factor as fctr

np.random.seed(0)


class TestFactor(unittest.TestCase):

    """ Unit tests for factor module """

    n_rows, n_cols = 20, 30
    X = np.random.rand(n_rows, n_cols)

    def test_get_nmf_decomposition(self):
        for n_roles in range(2, 8):
            G, F = fctr.get_nmf_decomposition(self.X, n_roles)
            expected_G_shape = (self.n_rows, n_roles)
            expected_F_shape = (n_roles, self.n_cols)
            self.assertTupleEqual(G.shape, expected_G_shape)
            self.assertTupleEqual(F.shape, expected_F_shape)
            self.assertTrue((G >= 0).all())
            self.assertTrue((F >= 0).all())

    def test_encode(self):
        for n_bins in range(1, 8):
            encoded = fctr.encode(self.X, n_bins)
            num_uniq_vals = len(np.unique(encoded))
            self.assertLessEqual(num_uniq_vals, n_bins)
