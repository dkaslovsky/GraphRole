import unittest

import numpy as np

import graphrole.roles.description_length as dl

np.random.seed(0)


class TestDescriptionLength(unittest.TestCase):

    """ Unit tests for description_length module """

    n_rows, n_cols = 20, 30
    X = np.random.rand(n_rows, n_cols)

    def test_get_encoding_cost(self):
        G = np.array([[1, 2, 3], [1, 2, 4]])  # 4 unique values
        F = np.array([[1, 2, 3], [4, 5, 5]])  # 5 unique values
        model = (G, F)
        cost = dl.get_encoding_cost(model)
        # estimated bits = np.ceil(np.log2(5)) = 3
        expected_cost = 3 * (G.size + F.size)
        self.assertEqual(cost, expected_cost)

    def test_get_error_cost(self):
        # assert positive error cost
        approx = abs(self.X - np.random.randn(*self.X.shape))
        cost = dl.get_error_cost(self.X, approx)
        self.assertGreater(cost, 0)
        # assert zero error cost for same input
        approx = self.X
        cost = dl.get_error_cost(self.X, approx)
        self.assertEqual(cost, 0)

    def test_get_description_length(self):
        n_roles = 4  # arbitrary
        model = (
            np.random.rand(self.n_rows, n_roles),
            np.random.rand(n_roles, self.n_cols)
        )
        costs = dl.get_description_length_costs(self.X, model)
        self.assertEqual(len(costs), 2)
