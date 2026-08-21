from __future__ import annotations

import unittest

import torch

from position_getter import OnnxPositionGetter


class PositionGetterTest(unittest.TestCase):
    def test_matches_cartesian_product_exactly(self):
        getter = OnnxPositionGetter()
        for height, width in ((1, 1), (2, 3), (37, 37)):
            expected = torch.cartesian_prod(
                torch.arange(height), torch.arange(width)
            ).view(1, height * width, 2).expand(2, -1, -1).clone()
            actual = getter(2, height, width, torch.device("cpu"))
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
