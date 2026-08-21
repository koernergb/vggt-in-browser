from __future__ import annotations

import unittest

import torch
from depth_position_embed import position_grid_to_embed
from vggt.heads.utils import position_grid_to_embed as upstream_position_grid_to_embed


class DepthPositionEmbedTest(unittest.TestCase):
    def test_matches_upstream_with_float32_tolerance(self):
        grid = torch.linspace(-1, 1, 30, dtype=torch.float32).view(3, 5, 2)
        expected = upstream_position_grid_to_embed(grid, 64)
        actual = position_grid_to_embed(grid, 64)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
