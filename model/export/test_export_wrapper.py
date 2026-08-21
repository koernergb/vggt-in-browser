from __future__ import annotations

import unittest

import torch

from export_camera_depth import CameraDepthExportWrapper


class FakeAggregator(torch.nn.Module):
    def forward(self, images: torch.Tensor):
        return [images + 1], 5


class FakeCameraHead(torch.nn.Module):
    def forward(self, aggregated):
        return [aggregated[0].mean(dim=(-1, -2, -3))]


class FakeDepthHead(torch.nn.Module):
    def forward(self, aggregated, images, patch_start_idx):
        self.last_patch_start_idx = patch_start_idx
        return images.mean(dim=2, keepdim=False).unsqueeze(-1), images.mean(dim=2)


class FakeVGGT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = FakeAggregator()
        self.camera_head = FakeCameraHead()
        self.depth_head = FakeDepthHead()


class ExportWrapperTest(unittest.TestCase):
    def test_stable_flat_outputs(self):
        wrapper = CameraDepthExportWrapper(FakeVGGT())
        images = torch.ones((1, 2, 3, 4, 4))
        pose, depth, confidence = wrapper(images)
        self.assertEqual(tuple(pose.shape), (1, 2))
        self.assertEqual(tuple(depth.shape), (1, 2, 4, 4, 1))
        self.assertEqual(tuple(confidence.shape), (1, 2, 4, 4))
        self.assertEqual(wrapper.depth_head.last_patch_start_idx, 5)


if __name__ == "__main__":
    unittest.main()
