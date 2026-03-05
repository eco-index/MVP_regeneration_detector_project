import unittest
import numpy as np
from PIL import Image
import sys
import os

# Adjust the Python path to include the parent directory (scripts)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.inference import create_patches, stitch_masks

class TestPatchCreation(unittest.TestCase):
    def setUp(self):
        """Create a dummy PIL Image for use in tests."""
        self.image_width = 100
        self.image_height = 80
        self.image = Image.new('RGB', (self.image_width, self.image_height), color='blue')

    def test_patch_creation_no_overlap(self):
        """Test patch creation with no overlap."""
        patch_size = 50
        overlap_px = 0
        patches = create_patches(self.image, patch_size, overlap_px)

        # Expected: 100x80 image, patch_size 50.
        # Width: 100 // 50 = 2 patches
        # Height: 80 // 50 = 1 full patch, 1 smaller patch (30px high) -> 2 patches
        # Total: 2 * 2 = 4 patches
        self.assertEqual(len(patches), 4)

        # Check coordinates and dimensions of sample patches
        # Patch 0 (0,0)
        self.assertEqual(patches[0]['x_start'], 0)
        self.assertEqual(patches[0]['y_start'], 0)
        self.assertEqual(patches[0]['x_end'], 50)
        self.assertEqual(patches[0]['y_end'], 50)
        self.assertEqual(patches[0]['patch_image'].size, (50, 50))

        # Patch 1 (50,0)
        self.assertEqual(patches[1]['x_start'], 50)
        self.assertEqual(patches[1]['y_start'], 0)
        self.assertEqual(patches[1]['x_end'], 100)
        self.assertEqual(patches[1]['y_end'], 50)
        self.assertEqual(patches[1]['patch_image'].size, (50, 50))

        # Patch 2 (0,50)
        self.assertEqual(patches[2]['x_start'], 0)
        self.assertEqual(patches[2]['y_start'], 50)
        self.assertEqual(patches[2]['x_end'], 50)
        self.assertEqual(patches[2]['y_end'], 80) # y_end should be image height
        self.assertEqual(patches[2]['patch_image'].size, (50, 30))

        # Patch 3 (50,50)
        self.assertEqual(patches[3]['x_start'], 50)
        self.assertEqual(patches[3]['y_start'], 50)
        self.assertEqual(patches[3]['x_end'], 100)
        self.assertEqual(patches[3]['y_end'], 80) # y_end should be image height
        self.assertEqual(patches[3]['patch_image'].size, (50, 30))

    def test_patch_creation_with_overlap(self):
        """Test patch creation with overlap."""
        patch_size = 50
        overlap_px = 10
        stride = patch_size - overlap_px # 40

        patches = create_patches(self.image, patch_size, overlap_px)
        
        # Width: 100, Patch 50, Stride 40
        # Patches start at x=0, x=40, x=80 (this one will be 20 wide: 80 to 100)
        # num_x_patches = ceil((image_width - patch_size) / stride) + 1 if image_width > patch_size else 1
        # num_x_patches = ceil((100 - 50) / 40) + 1 = ceil(50/40)+1 = ceil(1.25)+1 = 2+1 = 3
        # Height: 80, Patch 50, Stride 40
        # Patches start at y=0, y=40 (this one will be 40 high: 40 to 80)
        # num_y_patches = ceil((image_height - patch_size) / stride) + 1 if image_height > patch_size else 1
        # num_y_patches = ceil((80 - 50) / 40) + 1 = ceil(30/40)+1 = ceil(0.75)+1 = 1+1 = 2
        # Total expected patches: 3 * 2 = 6
        self.assertEqual(len(patches), 6)

        # Coords:
        # Row 1: (0,0)-(50,50), (40,0)-(90,50), (80,0)-(100,50)
        # Row 2: (0,40)-(50,80), (40,40)-(90,80), (80,40)-(100,80)

        # Patch 0 (0,0)
        self.assertEqual(patches[0]['x_start'], 0)
        self.assertEqual(patches[0]['y_start'], 0)
        self.assertEqual(patches[0]['x_end'], 50)
        self.assertEqual(patches[0]['y_end'], 50)
        self.assertEqual(patches[0]['patch_image'].size, (50, 50))

        # Patch 1 (40,0)
        self.assertEqual(patches[1]['x_start'], 40)
        self.assertEqual(patches[1]['y_start'], 0)
        self.assertEqual(patches[1]['x_end'], 90)
        self.assertEqual(patches[1]['y_end'], 50)
        self.assertEqual(patches[1]['patch_image'].size, (50, 50))

        # Patch 2 (80,0) - edge patch
        self.assertEqual(patches[2]['x_start'], 80)
        self.assertEqual(patches[2]['y_start'], 0)
        self.assertEqual(patches[2]['x_end'], 100)
        self.assertEqual(patches[2]['y_end'], 50)
        self.assertEqual(patches[2]['patch_image'].size, (20, 50))
        
        # Patch 3 (0,40)
        self.assertEqual(patches[3]['x_start'], 0)
        self.assertEqual(patches[3]['y_start'], 40)
        self.assertEqual(patches[3]['x_end'], 50)
        self.assertEqual(patches[3]['y_end'], 80) # y_end should be image height
        self.assertEqual(patches[3]['patch_image'].size, (50, 40))


    def test_patch_edge_cases(self):
        """Test with an image smaller than the patch size."""
        small_image = Image.new('RGB', (30, 40), color='red')
        patch_size = 50
        overlap_px = 10
        patches = create_patches(small_image, patch_size, overlap_px)

        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0]['x_start'], 0)
        self.assertEqual(patches[0]['y_start'], 0)
        self.assertEqual(patches[0]['x_end'], 30)
        self.assertEqual(patches[0]['y_end'], 40)
        self.assertEqual(patches[0]['patch_image'].size, (30, 40))


class TestMaskStitching(unittest.TestCase):
    def test_stitch_no_overlap(self):
        """Test stitching with non-overlapping masks."""
        original_width, original_height = 100, 50
        
        mask1_array = np.ones((50, 50), dtype=np.uint8)
        mask2_array = np.zeros((50, 50), dtype=np.uint8)

        patches_with_masks = [
            {'mask_array': mask1_array, 'x_start': 0, 'y_start': 0, 'x_end': 50, 'y_end': 50},
            {'mask_array': mask2_array, 'x_start': 50, 'y_start': 0, 'x_end': 100, 'y_end': 50},
        ]

        full_mask = stitch_masks(patches_with_masks, original_width, original_height)

        self.assertEqual(full_mask.shape, (original_height, original_width))
        np.testing.assert_array_equal(full_mask[:, :50], mask1_array)
        np.testing.assert_array_equal(full_mask[:, 50:], mask2_array)

    def test_stitch_with_overlap_union(self):
        """Test stitching with overlapping masks, ensuring union (np.maximum)."""
        original_width, original_height = 60, 50
        patch_dim = 50 # square patches

        # Patch1: (0,0) to (50,50), Mask: all 1s
        mask1_array = np.ones((patch_dim, patch_dim), dtype=np.uint8)
        
        # Patch2: (10,0) to (60,50), Mask: all 1s in its local coords
        # This means its mask is also 50x50, but it's placed starting at x=10
        mask2_array = np.ones((patch_dim, patch_dim), dtype=np.uint8)
        
        # Modify mask1 to have some 0s in the overlap region to test np.maximum
        # Overlap region for mask1 is x from 10 to 50 (mask1 local coords)
        mask1_array[:, 10:patch_dim] = 0 # Set right part of mask1 (in overlap) to 0

        patches_with_masks = [
            {'mask_array': mask1_array, 'x_start': 0, 'y_start': 0, 'x_end': 50, 'y_end': 50},
            {'mask_array': mask2_array, 'x_start': 10, 'y_start': 0, 'x_end': 60, 'y_end': 50},
        ]
        
        full_mask = stitch_masks(patches_with_masks, original_width, original_height)
        self.assertEqual(full_mask.shape, (original_height, original_width))

        # Expected behavior:
        # Region 0-10 (x): Only Patch1 (mask1_array[:, :10]), which is all 1s.
        # Region 10-50 (x): Overlap. Patch1 has 0s here (mask1_array[:, 10:]). Patch2 has 1s (mask2_array[:, :40]).
        #                    np.maximum means this region should be 1s.
        # Region 50-60 (x): Only Patch2 (mask2_array[:, 40:]), which is all 1s.
        
        # So, the entire full_mask should be 1s.
        expected_mask = np.ones((original_height, original_width), dtype=np.uint8)
        np.testing.assert_array_equal(full_mask, expected_mask)

    def test_stitch_single_patch(self):
        """Test stitching with a single patch covering the whole image."""
        original_width, original_height = 50, 50
        mask_array = np.random.randint(0, 2, (original_height, original_width), dtype=np.uint8)
        
        patches_with_masks = [
            {'mask_array': mask_array, 'x_start': 0, 'y_start': 0, 'x_end': original_width, 'y_end': original_height},
        ]

        full_mask = stitch_masks(patches_with_masks, original_width, original_height)

        self.assertEqual(full_mask.shape, (original_height, original_width))
        np.testing.assert_array_equal(full_mask, mask_array)

if __name__ == '__main__':
    unittest.main()
