import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import math
import argparse

class InfeasibleTransformError(Exception):
    """Custom exception for when the transform cannot find valid parameters in a single attempt,
    or if initial parameters suggest infeasibility for a given minimum image dimension."""
    pass

class RandomRotatedZoomedCropFailFast(DualTransform):
    """
    Takes a randomly positioned, randomly rotated, randomly zoomed square crop
    from the input image and resizes it to target dimensions.
    Ensures all 4 corners of the rotated square are within the image bounds.

    This version attempts to find valid parameters ONCE. If it cannot,
    it raises an InfeasibleTransformError. No retries, no fallbacks.

    Args:
        height (int): Height of the output crop.
        width (int): Width of the output crop.
        scale_limit (tuple[float, float]): Range from which to sample a random scale factor
            for the initial square's side length. The side length will be
            max(height, width) * scale.
        rotate_limit (tuple[float, float]): Range in degrees from which to sample a random
            rotation angle.
        interpolation (int): OpenCV interpolation method for image.
        mask_interpolation (int): OpenCV interpolation method for mask.
        border_mode (int): OpenCV border mode.
        value (int | float | list[int] | list[float] | None): Fill value for image if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int | float | list[int] | list[float] | None): Fill value for mask if border_mode is cv2.BORDER_CONSTANT.
        p (float): Probability of applying the transform.
        check_min_img_dim (int | None): If provided, checks at initialization time if the
            transform parameters (max_scale, output crop dimensions) are potentially
            infeasible for an image with this minimum dimension (height or width).
            The check uses the formula:
            `sqrt(output_height^2 + output_width^2) * max_scale > check_min_img_dim`.
            If true, raises InfeasibleTransformError.
    """
    def __init__(
        self,
        height: int = 1024,
        width: int = 1024,
        scale_limit: tuple[float, float] = (0.5, 2.0),
        rotate_limit: tuple[float, float] = (-45, 45),
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        border_mode: int = cv2.BORDER_CONSTANT,
        value: int | float | list[int] | list[float] | None = 0,
        mask_value: int | float | list[int] | list[float] | None = 0,
        p: float = 1.0,
        check_min_img_dim: int | None = None, # New parameter
    ):
        super().__init__(p=p)
        if not (0 < height and 0 < width):
            raise ValueError("Height and width must be positive.")
        if not (0 < scale_limit[0] <= scale_limit[1]):
            raise ValueError("Scale limit must be positive and min <= max.")
        if check_min_img_dim is not None and check_min_img_dim <= 0:
            raise ValueError("check_min_img_dim must be positive if provided.")


        self.height = height
        self.width = width
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.check_min_img_dim = check_min_img_dim

        if self.check_min_img_dim is not None:
            # Perform the early check as requested by the user.
            # Formula: ((crop_dim_a ** 2) + (crop_dim_b ** 2)) ** 0.5 * max_scale
            # where crop_dim_a is self.height, crop_dim_b is self.width.
            # This value is compared against self.check_min_img_dim.
            
            output_height = float(self.height)
            output_width = float(self.width)
            max_scale = self.scale_limit[1]

            # This is the LHS of the inequality provided in the prompt.
            # It represents a calculated "required dimension" based on output crop diagonal and max_scale.
            calculated_requirement = (
                math.sqrt(output_height**2 + output_width**2) * max_scale
            )

            if calculated_requirement > self.check_min_img_dim:
                raise InfeasibleTransformError(
                    f"Initial parameters may lead to an infeasible transform for an image "
                    f"with minimum dimension {self.check_min_img_dim}. "
                    f"The check 'sqrt(output_height^2 + output_width^2) * max_scale <= min_image_dim' failed. "
                    f"Calculated required dimension (sqrt({output_height}^2 + {output_width}^2) * {max_scale}) "
                    f"is {calculated_requirement:.2f}, which is greater than "
                    f"the provided check_min_img_dim ({self.check_min_img_dim})."
                )

    def get_params_dependent_on_data(self, params: dict, data: dict) -> dict:
        img_h, img_w = data["image"].shape[:2]

        # 1. Sample random scale for the source square
        scale = self.py_random.uniform(self.scale_limit[0], self.scale_limit[1])
        src_square_side = max(self.height, self.width) * scale

        # 2. Sample random rotation
        angle_deg = self.py_random.uniform(self.rotate_limit[0], self.rotate_limit[1])
        angle_rad = math.radians(angle_deg)

        # 3. Define corners of the unrotated source square, centered at origin
        half_side = src_square_side / 2.0
        src_pts_unrotated = np.array([
            [-half_side, -half_side], [half_side, -half_side],
            [half_side, half_side], [-half_side, half_side],
        ], dtype=np.float32)

        # 4. Rotate these points
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotation_matrix_2d = np.array([
            [cos_a, -sin_a], [sin_a, cos_a]
        ], dtype=np.float32)
        rotated_src_pts_centered = src_pts_unrotated @ rotation_matrix_2d.T

        # 5. Find bounding box of the rotated square
        min_x_rot, min_y_rot = np.min(rotated_src_pts_centered, axis=0)
        max_x_rot, max_y_rot = np.max(rotated_src_pts_centered, axis=0)

        bb_w = max_x_rot - min_x_rot
        bb_h = max_y_rot - min_y_rot

        # Check if the bounding box of the rotated square is too large for the image
        if bb_w >= img_w or bb_h >= img_h:
            raise InfeasibleTransformError(
                f"Rotated bounding box ({bb_w:.2f}x{bb_h:.2f}) with scale {scale:.2f} and angle {angle_deg:.2f} "
                f"is too large for image ({img_w}x{img_h}). Cannot place crop."
            )

        # 6. Determine valid range for the center of this rotated square
        # so that all its corners are within the image
        min_cx_sq = -min_x_rot
        max_cx_sq = img_w - max_x_rot - 1.0 
        min_cy_sq = -min_y_rot
        max_cy_sq = img_h - max_y_rot - 1.0

        if min_cx_sq > max_cx_sq + 1e-6 or min_cy_sq > max_cy_sq + 1e-6:
            raise InfeasibleTransformError(
                f"No valid center position found for rotated square (scale {scale:.2f}, angle {angle_deg:.2f}) "
                f"in image ({img_w}x{img_h}). BBox ({bb_w:.2f}x{bb_h:.2f}). "
                f"Required Center X range: [{min_cx_sq:.2f}, {max_cx_sq:.2f}], "
                f"Required Center Y range: [{min_cy_sq:.2f}, {max_cy_sq:.2f}]."
            )

        cx_sq = self.py_random.uniform(min(min_cx_sq,max_cx_sq), max(min_cx_sq,max_cx_sq))
        cy_sq = self.py_random.uniform(min(min_cy_sq,max_cy_sq), max(min_cy_sq,max_cy_sq))

        final_src_pts = rotated_src_pts_centered + np.array([cx_sq, cy_sq], dtype=np.float32)

        if not (np.all(final_src_pts[:, 0] >= -1e-3) and np.all(final_src_pts[:, 0] < img_w + 1e-3) and
                np.all(final_src_pts[:, 1] >= -1e-3) and np.all(final_src_pts[:, 1] < img_h + 1e-3)):
             raise InfeasibleTransformError(
                f"Calculated final_src_pts are outside image bounds. This indicates an internal logic error. "
                f"Pts:\n{final_src_pts}\nImage: {img_w}x{img_h}, Scale: {scale}, Angle: {angle_deg}, Center: ({cx_sq}, {cy_sq})"
            )
        if not np.all(np.isfinite(final_src_pts)):
             raise InfeasibleTransformError(
                f"final_src_pts CONTAINS NON-FINITE VALUES! "
                f"Scale: {scale}, Angle: {angle_deg}, Center: ({cx_sq}, {cy_sq})"
            )

        dst_pts = np.array([
            [0.0, 0.0], [float(self.width - 1), 0.0],
            [float(self.width - 1), float(self.height - 1)], [0.0, float(self.height - 1)],
        ], dtype=np.float32)

        try:
            perspective_matrix = cv2.getPerspectiveTransform(final_src_pts, dst_pts)
        except cv2.error as e:
            raise InfeasibleTransformError(
                f"cv2.getPerspectiveTransform failed: {e}. "
                f"This often happens with collinear/degenerate source points. "
                f"Scale: {scale}, Angle: {angle_deg}, Center: ({cx_sq}, {cy_sq}).\n"
                f"Source points:\n{final_src_pts}"
            ) from e

        return {"matrix": perspective_matrix, "dsize": (self.width, self.height)}

    def apply(self, img: np.ndarray, matrix: np.ndarray, dsize: tuple[int, int], **params) -> np.ndarray:
        return cv2.warpPerspective(
            img, matrix, dsize=dsize, flags=self.interpolation,
            borderMode=self.border_mode, borderValue=self.value,
        )

    def apply_to_mask(self, mask: np.ndarray, matrix: np.ndarray, dsize: tuple[int, int], **params) -> np.ndarray:
        return cv2.warpPerspective(
            mask, matrix, dsize=dsize, flags=self.mask_interpolation,
            borderMode=self.border_mode, borderValue=self.mask_value,
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "height", "width", "scale_limit", "rotate_limit",
            "interpolation", "mask_interpolation", "border_mode",
            "value", "mask_value", "check_min_img_dim" # Added new parameter
        )

def main():
    parser = argparse.ArgumentParser(description="Apply RandomRotatedZoomedCropFailFast augmentation.")
    parser.add_argument("--input-image", required=True, help="Path to the input image.")
    parser.add_argument("--output-image", required=True, help="Path to save the augmented image.")
    parser.add_argument("--input-mask", help="Path to the input segmentation mask (optional).")
    parser.add_argument("--output-mask", help="Path to save the augmented mask (optional).")
    parser.add_argument("--crop-size", type=int, nargs=2, default=[1024, 1024],
                        help="Output crop size (height width), default: 1024 1024.")
    parser.add_argument("--scale-limit", type=float, nargs=2, default=[0.5, 1.5],
                        help="Scale limit for the source square (min max), default: 0.5 1.5.")
    parser.add_argument("--rotate-limit", type=float, nargs=2, default=[-45, 45],
                        help="Rotation limit in degrees (min max), default: -45 45.")
    parser.add_argument("--check-min-img-dim", type=int, default=None,
                        help="Optional: Minimum dimension (height or width) of input images to check against. "
                             "If parameters are likely infeasible for this dimension, an error is raised at init.")


    args = parser.parse_args()

    image = cv2.imread(args.input_image)
    if image is None:
        print(f"Error: Could not load image from {args.input_image}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = None
    if args.input_mask:
        mask = cv2.imread(args.input_mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load mask from {args.input_mask}. Proceeding without mask.")

    try:
        transform = RandomRotatedZoomedCropFailFast(
            height=args.crop_size[0],
            width=args.crop_size[1],
            scale_limit=tuple(args.scale_limit),
            rotate_limit=tuple(args.rotate_limit),
            p=1.0, 
            check_min_img_dim=args.check_min_img_dim # Pass the new argument
        )

        pipeline = A.Compose([transform])

        print(f"Processing image: {args.input_image} (Shape: {image.shape})")
        if mask is not None:
            print(f"Processing mask: {args.input_mask} (Shape: {mask.shape})")

        augmented_data = {}
        if mask is not None:
            augmented_data = pipeline(image=image_rgb, mask=mask)
        else:
            augmented_data = pipeline(image=image_rgb)

        augmented_image = augmented_data.get('image')
        augmented_mask = augmented_data.get('mask')

        if augmented_image is None: # Should be caught by InfeasibleTransformError already
            print("ERROR: Augmented image is None, but no InfeasibleTransformError was caught.")
            return

        cv2.imwrite(args.output_image, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
        print(f"Saved augmented image to {args.output_image}")

        if augmented_mask is not None and args.output_mask:
            cv2.imwrite(args.output_mask, augmented_mask)
            print(f"Saved augmented mask to {args.output_mask}")
        elif augmented_mask is None and args.input_mask and args.output_mask:
             print(f"Warning: Mask was input but not augmented, cannot save to {args.output_mask}")

    except InfeasibleTransformError as e:
        # This can now be raised from __init__ or from get_params_dependent_on_data
        print(f"AUGMENTATION FAILED/PREFLIGHT CHECK FAILED for {args.input_image}: {e}")
        print("No output image will be saved for this input.")
    except Exception as e: 
        print(f"UNEXPECTED ERROR during augmentation pipeline for {args.input_image}: {e}")
        import traceback
        traceback.print_exc()
        print("No output image will be saved for this input.")


if __name__ == "__main__":
    main()