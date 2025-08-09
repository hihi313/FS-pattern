from pathlib import Path
import struct
import tempfile
import pytest
import numpy as np

from src import util


class TestOrderPoints:
    """
    Test suite for the util.util.order_points function using shared data for valid inputs.
    """

    @pytest.fixture(scope="class", autouse=True)
    def setup_points(self, request):
        """
        Sets up shared point data as class attributes.
        This fixture runs once per class.
        """
        # A perfect rectangle, ordered correctly.
        request.cls.ordered_rect = np.array(
            [
                [0, 0],  # Top-left
                [100, 0],  # Top-right
                [100, 100],  # Bottom-right
                [0, 100],  # Bottom-left
            ],
            dtype=np.float32,
        )

        # The same rectangle, but with points shuffled.
        request.cls.shuffled_rect = np.array(
            [
                [100, 100],  # Bottom-right
                [0, 0],  # Top-left
                [0, 100],  # Bottom-left
                [100, 0],  # Top-right
            ],
            dtype=np.float32,
        )

        # A non-rectangular quadrilateral (trapezoid), ordered correctly.
        request.cls.ordered_trapezoid = np.array(
            [
                [20, 10],  # Top-left
                [90, 20],  # Top-right
                [100, 90],  # Bottom-right
                [10, 80],  # Bottom-left
            ],
            dtype=np.float64,
        )

        # The same trapezoid, but with points shuffled.
        request.cls.shuffled_trapezoid = np.array(
            [[10, 80], [100, 90], [20, 10], [90, 20]], dtype=np.float64
        )

    def test_order_shuffled_rectangle(self):
        """
        Tests if the function correctly orders a shuffled set of rectangle points
        and preserves the original shape and dtype.
        """
        actual_ordered_pts = util.order_points(self.shuffled_rect)
        np.testing.assert_array_equal(actual_ordered_pts, self.ordered_rect)
        assert actual_ordered_pts.shape == self.shuffled_rect.shape
        assert actual_ordered_pts.dtype == self.shuffled_rect.dtype

    def test_already_ordered_input(self):
        """
        Tests if the function returns the same order and shape for already ordered points.
        """
        actual_ordered_pts = util.order_points(self.ordered_rect)
        np.testing.assert_array_equal(actual_ordered_pts, self.ordered_rect)
        assert actual_ordered_pts.shape == self.ordered_rect.shape

    def test_order_shuffled_trapezoid(self):
        """
        Tests if the function correctly orders a shuffled set of trapezoid points
        and preserves the original shape and dtype.
        """
        actual_ordered_pts = util.order_points(self.shuffled_trapezoid)
        np.testing.assert_array_equal(actual_ordered_pts, self.ordered_trapezoid)
        assert actual_ordered_pts.shape == self.shuffled_trapezoid.shape
        assert actual_ordered_pts.dtype == self.shuffled_trapezoid.dtype

    def test_input_as_list(self):
        """
        Tests if the function handles input given as a Python list.
        The function should cast it to a float64 numpy array of shape (4, 2).
        """
        pts_list = self.shuffled_rect.tolist()
        actual_ordered_pts = util.order_points(pts_list)

        # The expected output shape for a list input is (4, 2).
        expected_shape = (4, 2)

        np.testing.assert_array_equal(actual_ordered_pts, self.ordered_rect)
        assert actual_ordered_pts.shape == expected_shape
        assert actual_ordered_pts.dtype == np.float64

    def test_input_with_squeezable_dimension(self):
        """
        Tests an input with an extra dimension that can be removed by squeeze().
        The output shape must match the original input shape (e.g., (1, 4, 2)).
        """
        # Create an input with a squeezable dimension, shape (1, 4, 2).
        pts_3d = self.shuffled_rect[np.newaxis, ...]
        assert pts_3d.shape == (1, 4, 2)

        # The expected output should have the same shape as the input.
        expected_ordered_pts_3d = self.ordered_rect[np.newaxis, ...]
        assert expected_ordered_pts_3d.shape == (1, 4, 2)

        # Call the function and verify the output.
        actual_ordered_pts = util.order_points(pts_3d)

        # Check that the shape is preserved.
        assert actual_ordered_pts.shape == pts_3d.shape

        # Check that the points are correctly ordered within that shape.
        np.testing.assert_array_equal(actual_ordered_pts, expected_ordered_pts_3d)

    # --- Tests for invalid inputs that should raise errors ---

    @pytest.mark.parametrize(
        "invalid_pts",
        [
            np.random.rand(3, 2),  # Not enough points
            np.random.rand(5, 2),  # Too many points
            np.random.rand(4, 1),  # Incorrect coordinate dimension
            np.random.rand(4, 3),  # Incorrect coordinate dimension
            np.random.rand(4),  # Not a 2D array
            np.random.rand(2, 2, 2),  # Not squeezable to (4, 2)
            np.random.rand(1, 5, 2),  # Squeezes to (5, 2)
        ],
    )
    def test_invalid_shape_raises_value_error(self, invalid_pts):
        """
        Tests that inputs with various incorrect shapes raise a ValueError.
        """
        with pytest.raises(ValueError, match="shape \\(4, 2\\) after squeeze"):
            util.order_points(invalid_pts)

    def test_non_array_like_input_raises_error(self):
        """
        Tests that non-array-like input raises an error during np.array conversion.
        """
        # np.array will raise a ValueError for a string that cannot be parsed.
        with pytest.raises(ValueError):
            util.order_points("this is not a valid input")


class TestWritePNGWithDPI:
    """
    Test suite for the util.imwrite_png function.
    """

    @pytest.fixture(scope="class")
    def temp_dir(self):
        """
        Pytest fixture to create a temporary directory for test artifacts.
        This ensures tests do not leave files on the system.
        It runs once per test class.
        """
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture(scope="class")
    def dummy_image(self):
        """
        Pytest fixture to create a simple, shared dummy image for tests.
        A 10x10 black image.
        """
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def test_file_creation_and_dpi(self, temp_dir, dummy_image):
        """
        Tests if the file is created and contains the correct DPI information
        in a pHYs chunk.
        """
        # --- Arrange ---
        output_filename = temp_dir / "test_output.png"
        target_dpi = (300, 300)
        METERS_PER_INCH = 0.0254

        # --- Act ---
        util.imwrite_png(dummy_image, str(output_filename), dpi=target_dpi)

        # --- Assert ---
        # 1. Check if the file was actually created
        assert output_filename.exists(), "Output PNG file was not created."

        # 2. Read the file and verify the pHYs chunk and its content
        with open(output_filename, "rb") as f:
            png_data = f.read()

        # Check that the pHYs chunk exists
        assert b'pHYs' in png_data, "pHYs chunk is missing from the output file."

        # Find the start of the pHYs data (after length and type bytes)
        phys_chunk_start = png_data.find(b'pHYs') + 4
        phys_chunk_data = png_data[phys_chunk_start : phys_chunk_start + 9]

        # Unpack the data from the chunk
        ppm_x, ppm_y, unit = struct.unpack("!IIc", phys_chunk_data)
        
        # Convert pixels per meter back to DPI to verify
        read_dpi_x = int(round(ppm_x * METERS_PER_INCH))
        read_dpi_y = int(round(ppm_y * METERS_PER_INCH))

        assert read_dpi_x == target_dpi[0], f"X DPI mismatch: expected {target_dpi[0]}, got {read_dpi_x}"
        assert read_dpi_y == target_dpi[1], f"Y DPI mismatch: expected {target_dpi[1]}, got {read_dpi_y}"
        assert unit == b'\x01', "Unit specifier is not set to 'meters'."


class TestPadToSize:
    """
    Test suite for the pad_to_size function.
    Uses a class structure to share small, pre-defined test images.
    """

    @classmethod
    def setup_class(cls):
        """
        Set up small test images that are shared across all tests in this class.
        This method is run once before any tests in the class execute.
        """
        # A small 2x3 grayscale image for testing
        cls.gray_img = np.array([
            [10, 20, 30],
            [40, 50, 60]
        ], dtype=np.uint8)

        # A small 2x2 color image for testing
        cls.color_img = np.array([
            [[255, 0, 0], [0, 255, 0]],  # Red, Green
            [[0, 0, 255], [50, 50, 50]]   # Blue, Gray
        ], dtype=np.uint8)

    def test_pad_grayscale_default_bg(self):
        """
        Tests padding a grayscale image using the default background color,
        which should be the color of the top-left pixel (10).
        """
        padded_img = util.pad_to_size(self.gray_img, 4, 5)
        
        # The background color is inferred from the top-left pixel of the input image
        expected_bg = 10
        
        expected_img = np.full((4, 5), expected_bg, dtype=np.uint8)
        expected_img[1:3, 1:4] = self.gray_img # Centered placement

        assert padded_img.shape == (4, 5)
        assert np.array_equal(padded_img, expected_img)

    def test_pad_grayscale_custom_bg(self):
        """
        Tests padding a grayscale image with a specified background color (128).
        """
        custom_bg = 128
        padded_img = util.pad_to_size(self.gray_img, 4, 5, bg_color=custom_bg)

        expected_img = np.full((4, 5), custom_bg, dtype=np.uint8)
        expected_img[1:3, 1:4] = self.gray_img # Centered placement

        assert padded_img.shape == (4, 5)
        assert np.array_equal(padded_img, expected_img)

    def test_pad_color_default_bg(self):
        """
        Tests padding a color image using the default background color,
        which should be the color of the top-left pixel ([255, 0, 0]).
        """
        padded_img = util.pad_to_size(self.color_img, 4, 4)

        # The background color is inferred from the top-left pixel
        expected_bg = self.color_img[0,0]

        expected_img = np.full((4, 4, 3), expected_bg, dtype=np.uint8)
        expected_img[1:3, 1:3] = self.color_img # Centered placement

        assert padded_img.shape == (4, 4, 3)
        assert np.array_equal(padded_img, expected_img)

    def test_pad_color_custom_bg(self):
        """
        Tests padding a color image with a specified background color ([0, 0, 0]).
        """
        custom_bg = (0, 0, 0) # Black
        padded_img = util.pad_to_size(self.color_img, 4, 4, bg_color=custom_bg)

        expected_img = np.full((4, 4, 3), custom_bg, dtype=np.uint8)
        expected_img[1:3, 1:3] = self.color_img # Centered placement

        assert padded_img.shape == (4, 4, 3)
        assert np.array_equal(padded_img, expected_img)

    def test_no_padding_needed(self):
        """
        Tests the case where the target dimensions are the same as the image.
        The function should return an identical image.
        """
        H, W, _ = self.color_img.shape
        padded_img = util.pad_to_size(self.color_img, H, W)

        assert padded_img.shape == self.color_img.shape
        assert np.array_equal(padded_img, self.color_img)

    def test_raises_value_error_on_invalid_size(self):
        """
        Tests that a ValueError is raised if the target dimensions are smaller
        than the input image dimensions.
        """
        with pytest.raises(ValueError) as excinfo:
            # Target size (1x1) is smaller than image (2x3)
            util.pad_to_size(self.gray_img, 1, 1)
        
        # Optionally, check the exception message for clarity
        assert ">= the smaller image" in str(excinfo.value)
