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
