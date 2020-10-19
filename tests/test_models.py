"""Tests for statistics functions within the Model layer."""

import numpy as np
import pytest
import numpy.testing as npt
from unittest.mock import patch

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    # NB: the comment 'yapf: disable' disables automatic formatting using
    # a tool called 'yapf' which we have used when creating this project
    test_array = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([0, 0]), daily_mean(test_array))


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_array = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([3, 4]), daily_mean(test_array))


def test_daily_min():
    """Test that min function works for an array of integers."""
    from inflammation.models import daily_min

    test_array = np.array([[-4, 2, -5],
                           [0, 6, -8],
                           [1, 1, 9]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([-4,1,-8]), daily_min(test_array))


def test_daily_max():
    """Test that max function works for an array of positive integers."""
    from inflammation.models import daily_max

    test_array = np.array([[4, 2, 5],
                           [0, 6, 8],
                           [1, 1, 9]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([4,6,9]), daily_max(test_array))


# testing for errors:
def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])



@patch('inflammation.models.get_data_dir', return_value='/data_dir')
def test_load_csv(mock_get_data_dir):
    from inflammation.models import load_csv
    with patch('numpy.loadtxt') as mock_loadtxt:
        load_csv('test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[0]
        assert kwargs['fname'] == '/data_dir/test.csv'
        load_csv('/test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[1]
        assert kwargs['fname'] == '/test.csv'


# test more than one function INPUT at the same time by parameterizing the tests with multiple inputs 
# NB this is not testing more than one function (we are only testing daily_mean), this is condensing two TEST functions into one 
@pytest.mark.parametrize(
    "test, expected",
    [
     ([[0, 0], [0, 0], [0, 0]], [0, 0]),
     ([[1, 2], [3, 4], [5, 6]], [3, 4])
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(np.array(expected), daily_mean(np.array(test)))


# test multiple test cases at the same time, here for daily_max
@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[4, 2, 5], [1, 6, 2], [4, 1, 9]], [4, 6, 9]),
        ([[4, -2, 5], [1, -6, 2], [-4, -1, 9]], [4, -1, 9]),
    ])
def test_daily_max(test, expected):
    """Test max function works for zeroes, positive integers, mix of positive/negative integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(np.array(expected), daily_max(np.array(test)))


# and again for daily_min
@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[4, 2, 5], [1, 6, 2], [4, 1, 9]], [1, 1, 2]),
        ([[4, -2, 5], [1, -6, 2], [-4, -1, 9]], [-4, -6, 2]),
    ])
def test_daily_min(test, expected):
    """Test min function works for zeroes, positive integers, mix of positive/negative integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(np.array(expected), daily_min(np.array(test)))



# TODO(lesson-automatic) Implement tests for the other statistical functions
# TODO(lesson-mocking) Implement a unit test for the load_csv function
