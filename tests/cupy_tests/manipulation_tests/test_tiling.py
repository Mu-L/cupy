from __future__ import annotations

import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(
    {'repeats': 0, 'axis': None},
    {'repeats': 2, 'axis': None},
    {'repeats': 2, 'axis': 1},
    {'repeats': 2, 'axis': -1},
    {'repeats': [0, 0, 0], 'axis': 1},
    {'repeats': [1, 2, 3], 'axis': 1},
    {'repeats': [1, 2, 3], 'axis': -2},
)
class TestRepeat(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.repeat(x, self.repeats, self.axis)


# ---------------------------------------------------------------------------
# Helpers shared by ndarray-repeats test classes
# ---------------------------------------------------------------------------

def _reps(xp, vals):
    """Make a reps array on the right platform."""
    if xp is cupy:
        return cupy.array(vals)
    return numpy.array(vals)


# ---------------------------------------------------------------------------
# Comprehensive numpy_cupy_array_equal comparisons
# ---------------------------------------------------------------------------

@testing.parameterize(
    # --- basic 1-D cases (mirror numpy TestRepeat) ---
    {'shape': (6,), 'reps': [1, 3, 2, 1, 1, 2], 'axis': None},
    {'shape': (6,), 'reps': [2],                 'axis': None},  # broadcast
    # --- 2-D axis=0 ---
    {'shape': (2, 3), 'reps': [2, 1],     'axis': 0},
    {'shape': (2, 3), 'reps': [1, 3, 2],  'axis': 1},
    {'shape': (2, 3), 'reps': [2],        'axis': 0},  # broadcast
    {'shape': (2, 3), 'reps': [2],        'axis': 1},  # broadcast
    # --- 3-D various axes ---
    {'shape': (2, 3, 4), 'reps': [1, 2, 3, 4], 'axis': 2},
    {'shape': (2, 3, 4), 'reps': [0, 3],        'axis': 0},
    {'shape': (2, 3, 4), 'reps': [1, 2, 3],     'axis': 1},
    {'shape': (2, 3, 4), 'reps': [4],            'axis': 2},  # broadcast
    # --- negative axis ---
    {'shape': (2, 3, 4), 'reps': [1, 2, 3, 4], 'axis': -1},
    {'shape': (2, 3, 4), 'reps': [1, 2, 3],    'axis': -2},
    {'shape': (2, 3, 4), 'reps': [1, 2],        'axis': -3},
    # --- axis=None (ravels) ---
    {'shape': (2, 3), 'reps': [1, 2, 3, 4, 5, 0], 'axis': None},
    {'shape': (3, 4), 'reps': [2],                 'axis': None},  # broadcast
    # --- zeros in reps ---
    {'shape': (4,),   'reps': [0, 0, 0, 0],   'axis': 0},
    {'shape': (4,),   'reps': [0, 2, 0, 1],   'axis': 0},
    {'shape': (4,),   'reps': [3, 0, 0, 2],   'axis': 0},
    {'shape': (2, 3), 'reps': [0, 3, 0],       'axis': 1},
    # --- all ones (identity) ---
    {'shape': (5,),   'reps': [1, 1, 1, 1, 1], 'axis': 0},
    # --- broadcast scalar equivalence ---
    {'shape': (2, 3), 'reps': [0], 'axis': 0},
    {'shape': (2, 3), 'reps': [1], 'axis': 1},
    # --- 4-D ---
    {'shape': (2, 3, 4, 5), 'reps': [2, 1, 3], 'axis': 1},
    {'shape': (2, 3, 4, 5), 'reps': [3],        'axis': 3},
)
class TestRepeatNdarrayRepeats(unittest.TestCase):
    """CuPy ndarray repeats matches numpy for diverse cases."""

    @testing.numpy_cupy_array_equal()
    def test_func(self, xp):
        x = testing.shaped_arange(self.shape, xp)
        return xp.repeat(x, _reps(xp, self.reps), self.axis)

    @testing.numpy_cupy_array_equal()
    def test_method(self, xp):
        x = testing.shaped_arange(self.shape, xp)
        return x.repeat(_reps(xp, self.reps), self.axis)


@testing.parameterize(
    {'reps': [1, 2, 3, 4]},
    {'reps': [0, 0, 0, 0]},
    {'reps': [5, 0, 3, 1]},
    {'reps': [0, 1, 0, 2]},  # zeros at boundaries
    {'reps': [1, 1, 1, 1]},
    {'reps': [0]},            # broadcast via size-1: total=0
    {'reps': [4]},            # broadcast via size-1
)
class TestRepeatNdarrayRepeatsAxisNone(unittest.TestCase):
    """axis=None ravels ``a`` to 1-D before applying per-element repeats."""

    @testing.numpy_cupy_array_equal()
    def test_func(self, xp):
        x = testing.shaped_arange((4,), xp)
        return xp.repeat(x, _reps(xp, self.reps), None)


@testing.parameterize(
    # --- reps integer dtypes accepted by both numpy and CuPy ---
    {'rep_dtype': numpy.int8},
    {'rep_dtype': numpy.int16},
    {'rep_dtype': numpy.int32},
    {'rep_dtype': numpy.int64},
    {'rep_dtype': numpy.uint8},
    {'rep_dtype': numpy.uint16},
    {'rep_dtype': numpy.uint32},
    # uint64 excluded: numpy rejects (unsafe cast); tested below
)
class TestRepeatNdarrayRepeatsDtype(unittest.TestCase):
    """ndarray repeats works for signed and unsigned int dtypes."""

    @testing.numpy_cupy_array_equal()
    def test_repeat(self, xp):
        x = testing.shaped_arange((4,), xp)
        if xp is cupy:
            reps = cupy.array([1, 2, 3, 4], dtype=self.rep_dtype)
        else:
            reps = numpy.array([1, 2, 3, 4], dtype=self.rep_dtype)
        return xp.repeat(x, reps, axis=0)

    @testing.numpy_cupy_array_equal()
    def test_broadcast(self, xp):
        x = testing.shaped_arange((5,), xp)
        if xp is cupy:
            reps = cupy.array([3], dtype=self.rep_dtype)
        else:
            reps = numpy.array([3], dtype=self.rep_dtype)
        return xp.repeat(x, reps)


class TestRepeatNdarrayRepeatsUint64(unittest.TestCase):
    """uint64 reps: CuPy accepts; numpy rejects."""

    def test_uint64_accepted(self):
        x = testing.shaped_arange((4,), cupy)
        reps = cupy.array([1, 2, 3, 4], dtype=numpy.uint64)
        result = cupy.repeat(x, reps, axis=0)
        expected = cupy.repeat(
            x, cupy.array([1, 2, 3, 4], dtype=numpy.int64),
            axis=0)
        cupy.testing.assert_array_equal(result, expected)

    def test_uint64_broadcast_accepted(self):
        x = testing.shaped_arange((5,), cupy)
        reps = cupy.array([3], dtype=numpy.uint64)
        result = cupy.repeat(x, reps)
        expected = cupy.repeat(x, cupy.array([3], dtype=numpy.int64))
        cupy.testing.assert_array_equal(result, expected)


@testing.parameterize(
    # --- array ``a`` dtypes ---
    {'a_dtype': numpy.bool_},
    {'a_dtype': numpy.int8},
    {'a_dtype': numpy.int32},
    {'a_dtype': numpy.int64},
    {'a_dtype': numpy.float16},
    {'a_dtype': numpy.float32},
    {'a_dtype': numpy.float64},
    {'a_dtype': numpy.complex64},
    {'a_dtype': numpy.complex128},
)
class TestRepeatNdarrayRepeatsArrayDtype(unittest.TestCase):
    """Output dtype matches input ``a`` dtype for all numeric types."""

    @testing.numpy_cupy_array_equal()
    def test_dtype_preserved(self, xp):
        x = testing.shaped_arange((3, 4), xp, dtype=self.a_dtype)
        reps = _reps(xp, [1, 2, 3, 4])
        result = xp.repeat(x, reps, axis=1)
        # Verify dtype is preserved
        assert result.dtype == x.dtype
        return result


def _int_reps(xp, vals):
    """Make an integer reps array (int32 dtype), even for empty lists."""
    if xp is cupy:
        return cupy.array(vals, dtype=numpy.int32)
    return numpy.array(vals, dtype=numpy.int32)


@testing.parameterize(
    # --- empty array shapes ---
    {'shape': (0,),    'reps': [],     'axis': 0},
    {'shape': (0, 3),  'reps': [],     'axis': 0},
    {'shape': (2, 0),  'reps': [],     'axis': 1},
    {'shape': (0, 3),  'reps': [2],    'axis': 0},  # broadcast empty
    {'shape': (2, 0),  'reps': [3],    'axis': 1},  # broadcast empty
    {'shape': (2, 3),  'reps': [0, 0, 0], 'axis': 1},  # all-zero -> empty
)
class TestRepeatNdarrayRepeatsEmptyArrays(unittest.TestCase):
    """Empty input or all-zero repeats produce correct empty output shape."""

    @testing.numpy_cupy_array_equal()
    def test_empty(self, xp):
        x = testing.shaped_arange(self.shape, xp)
        return xp.repeat(x, _int_reps(xp, self.reps), self.axis)


class TestRepeatNdarrayRepeatsNonContiguous(unittest.TestCase):
    """Repeat works correctly on non-contiguous (strided) input arrays."""

    @testing.numpy_cupy_array_equal()
    def test_transposed(self, xp):
        # shape (3,4), not C-contiguous
        x = testing.shaped_arange((4, 3), xp).T
        return xp.repeat(x, _reps(xp, [2, 1, 3, 0]), axis=1)

    @testing.numpy_cupy_array_equal()
    def test_column_slice(self, xp):
        # Ticket #352 equivalent: non-contiguous via column slice
        x = testing.shaped_arange((4, 3), xp)[:, 2]  # shape (4,), stride 3
        return xp.repeat(x, _reps(xp, [3, 0, 1, 2]))

    @testing.numpy_cupy_array_equal()
    def test_strided_axis1(self, xp):
        x = testing.shaped_arange((3, 8), xp)[:, ::2]  # shape (3,4), stride 2
        return xp.repeat(x, _reps(xp, [1, 2, 3, 0]), axis=1)

    @testing.numpy_cupy_array_equal()
    def test_reversed(self, xp):
        x = testing.shaped_arange((5,), xp)[::-1]  # reversed
        return xp.repeat(x, _reps(xp, [0, 1, 2, 1, 0]))


class TestRepeatNdarrayRepeatsBroadcast(unittest.TestCase):
    """Size-1 ndarray repeats broadcast to any axis dimension."""

    @testing.numpy_cupy_array_equal()
    def test_broadcast_axis0(self, xp):
        x = testing.shaped_arange((4, 5), xp)
        return xp.repeat(x, _reps(xp, [3]), axis=0)

    @testing.numpy_cupy_array_equal()
    def test_broadcast_axis1(self, xp):
        x = testing.shaped_arange((4, 5), xp)
        return xp.repeat(x, _reps(xp, [2]), axis=1)

    @testing.numpy_cupy_array_equal()
    def test_broadcast_axis_none(self, xp):
        x = testing.shaped_arange((3, 4), xp)
        return xp.repeat(x, _reps(xp, [2]), None)

    @testing.numpy_cupy_array_equal()
    def test_broadcast_zero(self, xp):
        # Repeat 0 times via broadcast: empty output
        x = testing.shaped_arange((3, 4), xp)
        return xp.repeat(x, _reps(xp, [0]), axis=0)

    @testing.numpy_cupy_array_equal()
    def test_broadcast_one(self, xp):
        # Repeat 1 time via broadcast: copy
        x = testing.shaped_arange((3, 4), xp)
        return xp.repeat(x, _reps(xp, [1]), axis=0)

    @testing.numpy_cupy_array_equal()
    def test_scalar_equiv(self, xp):
        """size-1 ndarray matches scalar int repeats."""
        x = testing.shaped_arange((2, 3, 4), xp)
        r_scalar = xp.repeat(x, 3, axis=1)
        r_ndarray = xp.repeat(x, _reps(xp, [3]), axis=1)
        assert r_scalar.shape == r_ndarray.shape
        # Return one to compare with numpy
        return r_ndarray


class TestRepeatNdarrayRepeats0D(unittest.TestCase):
    """0-D reps array broadcasts like a scalar (axis=None only)."""

    @testing.numpy_cupy_array_equal()
    def test_0d_reps_broadcasts(self, xp):
        x = testing.shaped_arange((3,), xp)
        if xp is cupy:
            reps = cupy.array(2)   # 0-D CuPy array
        else:
            reps = numpy.array(2)  # 0-D numpy array
        return xp.repeat(x, reps)


class TestRepeatScalarEquivalence(unittest.TestCase):
    """All scalar-like ``repeats`` inputs produce identical results.

    Five inputs that all mean "repeat each element N times":
      1. Python int
      2. numpy integer scalar  (numpy.int32(N))
      3. size-1 list           ([N])
      4. 1-D CuPy size-1 array (cupy.array([N]))
      5. 0-D CuPy array        (cupy.array(N))

    Cases 1-3 take the no-sync broadcast path in ``_repeat``.
    Cases 4-5 perform one D2H sync to extract N, then the same broadcast.
    All five must produce byte-identical output.
    """

    def _check_all_equal(self, a, rep_n, axis):
        np_a = cupy.asnumpy(a)
        expected = cupy.array(numpy.repeat(np_a, rep_n, axis))
        forms = [
            rep_n,
            numpy.int32(rep_n),
            [rep_n],
            cupy.array([rep_n], dtype=numpy.int32),
            cupy.array(rep_n),
        ]
        for form in forms:
            result = cupy.repeat(a, form, axis)
            cupy.testing.assert_array_equal(
                result, expected, err_msg='form={!r}'.format(form))

    def test_axis_none(self):
        a = cupy.arange(6).reshape(2, 3)
        self._check_all_equal(a, 3, None)

    def test_axis0(self):
        a = cupy.arange(12).reshape(3, 4)
        self._check_all_equal(a, 2, 0)

    def test_axis1(self):
        a = cupy.arange(12).reshape(3, 4)
        self._check_all_equal(a, 4, 1)

    def test_rep_zero(self):
        a = cupy.arange(6)
        self._check_all_equal(a, 0, 0)

    def test_rep_one(self):
        a = cupy.arange(6)
        self._check_all_equal(a, 1, 0)

    def test_negative_raises(self):
        a = cupy.arange(3)
        forms = [
            -1,
            numpy.int32(-1),
            [-1],
            cupy.array([-1], dtype=numpy.int32),
            cupy.array(-1),
        ]
        for form in forms:
            with pytest.raises(ValueError, match=r'negative'):
                cupy.repeat(a, form)

    def test_numpy_ndarray_raises_typeerror(self):
        """numpy.ndarray rejected with TypeError."""
        a = cupy.arange(3)
        with pytest.raises(TypeError):
            cupy.repeat(a, numpy.array(2))
        with pytest.raises(TypeError):
            cupy.repeat(a, numpy.array([2]))


class TestRepeatNdarrayRepeatsErrors(unittest.TestCase):
    """ndarray repeats raises appropriate errors."""

    def test_length_mismatch_raises(self):
        a = testing.shaped_arange((4,), cupy)
        with pytest.raises(ValueError, match=r'same length'):
            cupy.repeat(a, cupy.array([1, 2]), axis=0)

    def test_length_mismatch_axis1(self):
        a = testing.shaped_arange((3, 4), cupy)
        with pytest.raises(ValueError, match=r'same length'):
            cupy.repeat(a, cupy.array([1, 2]), axis=1)  # needs 4 elements

    def test_negative_single_raises(self):
        a = testing.shaped_arange((3,), cupy)
        with pytest.raises(ValueError, match=r'negative'):
            cupy.repeat(a, cupy.array([-1, 1, 2]))

    def test_negative_all_raises(self):
        a = testing.shaped_arange((3,), cupy)
        with pytest.raises(ValueError, match=r'negative'):
            cupy.repeat(a, cupy.array([-2, -3, -1]))

    def test_float_dtype_rejected(self):
        a = testing.shaped_arange((3,), cupy)
        with pytest.raises(ValueError, match=r'integer dtype'):
            cupy.repeat(a, cupy.array([1.0, 1.0, 1.0]), axis=0)

    def test_complex_dtype_rejected(self):
        a = testing.shaped_arange((3,), cupy)
        with pytest.raises(ValueError, match=r'integer dtype'):
            cupy.repeat(a, cupy.array([1+0j, 1+0j, 1+0j]), axis=0)

    def test_numpy_ndarray_raises(self):
        """numpy.ndarray is not accepted; users must convert to CuPy first."""
        a = testing.shaped_arange((4,), cupy)
        with pytest.raises((ValueError, TypeError)):
            cupy.repeat(a, numpy.array([1, 2, 3, 4]), axis=0)

    def test_bad_axis_raises(self):
        a = testing.shaped_arange((3, 4), cupy)
        with pytest.raises(Exception):
            cupy.repeat(a, cupy.array([1, 2, 3]), axis=5)

    def test_method_interface_errors(self):
        """Errors raised via ndarray.repeat() mirror cupy.repeat()."""
        a = testing.shaped_arange((4,), cupy)
        with pytest.raises(ValueError, match=r'same length'):
            a.repeat(cupy.array([1, 2]), axis=0)


class TestRepeatNdarrayRepeatsLargeValues(unittest.TestCase):
    """Large individual repeat counts work correctly."""

    @testing.numpy_cupy_array_equal()
    def test_large_single_repeat(self, xp):
        x = testing.shaped_arange((3,), xp)
        return xp.repeat(x, _reps(xp, [0, 100000, 0]))

    @testing.numpy_cupy_array_equal()
    def test_large_broadcast(self, xp):
        x = testing.shaped_arange((3,), xp)
        return xp.repeat(x, _reps(xp, [50000]))

    def test_over_int32_max(self):
        """Cumsum in int32 would overflow; int64 reps handles this correctly.

        Requires ~57 GB of GPU memory (3 x 750M elements x 8 bytes x 3 arrays).
        Skipped on most hardware.
        """
        # 3 elements x 750_000_000 each = 2_250_000_000 > INT32_MAX
        # positions array alone: 2.25G x 8 bytes = 18 GB
        reps_val = 750_000_000
        required_bytes = reps_val * 3 * 8 * 4  # ~72 GB rough estimate
        mem = cupy.cuda.runtime.memGetInfo()
        if mem[0] < required_bytes:
            pytest.skip('insufficient GPU memory for int32-overflow test')
        x = cupy.arange(3, dtype=numpy.int8)
        reps = cupy.array([reps_val, reps_val, reps_val], dtype=numpy.int64)
        result = cupy.repeat(x, reps)
        assert result.shape == (reps_val * 3,)
        assert int(result[0]) == 0
        assert int(result[-1]) == 2


class TestRepeatNdarrayRepeatsOutputShape(unittest.TestCase):
    """Output shape is correct for all axis configurations."""

    def _shapes(self, xp, in_shape, reps, axis):
        x = testing.shaped_arange(in_shape, xp)
        r = _reps(xp, reps)
        return xp.repeat(x, r, axis).shape

    def test_shape_axis0(self):
        s = self._shapes(cupy, (3, 4), [1, 0, 2], axis=0)
        assert s == (3, 4)  # 1+0+2=3 rows

    def test_shape_axis1(self):
        s = self._shapes(cupy, (3, 4), [2, 0, 1, 3], axis=1)
        assert s == (3, 6)  # 2+0+1+3=6 cols

    def test_shape_axis_none(self):
        s = self._shapes(cupy, (2, 3), [1, 2, 3, 0, 1, 2], axis=None)
        assert s == (9,)  # 1+2+3+0+1+2=9

    def test_shape_all_zero(self):
        s = self._shapes(cupy, (3, 4), [0, 0, 0], axis=0)
        assert s == (0, 4)

    def test_shape_broadcast_3d(self):
        s = self._shapes(cupy, (2, 3, 4), [5], axis=2)
        assert s == (2, 3, 20)


class TestRepeatNdarrayRepeatsNumpyEquivalence(unittest.TestCase):
    """Direct values-comparison mirroring numpy's TestRepeat."""

    def _arr(self, xp, data):
        return xp.array(data) if xp is cupy else numpy.array(data)

    @testing.numpy_cupy_array_equal()
    def test_basic_varied(self, xp):
        m = self._arr(xp, [1, 2, 3, 4, 5, 6])
        return xp.repeat(m, _reps(xp, [1, 3, 2, 1, 1, 2]))

    @testing.numpy_cupy_array_equal()
    def test_axis_spec_rows(self, xp):
        m = self._arr(xp, [[1, 2, 3], [4, 5, 6]])
        return xp.repeat(m, _reps(xp, [2, 1]), axis=0)

    @testing.numpy_cupy_array_equal()
    def test_axis_spec_cols(self, xp):
        m = self._arr(xp, [[1, 2, 3], [4, 5, 6]])
        return xp.repeat(m, _reps(xp, [1, 3, 2]), axis=1)

    @testing.numpy_cupy_array_equal()
    def test_broadcast_rows(self, xp):
        m = self._arr(xp, [[1, 2, 3], [4, 5, 6]])
        return xp.repeat(m, _reps(xp, [2]), axis=0)

    @testing.numpy_cupy_array_equal()
    def test_broadcast_cols(self, xp):
        m = self._arr(xp, [[1, 2, 3], [4, 5, 6]])
        return xp.repeat(m, _reps(xp, [2]), axis=1)

    @testing.numpy_cupy_array_equal()
    def test_discont_regression(self, xp):
        """Regression: non-contiguous input (numpy test_repeat_discont)."""
        a = testing.shaped_arange((4, 3), xp)[:, 2]
        return xp.repeat(a, _reps(xp, [3, 3, 3, 3]))

    @testing.numpy_cupy_array_equal()
    def test_broadcast_list_equiv(self, xp):
        """gh-5743: broadcast via size-1 list vs scalar should match."""
        a = testing.shaped_arange((3, 4, 5), xp)
        return xp.repeat(a, _reps(xp, [2]), axis=1)


@testing.parameterize(
    {'repeats': [2], 'axis': None},
    {'repeats': [2], 'axis': 1},
)
class TestRepeatListBroadcast(unittest.TestCase):

    """Test for `repeats` argument using single element list.

    This feature is only supported in NumPy 1.10 or later.
    """

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': 0, 'axis': None},
    {'repeats': 2, 'axis': None},
    {'repeats': 2, 'axis': 0},
    {'repeats': [1, 2, 3, 4], 'axis': None},
    {'repeats': [1, 2, 3, 4], 'axis': 0},
)
class TestRepeat1D(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((4,), xp)
        return xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': [2], 'axis': None},
    {'repeats': [2], 'axis': 0},
)
class TestRepeat1DListBroadcast(unittest.TestCase):

    """See comment in TestRepeatListBroadcast class."""

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((4,), xp)
        return xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': -3, 'axis': None},
    {'repeats': [-3, -3], 'axis': 0},
    {'repeats': [1, 2, 3], 'axis': None},
    {'repeats': [1, 2], 'axis': 1},
    {'repeats': 2, 'axis': -4},
    {'repeats': 2, 'axis': 3},
)
class TestRepeatFailure(unittest.TestCase):

    def test_repeat_failure(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'reps': 0},
    {'reps': 1},
    {'reps': 2},
    {'reps': (0, 1)},
    {'reps': (2, 3)},
    {'reps': (2, 3, 4, 5)},
)
class TestTile(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_array_tile(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.tile(x, self.reps)


@testing.parameterize(
    {'reps': -1},
    {'reps': (-1, -2)},
)
class TestTileFailure(unittest.TestCase):

    def test_tile_failure(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.tile(x, -3)
