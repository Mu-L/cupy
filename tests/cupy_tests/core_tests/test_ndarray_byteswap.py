from __future__ import annotations

import pytest

from cupy import testing


@pytest.mark.parametrize('shape', [
    (),
    (0,),
    (10,),
    (2, 3),
    (2, 3, 4),
    (2, 3, 4, 5),
])
@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('order', ['C', 'F'])
class TestByteswap:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_byteswap(self, xp, dtype, shape, inplace, order):
        a = xp.array(
            testing.shaped_arange(shape, xp, dtype),
            order=order,
        )
        b = a.byteswap(inplace=inplace)
        if inplace:
            assert b is a
        return b
