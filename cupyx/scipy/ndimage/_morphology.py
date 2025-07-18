from __future__ import annotations

import math
import operator
import warnings

import numpy

import cupy
from cupy import _core

from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _measurements
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters


@cupy.memoize(for_each_device=True)
def _get_binary_erosion_kernel(
    w_shape, int_type, offsets, center_is_true, border_value, invert, masked,
    all_weights_nonzero
):
    if invert:
        border_value = int(not border_value)
        true_val = 0
        false_val = 1
    else:
        border_value = int(border_value)
        true_val = 1
        false_val = 0

    if masked:
        pre = f"""
            bool mv = (bool)mask[i];
            bool _in = (bool)x[i];
            if (!mv) {{
                y = cast<Y>(_in);
                return;
            }} else if ({int(center_is_true)} && _in == {false_val}) {{
                y = cast<Y>(_in);
                return;
            }}"""
    else:
        pre = f"""
            bool _in = (bool)x[i];
            if ({int(center_is_true)} && _in == {false_val}) {{
                y = cast<Y>(_in);
                return;
            }}"""
    pre = pre + f"""
            y = cast<Y>({true_val});"""

    # {{{{ required because format is called again within _generate_nd_kernel
    found = f"""
        if ({{cond}}) {{{{
            if (!{border_value}) {{{{
                y = cast<Y>({false_val});
                return;
            }}}}
        }}}} else {{{{
            bool nn = {{value}} ? {true_val} : {false_val};
            if (!nn) {{{{
                y = cast<Y>({false_val});
                return;
            }}}}
        }}}}"""

    name = 'binary_erosion'
    if false_val:
        name += '_invert'
    has_weights = not all_weights_nonzero

    modes = ('constant',) * len(w_shape)
    return _filters_core._generate_nd_kernel(
        name,
        pre,
        found,
        '',
        modes, w_shape, int_type, offsets, 0, ctype='Y',
        has_weights=has_weights, has_structure=False, has_mask=masked,
        binary_morphology=True)


def _center_is_true(structure, origin):
    coor = tuple([oo + ss // 2 for ss, oo in zip(structure.shape, origin)])
    return bool(structure[coor])  # device synchronization


def iterate_structure(structure, iterations, origin=None):
    """Iterate a structure by dilating it with itself.

    Args:
        structure(array_like): Structuring element (an array of bools,
            for example), to be dilated with itself.
        iterations(int): The number of dilations performed on the structure
            with itself.
        origin(int or tuple of int, optional): If origin is None, only the
            iterated structure is returned. If not, a tuple of the iterated
            structure and the modified origin is returned.

    Returns:
        cupy.ndarray: A new structuring element obtained by dilating
        ``structure`` (``iterations`` - 1) times with itself.

    .. seealso:: :func:`scipy.ndimage.iterate_structure`
    """
    if iterations < 2:
        return structure.copy()
    ni = iterations - 1
    shape = [ii + ni * (ii - 1) for ii in structure.shape]
    pos = [ni * (structure.shape[ii] // 2) for ii in range(len(shape))]
    slc = tuple(
        slice(pos[ii], pos[ii] + structure.shape[ii], None)
        for ii in range(len(shape))
    )
    out = cupy.zeros(shape, bool)
    out[slc] = structure != 0
    out = binary_dilation(out, structure, iterations=ni)
    if origin is None:
        return out
    else:
        origin = _util._fix_sequence_arg(origin, structure.ndim, 'origin', int)
        origin = [iterations * o for o in origin]
        return out, origin


def generate_binary_structure(rank, connectivity):
    """Generate a binary structure for binary morphological operations.

    Args:
        rank(int): Number of dimensions of the array to which the structuring
            element will be applied, as returned by ``np.ndim``.
        connectivity(int): ``connectivity`` determines which elements of the
            output array belong to the structure, i.e., are considered as
            neighbors of the central element. Elements up to a squared distance
            of ``connectivity`` from the center are considered neighbors.
            ``connectivity`` may range from 1 (no diagonal elements are
            neighbors) to ``rank`` (all elements are neighbors).

    Returns:
        cupy.ndarray: Structuring element which may be used for binary
        morphological operations, with ``rank`` dimensions and all
        dimensions equal to 3.

    .. seealso:: :func:`scipy.ndimage.generate_binary_structure`
    """
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return cupy.asarray(True, dtype=bool)
    output = numpy.fabs(numpy.indices([3] * rank) - 1)
    output = numpy.add.reduce(output, 0)
    output = output <= connectivity
    return cupy.asarray(output)


def _binary_erosion(input, structure, iterations, mask, output, border_value,
                    origin, invert, brute_force=True, *, axes=None):
    try:
        iterations = operator.index(iterations)
    except TypeError:
        raise TypeError('iterations parameter should be an integer')

    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported')
    ndim = input.ndim
    axes = _util._check_axes(axes, ndim)
    num_axes = len(axes)
    if structure is None:
        structure = generate_binary_structure(num_axes, 1)
        all_weights_nonzero = input.ndim == 1
        center_is_true = True
        structure_shape = structure.shape
    elif isinstance(structure, tuple):
        # For a structure that is true everywhere, can just provide the shape
        structure_shape = structure
        if len(structure_shape) == 0:
            raise RuntimeError("structure must not be empty")
    else:
        structure = structure.astype(dtype=bool, copy=False)
        structure_shape = structure.shape
        # transfer to CPU for use in determining if it is fully dense
        # structure_cpu = cupy.asnumpy(structure)
        if structure.ndim != num_axes:
            raise RuntimeError(
                'structure and input must have same dimensionality')
        if not structure.flags.c_contiguous:
            structure = cupy.ascontiguousarray(structure)
        if structure.size < 1:
            raise RuntimeError('structure must not be empty')

    if num_axes < ndim:
        structure = _util._expand_footprint(
            ndim, axes, structure, footprint_name="structure"
        )
        structure_shape = structure.shape

    if mask is not None:
        if mask.shape != input.shape:
            raise RuntimeError('mask and input must have equal sizes')
        if not mask.flags.c_contiguous:
            mask = cupy.ascontiguousarray(mask)
        masked = True
    else:
        masked = False
    origin = _util._fix_sequence_arg(origin, num_axes, 'origin', int)
    if num_axes < ndim:
        origin = _util._expand_origin(ndim, axes, origin)

    if isinstance(output, cupy.ndarray):
        if output.dtype.kind == 'c':
            raise TypeError('Complex output type not supported')
    else:
        output = bool
    output = _util._get_output(output, input)
    temp_needed = cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS')
    if temp_needed:
        # input and output arrays cannot share memory
        temp = output
        output = _util._get_output(output.dtype, input)
    if len(structure_shape) == 0:
        # kernel doesn't handle ndim=0, so special case it here
        if isinstance(structure, tuple) or float(structure):
            output[...] = cupy.asarray(input, dtype=bool)
        else:
            output[...] = ~cupy.asarray(input, dtype=bool)
        return output
    origin = tuple(origin)
    int_type = _util._get_inttype(input)
    offsets = _filters_core._origins_to_offsets(origin, structure_shape)
    if isinstance(structure, tuple):
        nnz = math.prod(structure_shape)
        all_weights_nonzero = True
        center_is_true = True
    else:
        # synchronize required to determine if all weights are non-zero
        nnz = int(cupy.count_nonzero(structure))
        all_weights_nonzero = nnz == structure.size
        if all_weights_nonzero:
            center_is_true = True
        else:
            center_is_true = _center_is_true(structure, origin)

    erode_kernel = _get_binary_erosion_kernel(
        structure_shape, int_type, offsets, center_is_true, border_value,
        invert, masked, all_weights_nonzero,
    )
    if all_weights_nonzero:
        if masked:
            in_args = (input, mask)
        else:
            in_args = (input,)
    else:
        if masked:
            in_args = (input, structure, mask)
        else:
            in_args = (input, structure)

    if iterations == 1:
        output = erode_kernel(*in_args, output)
    elif center_is_true and not brute_force:
        raise NotImplementedError(
            'only brute_force iteration has been implemented'
        )
    else:
        if cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS'):
            raise ValueError('output and input may not overlap in memory')
        tmp_in = cupy.empty_like(input, dtype=output.dtype)
        tmp_out = output
        if iterations >= 1 and not iterations & 1:
            tmp_in, tmp_out = tmp_out, tmp_in
        tmp_out = erode_kernel(*in_args, tmp_out)
        # TODO: kernel doesn't return the changed status, so determine it here
        changed = not (input == tmp_out).all()  # synchronize!
        ii = 1
        while ii < iterations or ((iterations < 1) and changed):
            tmp_in, tmp_out = tmp_out, tmp_in
            if all_weights_nonzero:
                if masked:
                    in_args = (tmp_in, mask)
                else:
                    in_args = (tmp_in,)
            else:
                if masked:
                    in_args = (tmp_in, structure, mask)
                else:
                    in_args = (tmp_in, structure)
            tmp_out = erode_kernel(*in_args, tmp_out)
            changed = not (tmp_in == tmp_out).all()
            ii += 1
            if not changed and (not ii & 1):  # synchronize!
                # can exit early if nothing changed
                # (only do this after even number of tmp_in/out swaps)
                break
        output = tmp_out
    if temp_needed:
        _core.elementwise_copy(output, temp)
        output = temp
    return output


def _prep_structure(structure, ndim):
    if structure is None:
        structure = generate_binary_structure(ndim, 1)
        return structure, structure.shape, True
    if isinstance(structure, int):
        structure = (structure,) * ndim
    elif isinstance(structure, list):
        structure = tuple(structure)
    if isinstance(structure, tuple):
        symmetric_structure = True
        structure_shape = structure
    else:
        # if user-provided, it is not guaranteed to be symmetric
        symmetric_structure = False
        structure_shape = structure.shape
    return structure, structure_shape, symmetric_structure


def binary_erosion(input, structure=None, iterations=1, mask=None, output=None,
                   border_value=0, origin=0, brute_force=False, *, axes=None):
    """Multidimensional binary erosion with a given structuring element.

    Binary erosion is a mathematical morphology operation used for image
    processing.

    Args:
        input(cupy.ndarray): The input binary array_like to be eroded.
            Non-zero (True) elements form the subset to be eroded.
        structure(cupy.ndarray or tuple or int, optional): The structuring
            element used for the erosion. Non-zero elements are considered
            true. If no structuring element is provided an element is
            generated with a square connectivity equal to one. If a tuple of
            integers is provided, a structuring element of the specified shape
            is used (all elements true). If an integer is provided, the
            structuring element will have the same size along all axes.
            (Default value = None).
        iterations(int, optional): The erosion is repeated ``iterations`` times
            (one, by default). If iterations is less than 1, the erosion is
            repeated until the result does not change anymore. Only an integer
            of iterations is accepted.
        mask(cupy.ndarray or None, optional): If a mask is given, only those
            elements with a True value at the corresponding mask element are
            modified at each iteration. (Default value = None)
        output(cupy.ndarray, optional): Array of the same shape as input, into
            which the output is placed. By default, a new array is created.
        border_value(int (cast to 0 or 1), optional): Value at the
            border in the output array. (Default value = 0)
        origin(int or tuple of ints, optional): Placement of the filter, by
            default 0.
        brute_force(boolean, optional): Memory condition: if False, only the
            pixels whose value was changed in the last iteration are tracked as
            candidates to be updated (eroded) in the current iteration; if
            True all pixels are considered as candidates for erosion,
            regardless of what happened in the previous iteration.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: The result of binary erosion.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.binary_erosion`
    """
    axes = _util._check_axes(axes, input.ndim)
    structure, _, _ = _prep_structure(structure, len(axes))
    return _binary_erosion(input, structure, iterations, mask, output,
                           border_value, origin, 0, brute_force, axes=axes)


def binary_dilation(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0, origin=0, brute_force=False,
                    *, axes=None):
    """Multidimensional binary dilation with the given structuring element.

    Args:
        input(cupy.ndarray): The input binary array_like to be dilated.
            Non-zero (True) elements form the subset to be dilated.
        structure(cupy.ndarray or tuple or int, optional): The structuring
            element used for the dilation. Non-zero elements are considered
            true. If no structuring element is provided an element is
            generated with a square connectivity equal to one. If a tuple of
            integers is provided, a structuring element of the specified shape
            is used (all elements true). If an integer is provided, the
            structuring element will have the same size along all axes.
            (Default value = None).
        iterations(int, optional): The dilation is repeated ``iterations``
            times (one, by default). If iterations is less than 1, the dilation
            is repeated until the result does not change anymore. Only an
            integer of iterations is accepted.
        mask(cupy.ndarray or None, optional): If a mask is given, only those
            elements with a True value at the corresponding mask element are
            modified at each iteration. (Default value = None)
        output(cupy.ndarray, optional): Array of the same shape as input, into
            which the output is placed. By default, a new array is created.
        border_value(int (cast to 0 or 1), optional): Value at the
            border in the output array. (Default value = 0)
        origin(int or tuple of ints, optional): Placement of the filter, by
            default 0.
        brute_force(boolean, optional): Memory condition: if False, only the
            pixels whose value was changed in the last iteration are tracked as
            candidates to be updated (dilated) in the current iteration; if
            True all pixels are considered as candidates for dilation,
            regardless of what happened in the previous iteration.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: The result of binary dilation.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.binary_dilation`
    """
    axes = _util._check_axes(axes, input.ndim)
    structure, structure_shape, symmetric = _prep_structure(structure,
                                                            len(axes))
    origin = _util._fix_sequence_arg(origin, len(axes), 'origin', int)
    # no point in flipping if already symmetric
    if not symmetric:
        structure = structure[tuple([slice(None, None, -1)] * structure.ndim)]
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure_shape[ii] & 1:
            origin[ii] -= 1
    return _binary_erosion(input, structure, iterations, mask, output,
                           border_value, origin, 1, brute_force, axes=axes)


def binary_opening(input, structure=None, iterations=1, output=None, origin=0,
                   mask=None, border_value=0, brute_force=False, *, axes=None):
    """
    Multidimensional binary opening with the given structuring element.

    The *opening* of an input image by a structuring element is the
    *dilation* of the *erosion* of the image by the structuring element.

    Args:
        input(cupy.ndarray): The input binary array to be opened.
            Non-zero (True) elements form the subset to be opened.
        structure(cupy.ndarray or tuple or int, optional): The structuring
            element used for the opening. Non-zero elements are considered
            true. If no structuring element is provided an element is
            generated with a square connectivity equal to one. If a tuple of
            integers is provided, a structuring element of the specified shape
            is used (all elements true). If an integer is provided, the
            structuring element will have the same size along all axes.
            (Default value = None).
        iterations(int, optional): The opening is repeated ``iterations`` times
            (one, by default). If iterations is less than 1, the opening is
            repeated until the result does not change anymore. Only an integer
            of iterations is accepted.
        output(cupy.ndarray, optional): Array of the same shape as input, into
            which the output is placed. By default, a new array is created.
        origin(int or tuple of ints, optional): Placement of the filter, by
            default 0.
        mask(cupy.ndarray or None, optional): If a mask is given, only those
            elements with a True value at the corresponding mask element are
            modified at each iteration. (Default value = None)
        border_value(int (cast to 0 or 1), optional): Value at the
            border in the output array. (Default value = 0)
        brute_force(boolean, optional): Memory condition: if False, only the
            pixels whose value was changed in the last iteration are tracked as
            candidates to be updated (dilated) in the current iteration; if
            True all pixels are considered as candidates for opening,
            regardless of what happened in the previous iteration.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: The result of binary opening.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.binary_opening`
    """
    axes = _util._check_axes(axes, input.ndim)
    structure, _, _ = _prep_structure(structure, len(axes))
    tmp = binary_erosion(input, structure, iterations, mask, None,
                         border_value, origin, brute_force, axes=axes)
    return binary_dilation(tmp, structure, iterations, mask, output,
                           border_value, origin, brute_force, axes=axes)


def binary_closing(input, structure=None, iterations=1, output=None, origin=0,
                   mask=None, border_value=0, brute_force=False, *, axes=None):
    """
    Multidimensional binary closing with the given structuring element.

    The *closing* of an input image by a structuring element is the
    *erosion* of the *dilation* of the image by the structuring element.

    Args:
        input(cupy.ndarray): The input binary array to be closed.
            Non-zero (True) elements form the subset to be closed.
        structure(cupy.ndarray or tuple or int, optional): The structuring
            element used for the closing. Non-zero elements are considered
            true. If no structuring element is provided an element is
            generated with a square connectivity equal to one. If a tuple of
            integers is provided, a structuring element of the specified shape
            is used (all elements true). If an integer is provided, the
            structuring element will have the same size along all axes.
            (Default value = None).
        iterations(int, optional): The closing is repeated ``iterations`` times
            (one, by default). If iterations is less than 1, the closing is
            repeated until the result does not change anymore. Only an integer
            of iterations is accepted.
        output(cupy.ndarray, optional): Array of the same shape as input, into
            which the output is placed. By default, a new array is created.
        origin(int or tuple of ints, optional): Placement of the filter, by
            default 0.
        mask(cupy.ndarray or None, optional): If a mask is given, only those
            elements with a True value at the corresponding mask element are
            modified at each iteration. (Default value = None)
        border_value(int (cast to 0 or 1), optional): Value at the
            border in the output array. (Default value = 0)
        brute_force(boolean, optional): Memory condition: if False, only the
            pixels whose value was changed in the last iteration are tracked as
            candidates to be updated (dilated) in the current iteration; if
            True all pixels are considered as candidates for closing,
            regardless of what happened in the previous iteration.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: The result of binary closing.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.binary_closing`
    """
    axes = _util._check_axes(axes, input.ndim)
    structure, _, _ = _prep_structure(structure, len(axes))
    tmp = binary_dilation(input, structure, iterations, mask, None,
                          border_value, origin, brute_force, axes=axes)
    return binary_erosion(tmp, structure, iterations, mask, output,
                          border_value, origin, brute_force, axes=axes)


def binary_hit_or_miss(input, structure1=None, structure2=None, output=None,
                       origin1=0, origin2=None, *, axes=None):
    """
    Multidimensional binary hit-or-miss transform.

    The hit-or-miss transform finds the locations of a given pattern
    inside the input image.

    Args:
        input (cupy.ndarray): Binary image where a pattern is to be detected.
        structure1 (cupy.ndarray, optional): Part of the structuring element to
            be fitted to the foreground (non-zero elements) of ``input``. If no
            value is provided, a structure of square connectivity 1 is chosen.
        structure2 (cupy.ndarray, optional): Second part of the structuring
            element that has to miss completely the foreground. If no value is
            provided, the complementary of ``structure1`` is taken.
        output (cupy.ndarray, dtype or None, optional): Array of the same shape
            as input, into which the output is placed. By default, a new array
            is created.
        origin1 (int or tuple of ints, optional): Placement of the first part
            of the structuring element ``structure1``, by default 0 for a
            centered structure.
        origin2 (int or tuple of ints or None, optional): Placement of the
            second part of the structuring element ``structure2``, by default 0
            for a centered structure. If a value is provided for ``origin1``
            and not for ``origin2``, then ``origin2`` is set to ``origin1``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: Hit-or-miss transform of ``input`` with the given
        structuring element (``structure1``, ``structure2``).

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.binary_hit_or_miss`
    """
    axes = _util._check_axes(axes, input.ndim)
    num_axes = len(axes)
    if structure1 is None:
        structure1 = generate_binary_structure(num_axes, 1)
    if structure2 is None:
        structure2 = cupy.logical_not(structure1)
    origin1 = _util._fix_sequence_arg(origin1, num_axes, 'origin1', int)
    if origin2 is None:
        origin2 = origin1
    else:
        origin2 = _util._fix_sequence_arg(origin2, num_axes, 'origin2', int)

    tmp1 = _binary_erosion(input, structure1, 1, None, None, 0, origin1, 0,
                           False, axes=axes)
    inplace = isinstance(output, cupy.ndarray)
    result = _binary_erosion(input, structure2, 1, None, output, 0, origin2, 1,
                             False, axes=axes)
    if inplace:
        cupy.logical_not(output, output)
        cupy.logical_and(tmp1, output, output)
    else:
        cupy.logical_not(result, result)
        return cupy.logical_and(tmp1, result)


def binary_propagation(input, structure=None, mask=None, output=None,
                       border_value=0, origin=0, *, axes=None):
    """
    Multidimensional binary propagation with the given structuring element.

    Args:
        input (cupy.ndarray): Binary image to be propagated inside ``mask``.
        structure (cupy.ndarray, optional): Structuring element used in the
            successive dilations. The output may depend on the structuring
            element, especially if ``mask`` has several connex components. If
            no structuring element is provided, an element is generated with a
            squared connectivity equal to one.
        mask (cupy.ndarray, optional): Binary mask defining the region into
            which ``input`` is allowed to propagate.
        output (cupy.ndarray, optional): Array of the same shape as input, into
            which the output is placed. By default, a new array is created.
        border_value (int, optional): Value at the border in the output array.
            The value is cast to 0 or 1.
        origin (int or tuple of ints, optional): Placement of the filter.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray : Binary propagation of ``input`` inside ``mask``.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.binary_propagation`
    """
    return binary_dilation(input, structure, -1, mask, output, border_value,
                           origin, brute_force=True, axes=axes)


def _binary_fill_holes_non_iterative(input, structure=None, output=None):
    """Non-iterative method for hole filling.

    This algorithm is based on inverting the input and then using `label` to
    label the holes distinctly from the background. This information is then
    used to create a holes mask which can be applied to fill the holes in the
    original input.

    Initial benchmarks indicate this is a faster approach than calling
    binary_dilation iteratively:

    https://github.com/cupy/cupy/issues/8867#issuecomment-2659471046
    """

    # make sure all background pixels at the boundary have the same label
    if input.dtype == cupy.uint8:
        input = input.view(bool)
    elif input.dtype != bool:
        input = input.astype(bool)
    ndim = input.ndim
    binary_mask = cupy.pad(input, 1, mode='constant', constant_values=0)

    # assign unique labels the background and holes
    inverse_binary_mask = ~binary_mask
    inverse_labels, _ = _measurements.label(
        inverse_binary_mask, structure=structure
    )

    # After inversion, what was originally the background will now be the
    # first foreground label encountered. This is ensured due to the
    # single voxel padding done above and the fact that the `label`
    # function scans linearly through the array.
    background_index = 1
    # set the background back to 0 in the inverse mask so we have a mask
    # of just the holes
    inverse_binary_mask[inverse_labels == background_index] = 0

    # add binary holes to the original mask and relabel
    temp = cupy.logical_or(binary_mask, inverse_binary_mask)

    remove_padding = (slice(1, -1),) * ndim
    temp = temp[remove_padding]
    if output is None:
        output = cupy.ascontiguousarray(temp)
    else:
        # handle output argument as in _binary_erosion
        if isinstance(output, cupy.ndarray):
            if output.dtype.kind == 'c':
                raise TypeError('Complex output type not supported')
        else:
            output = bool
        output = _util._get_output(output, input)
        output[:] = temp
    return output


def binary_fill_holes(input, structure=None, output=None, origin=0, *,
                      axes=None):
    """Fill the holes in binary objects.

    Args:
        input (cupy.ndarray): N-D binary array with holes to be filled.
        structure (cupy.ndarray, optional):  For CuPy, it is recommended to
            leave this None so that a faster non-iterative algorithm will be
            used. This default is equivalent in behavior to the default
            structure use by SciPy. If `structure` array is provided, the
            relatively slow iterative algorithm from SciPy will be used.
            In that case, a larger-size structure can make the iterative
            computations faster, but may miss holes separated from the
            background by thin regions. The default element (with a square
            connectivity equal to one) yields the intuitive result where all
            holes in the input have been filled.
        output (cupy.ndarray, dtype or None, optional): Array of the same shape
            as input, into which the output is placed. By default, a new array
            is created.
        origin (int, tuple of ints, optional): Position of the structuring
            element. Note that if this is changed from its default value of
            0, it will force a slower iterative algorithm to be used.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: Transformation of the initial image ``input`` where holes
        have been filled.

    .. warning::

        This function may synchronize the device.

    .. warning::

        It is recommended to keep the default setting of output=None and
        origin==0 so that a faster, non-iterative algorithm can be used.

    .. seealso:: :func:`scipy.ndimage.binary_fill_holes`
    """
    axes = _util._check_axes(axes, input.ndim)
    filter_all_axes = axes == tuple(range(input.ndim))
    if isinstance(origin, int):
        origin = (origin,) * len(axes)
    if all(o == 0 for o in origin) and filter_all_axes:
        return _binary_fill_holes_non_iterative(
            input, structure=structure, output=output)
    elif filter_all_axes:
        warnings.warn(
            'It is recommended to keep the default origin=0 so that a faster '
            'non-iterative algorithm can be used.'
        )
    mask = cupy.logical_not(input)
    tmp = cupy.zeros(mask.shape, bool)
    inplace = isinstance(output, cupy.ndarray)
    # TODO (grlee77): set brute_force=False below once implemented
    if inplace:
        binary_dilation(tmp, structure, -1, mask, output, 1, origin,
                        brute_force=True, axes=axes)
        cupy.logical_not(output, output)
    else:
        output = binary_dilation(tmp, structure, -1, mask, None, 1, origin,
                                 brute_force=True, axes=axes)
        cupy.logical_not(output, output)
        return output


def grey_erosion(input, size=None, footprint=None, structure=None, output=None,
                 mode='reflect', cval=0.0, origin=0, *, axes=None):
    """Calculates a greyscale erosion.

    Args:
        input (cupy.ndarray): The input array.
        size (tuple of ints): Shape of a flat and full structuring element used
            for the greyscale erosion. Optional if ``footprint`` or
            ``structure`` is provided.
        footprint (array of ints): Positions of non-infinite elements of a flat
            structuring element used for greyscale erosion. Non-zero values
            give the set of neighbors of the center over which minimum is
            chosen.
        structure (array of ints): Structuring element used for the greyscale
            erosion. ``structure`` may be a non-flat structuring element.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: The result of greyscale erosion.

    .. seealso:: :func:`scipy.ndimage.grey_erosion`
    """

    if size is None and footprint is None and structure is None:
        raise ValueError('size, footprint or structure must be specified')

    return _filters._min_or_max_filter(input, size, footprint, structure,
                                       output, mode, cval, origin, 'min',
                                       axes=axes)


def grey_dilation(input, size=None, footprint=None, structure=None,
                  output=None, mode='reflect', cval=0.0, origin=0, *,
                  axes=None):
    """Calculates a greyscale dilation.

    Args:
        input (cupy.ndarray): The input array.
        size (tuple of ints): Shape of a flat and full structuring element used
            for the greyscale dilation. Optional if ``footprint`` or
            ``structure`` is provided.
        footprint (array of ints): Positions of non-infinite elements of a flat
            structuring element used for greyscale dilation. Non-zero values
            give the set of neighbors of the center over which maximum is
            chosen.
        structure (array of ints): Structuring element used for the greyscale
            dilation. ``structure`` may be a non-flat structuring element.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: The result of greyscale dilation.

    .. seealso:: :func:`scipy.ndimage.grey_dilation`
    """

    if size is None and footprint is None and structure is None:
        raise ValueError('size, footprint or structure must be specified')
    if structure is not None:
        structure = cupy.array(structure)
        structure = structure[tuple([slice(None, None, -1)] * structure.ndim)]
    if footprint is not None:
        footprint = cupy.array(footprint)
        footprint = footprint[tuple([slice(None, None, -1)] * footprint.ndim)]

    axes = _util._check_axes(axes, input.ndim)
    origin = _util._fix_sequence_arg(origin, len(axes), 'origin', int)
    for i in range(len(origin)):
        origin[i] = -origin[i]
        if footprint is not None:
            sz = footprint.shape[i]
        elif structure is not None:
            sz = structure.shape[i]
        elif numpy.isscalar(size):
            sz = size
        else:
            sz = size[i]
        if sz % 2 == 0:
            origin[i] -= 1

    return _filters._min_or_max_filter(input, size, footprint, structure,
                                       output, mode, cval, origin, 'max',
                                       axes=axes)


def grey_closing(input, size=None, footprint=None, structure=None,
                 output=None, mode='reflect', cval=0.0, origin=0, *,
                 axes=None):
    """Calculates a multi-dimensional greyscale closing.

    Args:
        input (cupy.ndarray): The input array.
        size (tuple of ints): Shape of a flat and full structuring element used
            for the greyscale closing. Optional if ``footprint`` or
            ``structure`` is provided.
        footprint (array of ints): Positions of non-infinite elements of a flat
            structuring element used for greyscale closing. Non-zero values
            give the set of neighbors of the center over which closing is
            chosen.
        structure (array of ints): Structuring element used for the greyscale
            closing. ``structure`` may be a non-flat structuring element.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: The result of greyscale closing.

    .. seealso:: :func:`scipy.ndimage.grey_closing`
    """
    if (size is not None) and (footprint is not None):
        warnings.warn('ignoring size because footprint is set', UserWarning,
                      stacklevel=2)
    kwargs = dict(mode=mode, cval=cval, origin=origin, axes=axes)
    tmp = grey_dilation(input, size, footprint, structure, None, **kwargs)
    return grey_erosion(tmp, size, footprint, structure, output, **kwargs)


def grey_opening(input, size=None, footprint=None, structure=None,
                 output=None, mode='reflect', cval=0.0, origin=0, *,
                 axes=None):
    """Calculates a multi-dimensional greyscale opening.

    Args:
        input (cupy.ndarray): The input array.
        size (tuple of ints): Shape of a flat and full structuring element used
            for the greyscale opening. Optional if ``footprint`` or
            ``structure`` is provided.
        footprint (array of ints): Positions of non-infinite elements of a flat
            structuring element used for greyscale opening. Non-zero values
            give the set of neighbors of the center over which opening is
            chosen.
        structure (array of ints): Structuring element used for the greyscale
            opening. ``structure`` may be a non-flat structuring element.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: The result of greyscale opening.

    .. seealso:: :func:`scipy.ndimage.grey_opening`
    """
    if (size is not None) and (footprint is not None):
        warnings.warn('ignoring size because footprint is set', UserWarning,
                      stacklevel=2)
    kwargs = dict(mode=mode, cval=cval, origin=origin, axes=axes)
    tmp = grey_erosion(input, size, footprint, structure, None, **kwargs)
    return grey_dilation(tmp, size, footprint, structure, output, **kwargs)


def morphological_gradient(
    input,
    size=None,
    footprint=None,
    structure=None,
    output=None,
    mode='reflect',
    cval=0.0,
    origin=0,
    *,
    axes=None,
):
    """
    Multidimensional morphological gradient.

    The morphological gradient is calculated as the difference between a
    dilation and an erosion of the input with a given structuring element.

    Args:
        input (cupy.ndarray): The input array.
        size (tuple of ints): Shape of a flat and full structuring element used
            for the morphological gradient. Optional if ``footprint`` or
            ``structure`` is provided.
        footprint (array of ints): Positions of non-infinite elements of a flat
            structuring element used for morphological gradient. Non-zero
            values give the set of neighbors of the center over which opening
            is chosen.
        structure (array of ints): Structuring element used for the
            morphological gradient. ``structure`` may be a non-flat
            structuring element.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: The morphological gradient of the input.

    .. seealso:: :func:`scipy.ndimage.morphological_gradient`
    """
    kwargs = dict(mode=mode, cval=cval, origin=origin, axes=axes)
    tmp = grey_dilation(input, size, footprint, structure, None, **kwargs)
    if isinstance(output, cupy.ndarray):
        grey_erosion(input, size, footprint, structure, output, **kwargs)
        return cupy.subtract(tmp, output, output)
    else:
        return tmp - grey_erosion(
            input, size, footprint, structure, None, **kwargs
        )


def morphological_laplace(
    input,
    size=None,
    footprint=None,
    structure=None,
    output=None,
    mode='reflect',
    cval=0.0,
    origin=0,
    *,
    axes=None,
):
    """
    Multidimensional morphological laplace.

    Args:
        input (cupy.ndarray): The input array.
        size (tuple of ints): Shape of a flat and full structuring element used
            for the morphological laplace. Optional if ``footprint`` or
            ``structure`` is provided.
        footprint (array of ints): Positions of non-infinite elements of a flat
            structuring element used for morphological laplace. Non-zero
            values give the set of neighbors of the center over which opening
            is chosen.
        structure (array of ints): Structuring element used for the
            morphological laplace. ``structure`` may be a non-flat
            structuring element.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: The morphological laplace of the input.

    .. seealso:: :func:`scipy.ndimage.morphological_laplace`
    """
    kwargs = dict(mode=mode, cval=cval, origin=origin, axes=axes)
    tmp1 = grey_dilation(input, size, footprint, structure, None, **kwargs)
    if isinstance(output, cupy.ndarray):
        grey_erosion(input, size, footprint, structure, output, **kwargs)
        cupy.add(tmp1, output, output)
        cupy.subtract(output, input, output)
        return cupy.subtract(output, input, output)
    else:
        tmp2 = grey_erosion(input, size, footprint, structure, None, **kwargs)
        cupy.add(tmp1, tmp2, tmp2)
        cupy.subtract(tmp2, input, tmp2)
        cupy.subtract(tmp2, input, tmp2)
        return tmp2


def white_tophat(
    input,
    size=None,
    footprint=None,
    structure=None,
    output=None,
    mode='reflect',
    cval=0.0,
    origin=0,
    *,
    axes=None,
):
    """
    Multidimensional white tophat filter.

    Args:
        input (cupy.ndarray): The input array.
        size (tuple of ints): Shape of a flat and full structuring element used
            for the white tophat. Optional if ``footprint`` or ``structure`` is
            provided.
        footprint (array of ints): Positions of non-infinite elements of a flat
            structuring element used for the white tophat. Non-zero values
            give the set of neighbors of the center over which opening is
            chosen.
        structure (array of ints): Structuring element used for the white
            tophat. ``structure`` may be a non-flat structuring element.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarray: Result of the filter of ``input`` with ``structure``.

    .. seealso:: :func:`scipy.ndimage.white_tophat`
    """
    if (size is not None) and (footprint is not None):
        warnings.warn(
            'ignoring size because footprint is set', UserWarning, stacklevel=2
        )
    kwargs = dict(mode=mode, cval=cval, origin=origin, axes=axes)
    tmp = grey_erosion(input, size, footprint, structure, None, **kwargs)
    tmp = grey_dilation(tmp, size, footprint, structure, output, **kwargs)
    if input.dtype == numpy.bool_ and tmp.dtype == numpy.bool_:
        cupy.bitwise_xor(input, tmp, out=tmp)
    else:
        cupy.subtract(input, tmp, out=tmp)
    return tmp


def black_tophat(
    input,
    size=None,
    footprint=None,
    structure=None,
    output=None,
    mode='reflect',
    cval=0.0,
    origin=0,
    *,
    axes=None,
):
    """
    Multidimensional black tophat filter.

    Args:
        input (cupy.ndarray): The input array.
        size (tuple of ints): Shape of a flat and full structuring element used
            for the black tophat. Optional if ``footprint`` or ``structure`` is
            provided.
        footprint (array of ints): Positions of non-infinite elements of a flat
            structuring element used for the black tophat. Non-zero values
            give the set of neighbors of the center over which opening is
            chosen.
        structure (array of ints): Structuring element used for the black
            tophat. ``structure`` may be a non-flat structuring element.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If None, `input` is filtered along all axes. If an `origin` tuple
            is provided, its length must match the number of axes.

    Returns:
        cupy.ndarry : Result of the filter of ``input`` with ``structure``.

    .. seealso:: :func:`scipy.ndimage.black_tophat`
    """
    if (size is not None) and (footprint is not None):
        warnings.warn(
            'ignoring size because footprint is set', UserWarning, stacklevel=2
        )
    kwargs = dict(mode=mode, cval=cval, origin=origin, axes=axes)
    tmp = grey_dilation(input, size, footprint, structure, None, **kwargs)
    tmp = grey_erosion(tmp, size, footprint, structure, output, **kwargs)
    if input.dtype == numpy.bool_ and tmp.dtype == numpy.bool_:
        cupy.bitwise_xor(tmp, input, out=tmp)
    else:
        cupy.subtract(tmp, input, out=tmp)
    return tmp
