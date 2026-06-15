"""rsnum - A Rust-powered NumPy-compatible array library.

rsnum provides a NumPy-compatible API implemented in Rust for better
performance and memory efficiency while maintaining full API compatibility.

Examples:
    >>> import rsnum as np
    >>> a = np.ndarray([1, 2, 3])
    >>> np.sum(a)
    6
    >>> np.mean(a)
    2.0
"""

import rsnum._core as _core
from rsnum._core import ndarray_iter as NdArrayIter

def _ensure(x):
    """Convert list/tuple to ndarray if needed."""
    if isinstance(x, (list, tuple)):
        return _core.ndarray(x)
    elif isinstance(x, ndarray):
        return x._array
    elif hasattr(x, '__class__') and x.__class__.__name__ == 'ndarray':
        return x
    return x

def _wrap_result(result):
    """Wrap a raw ndarray result in our ndarray class."""
    if hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return ndarray._wrap(result)
    return result

class ndarray:
    """
    rsnum.ndarray - A multi-dimensional array object.

    rsnum.ndarray is a Rust-backed array type that provides NumPy-compatible
    functionality with improved performance and memory efficiency.

    Parameters:
        data (array_like): Initial data for the array. Can be a list, tuple,
            or any iterable.

    Attributes:
        shape: Tuple of array dimensions.
        ndim: Number of dimensions.
        size: Total number of elements.
        dtype: Data type of the array elements.

    Examples:
        >>> import rsnum as np
        >>> a = np.ndarray([1, 2, 3])
        >>> a.shape
        (3,)
        >>> a.ndim
        1
        >>> a.size
        3
    """
    
    def __init__(self, data):
        if isinstance(data, ndarray):
            self._array = data._array
        elif hasattr(data, '__class__') and data.__class__.__name__ == 'ndarray':
            self._array = data
        else:
            self._array = _core.ndarray(data)
    
    def tolist(self):
        """
        Convert the array to a Python list.

        Returns:
            list: A Python list representation of the array.

        Examples:
            >>> a = np.ndarray([1, 2, 3])
            >>> a.tolist()
            [1.0, 2.0, 3.0]
            >>> b = np.ndarray([[1, 2], [3, 4]])
            >>> b.tolist()
            [[1.0, 2.0], [3.0, 4.0]]
        """
        return self._array.tolist()
    
    def reshape(self, *shape):
        """
        Gives a new shape to an array without changing its data.

        Parameters:
            *shape: New shape. Can be a tuple or multiple arguments.

        Returns:
            ndarray: Reshaped array.

        Examples:
            >>> a = np.ndarray([1, 2, 3, 4])
            >>> a.reshape(2, 2)
            rsnum.ndarray([[1., 2.], [3., 4.]])
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            result = self._array.reshape(shape[0])
        else:
            result = self._array.reshape(shape)
        return ndarray._wrap(result)
    
    def flatten(self):
        """
        Return a copy of the array collapsed into one dimension.

        Returns:
            ndarray: Flattened array.

        Examples:
            >>> a = np.ndarray([[1, 2], [3, 4]])
            >>> a.flatten()
            rsnum.ndarray([1., 2., 3., 4.])
        """
        result = self._array.flatten()
        return ndarray._wrap(result)
    
    def squeeze(self):
        """
        Remove axes of length one from an array.

        Returns:
            ndarray: Array with squeezed dimensions.

        Examples:
            >>> a = np.ndarray([[[1, 2, 3]]])
            >>> a.squeeze()
            rsnum.ndarray([1., 2., 3.])
        """
        result = self._array.squeeze()
        return ndarray._wrap(result)
    
    def sort(self, axis=-1):
        """
        Sort an array in place along the specified axis.

        Parameters:
            axis (int, optional): Axis along which to sort. Default is -1.

        Returns:
            None: The array is sorted in place.

        Examples:
            >>> a = np.ndarray([3, 1, 2])
            >>> a.sort()
            >>> a
            rsnum.ndarray([1., 2., 3.])
        """
        self._array.sort(axis)
    
    def argsort(self, axis=-1):
        """
        Returns the indices that would sort an array.

        Parameters:
            axis (int, optional): Axis along which to sort. Default is -1.

        Returns:
            ndarray: Indices that would sort the array.

        Examples:
            >>> a = np.ndarray([3, 1, 2])
            >>> a.argsort()
            rsnum.ndarray([1., 2., 0.])
        """
        result = self._array.argsort(axis)
        return ndarray._wrap(result)
    
    def clip(self, a_min, a_max):
        """
        Clip (limit) the values in an array.

        Parameters:
            a_min: Minimum value.
            a_max: Maximum value.

        Returns:
            ndarray: Clipped array.

        Examples:
            >>> a = np.ndarray([-1, 2, 5, 8])
            >>> a.clip(0, 5)
            rsnum.ndarray([0., 2., 5., 5.])
        """
        result = self._array.clip(a_min, a_max)
        return ndarray._wrap(result)
    
    def copy(self):
        """
        Return a copy of the array.

        Returns:
            ndarray: A copy of the array.

        Examples:
            >>> a = np.ndarray([1, 2, 3])
            >>> b = a.copy()
            >>> b[0] = 10
            >>> a
            rsnum.ndarray([1., 2., 3.])
            >>> b
            rsnum.ndarray([10., 2., 3.])
        """
        result = self._array.copy()
        return ndarray._wrap(result)
    
    def astype(self, dtype):
        """
        Cast the array to a specified type.

        Parameters:
            dtype: Target data type.

        Returns:
            ndarray: Array cast to the specified type.
        """
        result = self._array.astype(dtype)
        return ndarray._wrap(result)
    
    def fill(self, value):
        """
        Fill the array with a scalar value.

        Parameters:
            value: Scalar value to fill the array with.

        Examples:
            >>> a = np.ndarray([1, 2, 3])
            >>> a.fill(0)
            >>> a
            rsnum.ndarray([0., 0., 0.])
        """
        self._array.fill(value)
    
    def repeat(self, repeats, axis=None):
        """
        Repeat elements of an array.

        Parameters:
            repeats: Number of repetitions.
            axis (int, optional): Axis along which to repeat.

        Returns:
            ndarray: Repeated array.
        """
        result = self._array.repeat(repeats, axis)
        return ndarray._wrap(result)
    
    def ravel(self):
        """
        Return a flattened array.

        Returns:
            ndarray: Flattened array.

        Examples:
            >>> a = np.ndarray([[1, 2], [3, 4]])
            >>> a.ravel()
            rsnum.ndarray([1., 2., 3., 4.])
        """
        result = self._array.ravel()
        return ndarray._wrap(result)
    
    def nonzero(self):
        """
        Return the indices of elements that are non-zero.

        Returns:
            ndarray: Indices of non-zero elements.
        """
        result = self._array.nonzero()
        return ndarray._wrap(result)
    
    @property
    def T(self):
        """
        Transpose of the array.

        Returns:
            ndarray: Transposed array.

        Examples:
            >>> a = np.ndarray([[1, 2], [3, 4]])
            >>> a.T
            rsnum.ndarray([[1., 3.], [2., 4.]])
        """
        result = self._array.T
        return ndarray._wrap(result)
    
    @property
    def shape(self):
        """Tuple of array dimensions."""
        return self._array.shape
    
    @property
    def ndim(self):
        """Number of array dimensions."""
        return self._array.ndim
    
    @property
    def size(self):
        """Total number of elements."""
        return self._array.size
    
    @property
    def dtype(self):
        """Data type of the array elements."""
        return self._array.dtype
    
    def sum(self, axis=None):
        """
        Sum of array elements over a given axis.

        Parameters:
            axis (int, optional): Axis or axes along which a sum is performed.

        Returns:
            ndarray or scalar: Sum of array elements.

        Examples:
            >>> a = np.ndarray([1, 2, 3])
            >>> a.sum()
            6.0
            >>> b = np.ndarray([[1, 2], [3, 4]])
            >>> b.sum(axis=0)
            rsnum.ndarray([4., 6.])
        """
        result = self._array.sum(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return ndarray._wrap(result)
    
    def mean(self, axis=None):
        """
        Compute the arithmetic mean along the specified axis.

        Parameters:
            axis (int, optional): Axis or axes along which the mean is computed.

        Returns:
            ndarray or scalar: Arithmetic mean of array elements.

        Examples:
            >>> a = np.ndarray([1, 2, 3, 4, 5])
            >>> a.mean()
            3.0
        """
        result = self._array.mean(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return ndarray._wrap(result)
    
    def std(self, axis=None):
        """
        Compute the standard deviation along the specified axis.

        Parameters:
            axis (int, optional): Axis or axes along which to compute the standard deviation.

        Returns:
            ndarray or scalar: Standard deviation.

        Examples:
            >>> a = np.ndarray([1, 2, 3, 4, 5])
            >>> a.std()
            1.4142135623730951
        """
        result = self._array.std(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return ndarray._wrap(result)
    
    def var(self, axis=None):
        """
        Compute the variance along the specified axis.

        Parameters:
            axis (int, optional): Axis or axes along which to compute the variance.

        Returns:
            ndarray or scalar: Variance.

        Examples:
            >>> a = np.ndarray([1, 2, 3, 4, 5])
            >>> a.var()
            2.0
        """
        result = self._array.var(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return ndarray._wrap(result)
    
    def min(self, axis=None):
        """
        Find the minimum value in an array or along an axis.

        Parameters:
            axis (int, optional): Axis or axes along which to find the minimum.

        Returns:
            ndarray or scalar: Minimum value(s).

        Examples:
            >>> a = np.ndarray([3, 1, 4, 1, 5])
            >>> a.min()
            1.0
        """
        result = self._array.min(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return ndarray._wrap(result)
    
    def max(self, axis=None):
        """
        Find the maximum value in an array or along an axis.

        Parameters:
            axis (int, optional): Axis or axes along which to find the maximum.

        Returns:
            ndarray or scalar: Maximum value(s).

        Examples:
            >>> a = np.ndarray([3, 1, 4, 1, 5])
            >>> a.max()
            5.0
        """
        result = self._array.max(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return ndarray._wrap(result)
    
    def argmin(self, axis=None):
        """
        Returns the indices of the minimum values along an axis.

        Parameters:
            axis (int, optional): Axis or axes along which to find the minimum indices.

        Returns:
            ndarray or scalar: Indices of minimum values.

        Examples:
            >>> a = np.ndarray([3, 1, 4, 1, 5])
            >>> a.argmin()
            1.0
        """
        result = self._array.argmin(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return ndarray._wrap(result)
    
    def argmax(self, axis=None):
        """
        Returns the indices of the maximum values along an axis.

        Parameters:
            axis (int, optional): Axis or axes along which to find the maximum indices.

        Returns:
            ndarray or scalar: Indices of maximum values.

        Examples:
            >>> a = np.ndarray([3, 1, 4, 1, 5])
            >>> a.argmax()
            4.0
        """
        result = self._array.argmax(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return ndarray._wrap(result)
    
    def all(self):
        """
        Test whether all array elements evaluate to True.

        Returns:
            bool: True if all elements are truthy, else False.

        Examples:
            >>> a = np.ndarray([1, 2, 3])
            >>> a.all()
            True
            >>> b = np.ndarray([0, 1, 2])
            >>> b.all()
            False
        """
        return self._array.all()
    
    def any(self):
        """
        Test whether any array element evaluates to True.

        Returns:
            bool: True if any element is truthy, else False.

        Examples:
            >>> a = np.ndarray([0, 0, 1])
            >>> a.any()
            True
            >>> b = np.ndarray([0, 0, 0])
            >>> b.any()
            False
        """
        return self._array.any()
    
    def __getitem__(self, key):
        result = self._array[key]
        if hasattr(result, 'tolist'):
            return ndarray._wrap(result)
        return result
    
    def __setitem__(self, key, value):
        if isinstance(value, ndarray):
            self._array[key] = value._array
        else:
            self._array[key] = value
    
    def __len__(self):
        return len(self._array)
    
    def __iter__(self):
        return iter(self._array)
    
    def __add__(self, other):
        if isinstance(other, ndarray):
            result = self._array + other._array
        else:
            result = self._array + other
        return ndarray._wrap(result)
    
    def __radd__(self, other):
        result = other + self._array
        return ndarray._wrap(result)
    
    def __sub__(self, other):
        if isinstance(other, ndarray):
            result = self._array - other._array
        else:
            result = self._array - other
        return ndarray._wrap(result)
    
    def __rsub__(self, other):
        result = other - self._array
        return ndarray._wrap(result)
    
    def __mul__(self, other):
        if isinstance(other, ndarray):
            result = self._array * other._array
        else:
            result = self._array * other
        return ndarray._wrap(result)
    
    def __rmul__(self, other):
        result = other * self._array
        return ndarray._wrap(result)
    
    def __truediv__(self, other):
        if isinstance(other, ndarray):
            result = self._array / other._array
        else:
            result = self._array / other
        return ndarray._wrap(result)
    
    def __rtruediv__(self, other):
        result = other / self._array
        return ndarray._wrap(result)
    
    def __pow__(self, power):
        result = self._array ** power
        return ndarray._wrap(result)
    
    def __neg__(self):
        result = -self._array
        return ndarray._wrap(result)
    
    def __pos__(self):
        result = +self._array
        return ndarray._wrap(result)
    
    def __abs__(self):
        result = abs(self._array)
        return ndarray._wrap(result)
    
    def __lt__(self, other):
        if isinstance(other, ndarray):
            return self._array < other._array
        return self._array < other
    
    def __le__(self, other):
        if isinstance(other, ndarray):
            return self._array <= other._array
        return self._array <= other
    
    def __gt__(self, other):
        if isinstance(other, ndarray):
            return self._array > other._array
        return self._array > other
    
    def __ge__(self, other):
        if isinstance(other, ndarray):
            return self._array >= other._array
        return self._array >= other
    
    def __eq__(self, other):
        if isinstance(other, ndarray):
            return self._array == other._array
        return self._array == other
    
    def __ne__(self, other):
        if isinstance(other, ndarray):
            return self._array != other._array
        return self._array != other
    
    def __repr__(self):
        return repr(self._array)
    
    def __str__(self):
        return str(self._array)
    
    @staticmethod
    def _wrap(obj):
        """Wrap a raw ndarray object in the ndarray class."""
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'ndarray':
            result = ndarray.__new__(ndarray)
            result._array = obj
            return result
        return obj


def zeros(shape, dtype=None):
    """Return a new array of given shape and type, filled with zeros."""
    return ndarray(_core.zeros(shape))

def ones(shape, dtype=None):
    """Return a new array of given shape and type, filled with ones."""
    return ndarray(_core.ones(shape))

def eye(n, m=None, k=0):
    """Return a 2-D array with ones on the diagonal and zeros elsewhere."""
    return ndarray(_core.eye(n, m, k))

def arange(*args):
    """Return evenly spaced values within a given interval."""
    return ndarray(_core.arange(*args))

def full(shape, fill_value, dtype=None):
    """Return a new array of given shape and type, filled with fill_value."""
    return ndarray(_core.full(shape, fill_value))

def empty(shape, dtype=None):
    """Return a new array of given shape and type, without initializing entries."""
    return ndarray(_core.empty(shape))

def argmin(x, axis=None):
    """Returns the indices of the minimum values along an axis."""
    if axis is None:
        return _core.argmin(_ensure(x))
    return ndarray(_core.argmin(_ensure(x), axis))

def argmax(x, axis=None):
    """Returns the indices of the maximum values along an axis."""
    if axis is None:
        return _core.argmax(_ensure(x))
    return ndarray(_core.argmax(_ensure(x), axis))

def concatenate(arrays, axis=0):
    """Join a sequence of arrays along an existing axis."""
    arrays = [_ensure(a) for a in arrays]
    return ndarray(_core.concatenate(arrays, axis))

def stack(arrays, axis=0):
    """Join a sequence of arrays along a new axis."""
    arrays = [_ensure(a) for a in arrays]
    return ndarray(_core.stack(arrays, axis))

def vstack(tup):
    """Stack arrays in sequence vertically (row wise)."""
    arrays = [_ensure(a) for a in tup]
    return ndarray(_core.vstack(arrays))

def hstack(tup):
    """Stack arrays in sequence horizontally (column wise)."""
    arrays = [_ensure(a) for a in tup]
    return ndarray(_core.hstack(arrays))

def nonzero(x):
    """Return the indices of elements that are non-zero."""
    return _wrap_result(_core.nonzero(_ensure(x)))

def argwhere(x):
    """Find the indices of array elements that are non-zero."""
    return ndarray(_core.argwhere(_ensure(x)))

def count_nonzero(x):
    """Count the number of non-zero values in the array."""
    return _core.count_nonzero(_ensure(x))

def where(condition, x=None, y=None):
    """Return elements chosen from x or y depending on condition."""
    if x is not None:
        x = _ensure(x)
    if y is not None:
        y = _ensure(y)
    return ndarray(_core.where(_ensure(condition), x, y))

def meshgrid(*xi, **kwargs):
    """Return coordinate matrices from coordinate vectors."""
    arrays = [_ensure(x) for x in xi]
    result = _core.meshgrid(*arrays, **kwargs)
    if isinstance(result, tuple):
        return tuple(ndarray._wrap(r) for r in result)
    return ndarray._wrap(result)

def histogram(x, bins=10, range=None, density=False):
    """Compute the histogram of a set of data."""
    result = _core.histogram(_ensure(x), bins, range, density)
    if isinstance(result, tuple):
        return tuple(ndarray._wrap(r) if hasattr(r, '__class__') and r.__class__.__name__ == 'ndarray' else r for r in result)
    return ndarray._wrap(result)

def fft(a, n=None, axis=-1):
    """
    Compute the one-dimensional discrete Fourier Transform.

    Parameters:
        a (array_like): Input array.
        n (int, optional): Number of points to use.
        axis (int, optional): Axis along which to compute the FFT.

    Returns:
        list: FFT result as list of (real, imaginary) tuples.
    """
    if isinstance(a, ndarray):
        a = a.tolist()
    elif isinstance(a, (list, tuple)):
        pass
    else:
        a = [float(a)]
    return _core.py_fft(a)

def ifft(a, n=None, axis=-1):
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    Parameters:
        a (array_like): Input array of complex numbers.
        n (int, optional): Number of points to use.
        axis (int, optional): Axis along which to compute the IFFT.

    Returns:
        list: IFFT result as list of (real, imaginary) tuples.
    """
    return _core.py_ifft(a)

def rfft(a, n=None, axis=-1):
    """
    Compute the one-dimensional discrete Fourier Transform for real input.

    Parameters:
        a (array_like): Input array.
        n (int, optional): Number of points to use.
        axis (int, optional): Axis along which to compute the RFFT.

    Returns:
        list: RFFT result as list of (real, imaginary) tuples.
    """
    if isinstance(a, ndarray):
        a = a.tolist()
    elif isinstance(a, (list, tuple)):
        pass
    else:
        a = [float(a)]
    return _core.py_rfft(a)

def irfft(a, n=None, axis=-1):
    """
    Compute the inverse of rfft.

    Parameters:
        a (array_like): Input array of complex numbers.
        n (int, optional): Number of points.
        axis (int, optional): Axis along which to compute the IRFFT.

    Returns:
        list: IRFFT result as list of floats.
    """
    return _core.py_irfft(a, n)

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """
    Return evenly spaced numbers over a specified interval.

    Parameters:
        start (array_like): Start of interval.
        stop (array_like): End of interval.
        num (int, optional): Number of samples to generate. Default is 50.
        endpoint (bool, optional): If True, stop is included. Default is True.
        retstep (bool, optional): If True, return (samples, step). Default is False.
        dtype: Data type of the output array.
        axis (int, optional): Axis along which to generate the samples.

    Returns:
        ndarray: Array of evenly spaced numbers.

    Examples:
        >>> np.linspace(0, 1, 5)
        rsnum.ndarray([0., 0.25, 0.5, 0.75, 1.])
    """
    start = _scalar(_ensure(start))
    stop = _scalar(_ensure(stop))
    return ndarray(_core.linspace(start, stop, num))

def sum(x, axis=None):
    """
    Sum of array elements over a given axis.

    Parameters:
        x (array_like): Input array.
        axis (int, optional): Axis or axes along which a sum is performed.

    Returns:
        ndarray or scalar: Sum of array elements.

    Examples:
        >>> np.sum([1, 2, 3, 4, 5])
        15
        >>> np.sum([[1, 2], [3, 4]], axis=0)
        rsnum.ndarray([4., 6.])
    """
    result = _core.sum(_ensure(x), axis)
    if axis is None and hasattr(result, 'tolist'):
        return result.tolist()
    return ndarray._wrap(result)

def mean(x, axis=None):
    """
    Compute the arithmetic mean along the specified axis.

    Parameters:
        x (array_like): Input array.
        axis (int, optional): Axis or axes along which the mean is computed.

    Returns:
        ndarray or scalar: Arithmetic mean of array elements.

    Examples:
        >>> np.mean([1, 2, 3, 4, 5])
        3.0
        >>> np.mean([[1, 2], [3, 4]], axis=0)
        rsnum.ndarray([2., 3.])
    """
    result = _core.mean(_ensure(x), axis)
    if axis is None and hasattr(result, 'tolist'):
        return result.tolist()
    return ndarray._wrap(result)

def min(x, axis=None):
    """
    Find the minimum value in an array or along an axis.

    Parameters:
        x (array_like): Input array.
        axis (int, optional): Axis or axes along which to find the minimum.

    Returns:
        ndarray or scalar: Minimum value(s).

    Examples:
        >>> np.min([3, 1, 4, 1, 5])
        1.0
        >>> np.min([[1, 2], [3, 0]], axis=0)
        rsnum.ndarray([1., 0.])
    """
    result = _core.min(_ensure(x), axis)
    if axis is None and hasattr(result, 'tolist'):
        return result.tolist()
    return ndarray._wrap(result)

def max(x, axis=None):
    """
    Find the maximum value in an array or along an axis.

    Parameters:
        x (array_like): Input array.
        axis (int, optional): Axis or axes along which to find the maximum.

    Returns:
        ndarray or scalar: Maximum value(s).

    Examples:
        >>> np.max([3, 1, 4, 1, 5])
        5.0
        >>> np.max([[1, 2], [3, 0]], axis=0)
        rsnum.ndarray([3., 2.])
    """
    result = _core.max(_ensure(x), axis)
    if axis is None and hasattr(result, 'tolist'):
        return result.tolist()
    return ndarray._wrap(result)

def std(x, axis=None):
    """
    Compute the standard deviation along the specified axis.

    Parameters:
        x (array_like): Input array.
        axis (int, optional): Axis or axes along which to compute the standard deviation.

    Returns:
        ndarray or scalar: Standard deviation.

    Examples:
        >>> np.std([1, 2, 3, 4, 5])
        1.4142135623730951
    """
    result = _core.std(_ensure(x), axis)
    if axis is None and hasattr(result, 'tolist'):
        return result.tolist()
    return ndarray._wrap(result)

def var(x, axis=None):
    """
    Compute the variance along the specified axis.

    Parameters:
        x (array_like): Input array.
        axis (int, optional): Axis or axes along which to compute the variance.

    Returns:
        ndarray or scalar: Variance.

    Examples:
        >>> np.var([1, 2, 3, 4, 5])
        2.0
    """
    result = _core.var(_ensure(x), axis)
    if axis is None and hasattr(result, 'tolist'):
        return result.tolist()
    return ndarray._wrap(result)

def _scalar(x):
    """Extract scalar value from ndarray if it's a single-element array."""
    if isinstance(x, ndarray):
        return x[0]
    elif hasattr(x, '__class__') and x.__class__.__name__ == 'ndarray':
        return x[0]
    return x

def sin(x):
    """
    Compute the sine of each element in the input array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: Sine of each element.

    Examples:
        >>> np.sin([0, np.pi/2, np.pi])
        rsnum.ndarray([0., 1., 0.])
    """
    return ndarray(_core.sin(_ensure(x)))

def cos(x):
    """
    Compute the cosine of each element in the input array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: Cosine of each element.

    Examples:
        >>> np.cos([0, np.pi/2, np.pi])
        rsnum.ndarray([1., 0., -1.])
    """
    return ndarray(_core.cos(_ensure(x)))

def tan(x):
    """
    Compute the tangent of each element in the input array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: Tangent of each element.

    Examples:
        >>> np.tan([0, np.pi/4])
        rsnum.ndarray([0., 1.])
    """
    return ndarray(_core.tan(_ensure(x)))

def sqrt(x):
    """
    Compute the square root of each element in the input array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: Square root of each element.

    Examples:
        >>> np.sqrt([1, 4, 9])
        rsnum.ndarray([1., 2., 3.])
    """
    return ndarray(_core.sqrt(_ensure(x)))

def exp(x):
    """
    Compute the exponential of each element in the input array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: Exponential of each element.

    Examples:
        >>> np.exp([0, 1, 2])
        rsnum.ndarray([1., 2.718281828459045, 7.38905609893065])
    """
    return ndarray(_core.exp(_ensure(x)))

def log(x):
    """
    Compute the natural logarithm of each element in the input array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: Natural logarithm of each element.

    Examples:
        >>> np.log([1, np.e, np.e**2])
        rsnum.ndarray([0., 1., 2.])
    """
    return ndarray(_core.log(_ensure(x)))

def log10(x):
    """
    Compute the base 10 logarithm of each element in the input array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: Base 10 logarithm of each element.

    Examples:
        >>> np.log10([1, 10, 100])
        rsnum.ndarray([0., 1., 2.])
    """
    return ndarray(_core.log10(_ensure(x)))

def log2(x):
    """
    Compute the base 2 logarithm of each element in the input array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: Base 2 logarithm of each element.

    Examples:
        >>> np.log2([1, 2, 4])
        rsnum.ndarray([0., 1., 2.])
    """
    return ndarray(_core.log2(_ensure(x)))

def log1p(x):
    """
    Compute log(1 + x) for each element in the input array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: log(1 + x) for each element.

    Examples:
        >>> np.log1p([0, 1, np.e-1])
        rsnum.ndarray([0., 0.6931471805599453, 1.])
    """
    return ndarray(_core.log1p(_ensure(x)))

def abs(x):
    """
    Compute the absolute value of each element in the input array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: Absolute value of each element.

    Examples:
        >>> np.abs([-1, -2, 3])
        rsnum.ndarray([1., 2., 3.])
    """
    return ndarray(_core.abs(_ensure(x)))

def cosh(x):
    """Compute the hyperbolic cosine of each element."""
    return ndarray(_core.cosh(_ensure(x)))

def sinh(x):
    """Compute the hyperbolic sine of each element."""
    return ndarray(_core.sinh(_ensure(x)))

def tanh(x):
    """Compute the hyperbolic tangent of each element."""
    return ndarray(_core.tanh(_ensure(x)))

def acosh(x):
    """Compute the inverse hyperbolic cosine of each element."""
    return ndarray(_core.acosh(_ensure(x)))

def asinh(x):
    """Compute the inverse hyperbolic sine of each element."""
    return ndarray(_core.asinh(_ensure(x)))

def atanh(x):
    """Compute the inverse hyperbolic tangent of each element."""
    return ndarray(_core.atanh(_ensure(x)))

def asin(x):
    """Compute the inverse sine of each element."""
    return ndarray(_core.asin(_ensure(x)))

def acos(x):
    """Compute the inverse cosine of each element."""
    return ndarray(_core.acos(_ensure(x)))

def atan(x):
    """Compute the inverse tangent of each element."""
    return ndarray(_core.atan(_ensure(x)))

def expm1(x):
    """Compute exp(x) - 1 for each element."""
    return ndarray(_core.expm1(_ensure(x)))

def floor(x):
    """Return the floor of each element."""
    return ndarray(_core.floor(_ensure(x)))

def ceil(x):
    """Return the ceiling of each element."""
    return ndarray(_core.ceil(_ensure(x)))

def round(x):
    """Round each element to the nearest integer."""
    return ndarray(_core.round(_ensure(x)))

def isnan(x):
    """Test element-wise for NaN and return result as a boolean array."""
    return _core.isnan(_ensure(x))

def isinf(x):
    """Test element-wise for infinity and return result as a boolean array."""
    return _core.isinf(_ensure(x))

def isfinite(x):
    """Test element-wise for finite values and return result as a boolean array."""
    return _core.isfinite(_ensure(x))

def all(x):
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters:
        x (array_like): Input array.

    Returns:
        bool: True if all elements are truthy, else False.

    Examples:
        >>> np.all([True, True, True])
        True
        >>> np.all([True, False, True])
        False
    """
    return _core.all(_ensure(x))

def any(x):
    """
    Test whether any array element along a given axis evaluates to True.

    Parameters:
        x (array_like): Input array.

    Returns:
        bool: True if any element is truthy, else False.

    Examples:
        >>> np.any([False, False, True])
        True
        >>> np.any([False, False, False])
        False
    """
    return _core.any(_ensure(x))

def sort(x, axis=-1):
    """Sort an array along the specified axis."""
    result = _core.sort(_ensure(x), axis)
    return ndarray._wrap(result)

def argsort(x, axis=-1):
    """Returns the indices that would sort an array."""
    result = _core.argsort(_ensure(x), axis)
    return ndarray._wrap(result)

def median(x):
    """
    Compute the median of the array.

    Parameters:
        x (array_like): Input array.

    Returns:
        scalar: Median value.

    Examples:
        >>> np.median([1, 2, 3, 4, 5])
        3.0
    """
    result = _core.median(_ensure(x))
    if hasattr(result, 'tolist'):
        return result.tolist()
    return result

def percentile(x, q):
    """
    Compute the q-th percentile of the array.

    Parameters:
        x (array_like): Input array.
        q (float): Percentile to compute (0-100).

    Returns:
        scalar: Percentile value.
    """
    result = _core.percentile(_ensure(x), q)
    if hasattr(result, 'tolist'):
        return result.tolist()
    return result

def clip(x, a_min, a_max):
    """
    Clip (limit) the values in an array.

    Parameters:
        x (array_like): Input array.
        a_min: Minimum value.
        a_max: Maximum value.

    Returns:
        ndarray: Clipped array.
    """
    return ndarray(_core.clip(_ensure(x), a_min, a_max))

def unique(x):
    """
    Find the unique elements of an array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: Sorted unique elements.
    """
    return ndarray(_core.unique(_ensure(x)))

def transpose(x):
    """
    Permute the dimensions of an array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: Transposed array.
    """
    return ndarray(_core.transpose(_ensure(x)))

def gradient(x):
    """
    Return the gradient of an array.

    Parameters:
        x (array_like): Input array.

    Returns:
        ndarray: Gradient array.
    """
    return ndarray(_core.gradient(_ensure(x)))

def diff(x, n=1, axis=-1):
    """
    Calculate the n-th discrete difference along the given axis.

    Parameters:
        x (array_like): Input array.
        n (int, optional): Number of times to difference. Default is 1.
        axis (int, optional): Axis along which to compute the difference. Default is -1.

    Returns:
        ndarray: n-th differences.
    """
    return ndarray(_core.diff(_ensure(x), n, axis))

def trapz(y, dx=1.0):
    """
    Integrate along the given axis using the composite trapezoidal rule.

    Parameters:
        y (array_like): Input array to integrate.
        dx (float, optional): Spacing between samples. Default is 1.0.

    Returns:
        scalar or ndarray: Integral of y.
    """
    result = _core.trapz(_ensure(y), dx)
    if hasattr(result, 'tolist'):
        return result.tolist()
    return result

def cross(a, b):
    """
    Compute the cross product of two arrays.

    Parameters:
        a (array_like): First input array.
        b (array_like): Second input array.

    Returns:
        ndarray: Cross product.
    """
    return ndarray(_core.cross(_ensure(a), _ensure(b)))

def allclose(a, b, rtol=1e-5, atol=1e-8):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    Parameters:
        a (array_like): First input array.
        b (array_like): Second input array.
        rtol (float, optional): Relative tolerance. Default is 1e-5.
        atol (float, optional): Absolute tolerance. Default is 1e-8.

    Returns:
        bool: True if arrays are close.
    """
    return _core.allclose(_ensure(a), _ensure(b), rtol, atol)

def maximum(x1, x2):
    """
    Element-wise maximum of two arrays.

    Parameters:
        x1 (array_like): First input array.
        x2 (array_like): Second input array.

    Returns:
        ndarray: Element-wise maximum.
    """
    return ndarray(_core.maximum(_ensure(x1), _ensure(x2)))

def minimum(x1, x2):
    """
    Element-wise minimum of two arrays.

    Parameters:
        x1 (array_like): First input array.
        x2 (array_like): Second input array.

    Returns:
        ndarray: Element-wise minimum.
    """
    return ndarray(_core.minimum(_ensure(x1), _ensure(x2)))

def tile(a, reps):
    """
    Construct an array by repeating a the number of times given by reps.

    Parameters:
        a (array_like): Input array.
        reps: The number of repetitions of a along each axis.

    Returns:
        ndarray: tiled array.
    """
    return ndarray(_core.tile(_ensure(a), reps))

def squeeze(a):
    """
    Remove axes of length one from an array.

    Parameters:
        a (array_like): Input array.

    Returns:
        ndarray: Array with squeezed dimensions.
    """
    return ndarray(_core.squeeze(_ensure(a)))

def flatten(a):
    """
    Return a copy of the array collapsed into one dimension.

    Parameters:
        a (array_like): Input array.

    Returns:
        ndarray: Flattened array.
    """
    return ndarray(_core.flatten(_ensure(a)))

def reshape(a, shape):
    """
    Gives a new shape to an array without changing its data.

    Parameters:
        a (array_like): Input array.
        shape: New shape.

    Returns:
        ndarray: Reshaped array.
    """
    return ndarray(_core.reshape(_ensure(a), shape))

class linalg_module:
    """Linear algebra module.
    
    Provides linear algebra functions like dot product, matrix inversion,
    determinant, etc.
    """
    
    def dot(self, a, b):
        """
        Dot product of two arrays.

        Parameters:
            a (array_like): First input array.
            b (array_like): Second input array.

        Returns:
            ndarray: Dot product.
        """
        return ndarray(_core.linalg.dot(_ensure(a), _ensure(b)))
    
    def matmul(self, a, b):
        """
        Matrix product of two arrays.

        Parameters:
            a (array_like): First input array.
            b (array_like): Second input array.

        Returns:
            ndarray: Matrix product.
        """
        return ndarray(_core.linalg.matmul(_ensure(a), _ensure(b)))
    
    def inv(self, a):
        """
        Compute the inverse of a matrix.

        Parameters:
            a (array_like): Input matrix.

        Returns:
            ndarray: Inverse matrix.
        """
        return ndarray(_core.linalg.inv(_ensure(a)))
    
    def det(self, a):
        """
        Compute the determinant of a matrix.

        Parameters:
            a (array_like): Input matrix.

        Returns:
            scalar: Determinant.
        """
        return _core.linalg.det(_ensure(a))
    
    def norm(self, x, ord=None, axis=None):
        """
        Compute the norm of an array.

        Parameters:
            x (array_like): Input array.
            ord: Order of the norm.
            axis: Axis or axes along which to compute the norm.

        Returns:
            scalar or ndarray: Norm.
        """
        result = _core.linalg.norm(_ensure(x), ord, axis)
        if hasattr(result, 'tolist'):
            return result.tolist()
        return result
    
    def solve(self, a, b):
        """
        Solve a linear matrix equation.

        Parameters:
            a (array_like): Coefficient matrix.
            b (array_like): Right-hand side.

        Returns:
            ndarray: Solution.
        """
        return ndarray(_core.linalg.solve(_ensure(a), _ensure(b)))
    
    def eig(self, a):
        """
        Compute the eigenvalues and right eigenvectors of a square array.

        Parameters:
            a (array_like): Input array.

        Returns:
            tuple: (eigenvalues, eigenvectors)
        """
        result = _core.linalg.eig(_ensure(a))
        if isinstance(result, tuple):
            return tuple(ndarray._wrap(r) if hasattr(r, '__class__') and r.__class__.__name__ == 'ndarray' else r for r in result)
        return ndarray._wrap(result)
    
    def eigvals(self, a):
        """
        Compute the eigenvalues of a square array.

        Parameters:
            a (array_like): Input array.

        Returns:
            ndarray: Eigenvalues.
        """
        return ndarray(_core.linalg.eigvals(_ensure(a)))
    
    def svd(self, a):
        """
        Singular Value Decomposition.

        Parameters:
            a (array_like): Input array.

        Returns:
            tuple: (U, S, V)
        """
        result = _core.linalg.svd(_ensure(a))
        if isinstance(result, tuple):
            return tuple(ndarray._wrap(r) if hasattr(r, '__class__') and r.__class__.__name__ == 'ndarray' else r for r in result)
        return ndarray._wrap(result)
    
    def qr(self, a):
        """
        Compute the QR decomposition of a matrix.

        Parameters:
            a (array_like): Input matrix.

        Returns:
            tuple: (Q, R)
        """
        result = _core.linalg.qr(_ensure(a))
        if isinstance(result, tuple):
            return tuple(ndarray._wrap(r) if hasattr(r, '__class__') and r.__class__.__name__ == 'ndarray' else r for r in result)
        return ndarray._wrap(result)
    
    def cholesky(self, a):
        """
        Compute the Cholesky decomposition of a matrix.

        Parameters:
            a (array_like): Input matrix.

        Returns:
            ndarray: Cholesky decomposition.
        """
        return ndarray(_core.linalg.cholesky(_ensure(a)))
    
    def matrix_power(self, a, n):
        """
        Raise a square matrix to the (integer) power n.

        Parameters:
            a (array_like): Input matrix.
            n (int): Power.

        Returns:
            ndarray: Matrix raised to power n.
        """
        return ndarray(_core.linalg.matrix_power(_ensure(a), n))
    
    def pinv(self, a):
        """
        Compute the Moore-Penrose pseudo-inverse of a matrix.

        Parameters:
            a (array_like): Input matrix.

        Returns:
            ndarray: Pseudo-inverse.
        """
        return ndarray(_core.linalg.pinv(_ensure(a)))
    
    def trace(self, a):
        """
        Compute the trace of a matrix.

        Parameters:
            a (array_like): Input matrix.

        Returns:
            scalar: Trace.
        """
        return _core.linalg.trace(_ensure(a))
    
    def diagonal(self, a):
        """
        Return specified diagonals.

        Parameters:
            a (array_like): Input array.

        Returns:
            ndarray: Diagonals.
        """
        return ndarray(_core.linalg.diagonal(_ensure(a)))
    
    def svdvals(self, a):
        """
        Compute the singular values of a matrix.

        Parameters:
            a (array_like): Input matrix.

        Returns:
            ndarray: Singular values.
        """
        return ndarray(_core.linalg.svdvals(_ensure(a)))
    
    def eigh(self, a):
        """
        Compute the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.

        Parameters:
            a (array_like): Input matrix.

        Returns:
            tuple: (eigenvalues, eigenvectors)
        """
        result = _core.linalg.eigh(_ensure(a))
        if isinstance(result, tuple):
            return tuple(ndarray._wrap(r) if hasattr(r, '__class__') and r.__class__.__name__ == 'ndarray' else r for r in result)
        return ndarray._wrap(result)
    
    def solve_banded(self, lower, upper, ab, b):
        """
        Solve a banded linear system.

        Parameters:
            lower (int): Number of lower diagonals.
            upper (int): Number of upper diagonals.
            ab (array_like): Banded matrix.
            b (array_like): Right-hand side.

        Returns:
            ndarray: Solution.
        """
        return ndarray(_core.linalg.solve_banded(lower, upper, _ensure(ab), _ensure(b)))

class random_module:
    """Random number generation module.
    
    Provides functions for generating random numbers.
    """
    
    @staticmethod
    def seed(val):
        """
        Seed the random number generator.

        Parameters:
            val (int): Seed value.
        """
        _core.random.seed(val)
    
    @staticmethod
    def rand(*args):
        """
        Random values in a given shape.

        Parameters:
            *args: Shape of the output array.

        Returns:
            ndarray or scalar: Random values.

        Examples:
            >>> np.random.rand()
            0.6394267984578837
            >>> np.random.rand(2, 3)
            rsnum.ndarray([[0.02501075, 0.27502932, 0.22321063], [0.73647121, 0.67669948, 0.89218692]])
        """
        result = _core.random.rand(*args)
        if len(args) == 0:
            return result.tolist() if hasattr(result, 'tolist') else result
        return ndarray._wrap(result)
    
    @staticmethod
    def randn(*args):
        """
        Return a sample (or samples) from the "standard normal" distribution.

        Parameters:
            *args: Shape of the output array.

        Returns:
            ndarray or scalar: Random values from standard normal distribution.
        """
        result = _core.random.randn(*args)
        if len(args) == 0:
            return result.tolist() if hasattr(result, 'tolist') else result
        return ndarray._wrap(result)
    
    @staticmethod
    def randint(low, high=None, size=None):
        """
        Return random integers from low (inclusive) to high (exclusive).

        Parameters:
            low (int): Lowest (signed) integers to be drawn from the distribution.
            high (int, optional): If provided, one above the largest (signed) integer.
            size: Output shape.

        Returns:
            ndarray or scalar: Random integers.
        """
        result = _core.random.randint(low, high, size)
        return ndarray._wrap(result)
    
    @staticmethod
    def uniform(low=0.0, high=1.0, size=None):
        """
        Draw samples from a uniform distribution.

        Parameters:
            low (float, optional): Lower boundary of the output interval. Default is 0.0.
            high (float, optional): Upper boundary of the output interval. Default is 1.0.
            size: Output shape.

        Returns:
            ndarray or scalar: Random values.
        """
        result = _core.random.uniform(low, high, size)
        return ndarray._wrap(result)


linalg = linalg_module()
random = random_module()

__all__ = [
    'ndarray', 'NdArrayIter', 'zeros', 'ones', 'eye', 'arange', 'full', 'empty',
    'linspace', 'sum', 'mean', 'min', 'max', 'std', 'var', 'sin', 'cos', 'tan',
    'sqrt', 'exp', 'log', 'log10', 'log2', 'log1p', 'abs', 'cosh', 'sinh', 'tanh',
    'acosh', 'asinh', 'atanh', 'asin', 'acos', 'atan', 'expm1', 'floor', 'ceil',
    'round', 'isnan', 'isinf', 'isfinite', 'all', 'any', 'sort', 'argsort',
    'median', 'percentile', 'clip', 'unique', 'transpose', 'gradient', 'diff',
    'trapz', 'cross', 'allclose', 'maximum', 'minimum', 'tile', 'squeeze',
    'flatten', 'reshape', 'argmin', 'argmax', 'concatenate', 'stack', 'vstack',
    'hstack', 'nonzero', 'argwhere', 'count_nonzero', 'where', 'meshgrid',
    'histogram', 'fft', 'ifft', 'rfft', 'irfft', 'linalg', 'random'
]
