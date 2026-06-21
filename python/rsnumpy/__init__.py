"""rsnumpy - A Rust-powered NumPy-compatible array library.

rsnumpy provides a NumPy-compatible API implemented in Rust for better
performance and memory efficiency while maintaining full API compatibility.

Examples:
    >>> import rsnumpy as np
    >>> a = np.ndarray([1, 2, 3])
    >>> np.sum(a)
    6
    >>> np.mean(a)
    2.0
"""

import rsnumpy._core as _core
from rsnumpy._core import ndarray_iter as NdArrayIter
# ========== 子模块导入和函数挂载 ==========
# 这些导入放在 ndarray 类定义之后以避免循环导入
from . import array_methods
from . import math_functions as _math_functions_module
from . import statistics as _statistics_module
from . import array_ops as _array_ops_module

from .io import save, load, loadtxt, savetxt, savez, load_npz
from .polynomial import Poly, polyval, polyfit, polyder, polyint, polyroots
from .linalg import linalg_module as _linalg_module
from .random import random_module as _random_module


class ndarray:
    """
    rsnumpy.ndarray - 多维数组对象。

    这是一个 Rust 后端支持的数组类型，提供与 NumPy 兼容的功能。

    参数:
        data (array_like): 数组的初始数据，可以是列表、元组或可迭代对象。

    属性:
        shape: 数组维度的元组。
        ndim: 维度数量。
        size: 元素总数。
        dtype: 元素数据类型。
        itemsize: 每个元素的字节大小。
        nbytes: 总字节数。
        T: 数组的转置。
        real: 数组的实部。
        imag: 数组的虚部。
    """

    def __init__(self, data, _dtype="float64", _fields=None):
        if isinstance(data, ndarray):
            self._array = data._array
            self._dtype = data._dtype
            self._fields = getattr(data, '_fields', None)
        elif hasattr(data, '__class__') and data.__class__.__name__ == 'ndarray':
            self._array = data
            self._dtype = _dtype
            self._fields = _fields
        else:
            self._array = _core.ndarray(data)
            self._dtype = _dtype
            self._fields = _fields

    @staticmethod
    def _wrap(raw_array, _dtype="float64", _fields=None):
        """包装原始 Rust ndarray 到 Python 类。"""
        obj = ndarray.__new__(ndarray)
        obj._array = raw_array
        obj._dtype = _dtype
        obj._fields = _fields
        return obj

    def __repr__(self):
        fields = getattr(self, '_fields', None)
        if fields:
            return _format_structured_repr(self)
        if getattr(self, '_dtype', "float64") == "complex128":
            return _core._format_complex_repr(self._array)
        return repr(self._array)

    def __str__(self):
        fields = getattr(self, '_fields', None)
        if fields:
            return _format_structured_str(self)
        if getattr(self, '_dtype', "float64") == "complex128":
            return _core._format_complex_str(self._array)
        return str(self._array)

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return len(self._array)

    def __bool__(self):
        if self.ndim == 0:
            return bool(self.tolist())
        raise ValueError("The truth value of an array with more than one element is ambiguous.")

    def __getitem__(self, key):
        if isinstance(key, tuple):
            # 拆分为整数索引和切片，避免把整数当作 (i, i+1, 1) 的范围
            int_indices = []
            slice_ranges = []
            int_positions = []
            slice_positions = []
            for i, k in enumerate(key):
                if isinstance(k, slice):
                    start = k.start or 0
                    end = k.stop if k.stop is not None else self.shape[i]
                    step = k.step or 1
                    slice_ranges.append((start, end, step))
                    slice_positions.append(i)
                else:
                    int_indices.append(int(k))
                    int_positions.append(i)
            if not slice_ranges:
                # 全部是整数索引，递归走 Rust 的 __getitem__，最后变成标量
                cur = self
                for idx in int_indices:
                    cur = _wrap_result(cur._array[idx], self._dtype)
                return cur
            # 既有整数又有切片：先按整数逐维减维，再按切片
            cur = self
            for idx in reversed(int_indices):
                cur = _wrap_result(cur._array[idx], self._dtype)
            if not isinstance(cur, ndarray):
                return cur
            # cur 的 ndim 现在等于切片数量，把切片按原 key 中的轴位置重新组装
            new_key = []
            for i in range(len(key)):
                if i in int_positions:
                    new_key.append(slice(0, cur.shape[0]))  # 占位，不再使用
                else:
                    new_key.append(key[i])
            # 直接调用 _core.tuple_getitem 但只针对剩余维度的切片
            return _wrap_result(_core.tuple_getitem(cur._array, slice_ranges), self._dtype)
        elif isinstance(key, ndarray):
            # 布尔 / 花式索引：把包装的 ndarray 转换为底层 _array
            return _wrap_result(self._array[key._array], self._dtype)
        elif isinstance(key, (list, tuple)) and any(isinstance(x, bool) for x in key):
            # 布尔列表索引
            return _wrap_result(self._array[key], self._dtype)
        else:
            result = self._array[key]
            return _wrap_result(result, self._dtype)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            import builtins
            val = ndarray(value)
            flat_val = val.ravel().tolist()
            strides = [1]
            for s in reversed(self.shape[1:]):
                strides.insert(0, strides[0] * s)
            flat_key = []
            indices = []
            for i, k in enumerate(key):
                if isinstance(k, slice):
                    indices.append(range(k.start or 0, k.stop or self.shape[i], k.step or 1))
                else:
                    indices.append([k])

            def assign(idx, dim):
                if dim == len(key):
                    flat_idx = builtins.sum(i * s for i, s in zip(idx, strides))
                    flat_key.append(flat_idx)
                else:
                    for i in indices[dim]:
                        assign(idx + [i], dim + 1)
            assign([], 0)
            flat_data = self.ravel().tolist()
            for i, idx in enumerate(flat_key):
                flat_data[idx] = flat_val[i % len(flat_val)]
            new_arr = ndarray(flat_data).reshape(self.shape)
            self._array = new_arr._array
        else:
            self._array[key] = value

    def __add__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array + other._array, self._dtype)
        return _wrap_result(self._array + other, self._dtype)

    def __radd__(self, other):
        return _wrap_result(other + self._array, self._dtype)

    def __sub__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array - other._array, self._dtype)
        return _wrap_result(self._array - other, self._dtype)

    def __rsub__(self, other):
        return _wrap_result(other - self._array, self._dtype)

    def __mul__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array * other._array, self._dtype)
        return _wrap_result(self._array * other, self._dtype)

    def __rmul__(self, other):
        return _wrap_result(other * self._array, self._dtype)

    def __truediv__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array / other._array, self._dtype)
        return _wrap_result(self._array / other, self._dtype)

    def __rtruediv__(self, other):
        return _wrap_result(other / self._array, self._dtype)

    def __matmul__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(_core.linalg.matmul(self._array, other._array), self._dtype)
        return _wrap_result(_core.linalg.matmul(self._array, other), self._dtype)

    def __pow__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(_core.power(self._array, other._array), self._dtype)
        return _wrap_result(_core.power(self._array, _core.ndarray([other])), self._dtype)

    def __eq__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__eq__(other._array), self._dtype)
        return _wrap_result(self._array.__eq__(other), self._dtype)

    def __ne__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__ne__(other._array), self._dtype)
        return _wrap_result(self._array.__ne__(other), self._dtype)

    def __lt__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__lt__(other._array), self._dtype)
        return _wrap_result(self._array.__lt__(other), self._dtype)

    def __le__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__le__(other._array), self._dtype)
        return _wrap_result(self._array.__le__(other), self._dtype)

    def __gt__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__gt__(other._array), self._dtype)
        return _wrap_result(self._array.__gt__(other), self._dtype)

    def __ge__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__ge__(other._array), self._dtype)
        return _wrap_result(self._array.__ge__(other), self._dtype)

    def __round__(self, ndigits=None):
        if ndigits is None:
            return _wrap_result(_core.round(self._array, 0), self._dtype)
        return _wrap_result(_core.round(self._array, ndigits), self._dtype)

    # ========== 属性 ==========

    @property
    def shape(self):
        """返回数组维度的元组。"""
        return self._array.shape

    @property
    def ndim(self):
        """返回数组维度数量。"""
        return self._array.ndim

    @property
    def size(self):
        """返回元素总数。"""
        return self._array.size

    @property
    def dtype(self):
        """返回元素数据类型。"""
        return getattr(self, '_dtype', "float64")

    @property
    def itemsize(self):
        """每个元素的字节大小。"""
        return 8  # f64

    @property
    def nbytes(self):
        """总字节数。"""
        return self.size * self.itemsize

    @property
    def T(self):
        """数组的转置。"""
        return _ndarray_methods().transpose(self)

    @property
    def real(self):
        """数组的实部。"""
        return self.copy()

    @property
    def imag(self):
        """数组的虚部（全零）。"""
        return ndarray([0.0] * self.size).reshape(self.shape)

    # ========== 对象方法 ==========

    def astype(self, dtype):
        """转换数组为指定类型。"""
        return _ndarray_methods().astype(self, dtype)

    def reshape(self, *shape):
        """改变数组形状。"""
        return _ndarray_methods().reshape(self, *shape)

    def resize(self, new_shape):
        """改变数组大小。"""
        _ndarray_methods().resize(self, new_shape)

    def ravel(self, order='C'):
        """展平数组。"""
        return _ndarray_methods().ravel(self, order)

    def flatten(self, order='C'):
        """返回数组的一维副本。"""
        return _ndarray_methods().flatten(self, order)

    def copy(self):
        """返回数组的副本。"""
        return _wrap_result(self._array.copy(), self._dtype)

    def transpose(self, *axes):
        """转置数组。"""
        return _ndarray_methods().transpose(self, *axes)

    def swapaxes(self, axis1, axis2):
        """交换两个轴。"""
        return _ndarray_methods().swapaxes(self, axis1, axis2)

    def squeeze(self, axis=None):
        """移除长度为1的轴。"""
        return _ndarray_methods().squeeze(self, axis)

    def max(self, axis=None):
        """返回最大值。"""
        return _ndarray_methods().max(self, axis)

    def min(self, axis=None):
        """返回最小值。"""
        return _ndarray_methods().min(self, axis)

    def mean(self, axis=None):
        """计算平均值。"""
        return _ndarray_methods().mean(self, axis)

    def std(self, axis=None):
        """计算标准差。"""
        return _ndarray_methods().std(self, axis)

    def var(self, axis=None):
        """计算方差。"""
        return _ndarray_methods().var(self, axis)

    def sum(self, axis=None):
        """计算和。"""
        return _ndarray_methods().sum(self, axis)

    def prod(self, axis=None):
        """计算乘积。"""
        return _ndarray_methods().prod(self, axis)

    def cumsum(self, axis=None):
        """计算累积和。"""
        return _ndarray_methods().cumsum(self, axis)

    def cumprod(self, axis=None):
        """计算累积乘积。"""
        return _ndarray_methods().cumprod(self, axis)

    def argmax(self, axis=None):
        """返回最大值的索引。"""
        return _ndarray_methods().argmax(self, axis)

    def argmin(self, axis=None):
        """返回最小值的索引。"""
        return _ndarray_methods().argmin(self, axis)

    def argsort(self, axis=-1):
        """返回排序后的索引。"""
        return _ndarray_methods().argsort(self, axis)

    def sort(self, axis=-1):
        """原地排序。"""
        _ndarray_methods().sort(self, axis)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """返回对角线元素。"""
        return _ndarray_methods().diagonal(self, offset, axis1, axis2)

    def trace(self, offset=0, axis1=0, axis2=1):
        """计算迹。"""
        return _ndarray_methods().trace(self, offset, axis1, axis2)

    def fill(self, value):
        """用值填充数组。"""
        _ndarray_methods().fill(self, value)

    def item(self, *args):
        """获取单个元素。"""
        return _ndarray_methods().item(self, *args)

    def tolist(self):
        """转换为 Python 列表。"""
        return _ndarray_methods().tolist(self)

    def take(self, indices, axis=None):
        """根据索引取元素。"""
        return _ndarray_methods().take(self, indices, axis)

    def put(self, indices, values):
        """设置指定位置的元素。"""
        _ndarray_methods().put(self, indices, values)

    def repeat(self, repeats, axis=None):
        """重复元素。"""
        return _ndarray_methods().repeat(self, repeats, axis)

    def nonzero(self):
        """返回非零元素索引。"""
        return _ndarray_methods().nonzero(self)

    # ========== 数学方法 ==========

    def sin(self):
        """计算正弦。"""
        return _math_funcs().sin(self)

    def cos(self):
        """计算余弦。"""
        return _math_funcs().cos(self)

    def tan(self):
        """计算正切。"""
        return _math_funcs().tan(self)

    def sqrt(self):
        """计算平方根。"""
        return _math_funcs().sqrt(self)

    def abs(self):
        """计算绝对值。"""
        return _math_funcs().abs(self)

    def exp(self):
        """计算指数。"""
        return _math_funcs().exp(self)

    def log(self):
        """计算自然对数。"""
        return _math_funcs().log(self)

    def log10(self):
        """计算以10为底的对数。"""
        return _math_funcs().log10(self)

    def log2(self):
        """计算以2为底的对数。"""
        return _math_funcs().log2(self)

    def clip(self, a_min, a_max):
        """限制值范围。"""
        return _math_funcs().clip(self, a_min, a_max)

    def round(self, decimals=0):
        """四舍五入到指定小数位。"""
        if decimals == 0:
            return _wrap_result(self._array.__round__(None), self._dtype)
        return _wrap_result(self._array.__round__(decimals), self._dtype)

    def floor(self):
        """向下取整。"""
        return _math_funcs().floor(self)

    def ceil(self):
        """向上取整。"""
        return _math_funcs().ceil(self)

    def all(self):
        """测试是否所有元素为真。"""
        return _core.all(self._array)

    def any(self):
        """测试是否有任何元素为真。"""
        return _core.any(self._array)


def _ensure(x):
    """将列表/元组转换为 ndarray。"""
    if isinstance(x, (list, tuple)):
        return _core.ndarray(x)
    elif hasattr(x, '_array'):
        return x._array
    elif hasattr(x, '__class__') and x.__class__.__name__ == 'ndarray':
        return x
    return x


def _wrap_result(result, dtype="float64"):
    """将原始 ndarray 结果包装到 ndarray 类中。"""
    if hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return ndarray._wrap(result, _dtype=dtype)
    return result


def _scalar(x):
    """转换为标量。"""
    if hasattr(x, 'tolist'):
        return x.tolist()
    return x


def _ndarray_methods():
    """延迟导入 array_methods 模块。"""
    return array_methods.NdArrayMethods


def _math_funcs():
    """延迟导入 math_functions 模块。"""
    from . import math_functions as mf
    return mf


def _stat_funcs():
    """延迟导入 statistics 模块。"""
    from . import statistics as st
    return st


def _array_ops():
    """延迟导入 array_ops 模块。"""
    from . import array_ops as ao
    return ao


# ========== 类型代码映射 ==========

_RS_DTYPE_CODES = {
    # 有符号整数
    'i1': 'int8', 'int8': 'int8',
    'i2': 'int16', 'int16': 'int16',
    'i4': 'int32', 'int32': 'int32',
    'i8': 'int64', 'int64': 'int64',
    # 无符号整数
    'u1': 'uint8', 'uint8': 'uint8',
    'u2': 'uint16', 'uint16': 'uint16',
    'u4': 'uint32', 'uint32': 'uint32',
    'u8': 'uint64', 'uint64': 'uint64',
    # 浮点
    'f2': 'float16', 'float16': 'float16',
    'f4': 'float32', 'float32': 'float32',
    'f8': 'float64', 'float64': 'float64', 'float': 'float64',
    # 复数
    'c8': 'complex64', 'c64': 'complex64', 'complex64': 'complex64',
    'c16': 'complex128', 'c128': 'complex128', 'complex128': 'complex128', 'complex': 'complex128',
    # 其他
    'bool': 'bool', 'b1': 'bool',
    'object': 'object', 'O': 'object',
}

_PY_TYPE_TO_RS = {
    int: 'int64',
    float: 'float64',
    complex: 'complex128',
    bool: 'bool',
    bytes: 'bytes',
    str: 'str',
}

# 运行时收集已知的 numpy 类型名称
_numpy_type_names = {}


def _init_numpy_types():
    """尝试导入 numpy 并注册其标量类型名称。"""
    global _numpy_type_names
    try:
        import numpy
    except ImportError:
        return
    for name in dir(numpy):
        if name.startswith('int') or name.startswith('uint') or name.startswith('float') \
                or name.startswith('complex') or name == 'bool_' or name == 'bool':
            _numpy_type_names[name] = name


def _resolve_type_name(tp):
    """将各种 dtype 输入解析为类型名字符串。"""
    if isinstance(tp, str):
        # 直接映射
        lower = tp.lower()
        if lower in _RS_DTYPE_CODES:
            return _RS_DTYPE_CODES[lower]
        return tp  # 未知字符串，原样返回
    if isinstance(tp, DType):
        return tp.name
    if type(tp).__name__ == 'dtype' and hasattr(tp, 'name'):
        return _RS_DTYPE_CODES.get(tp.name, tp.name)
    # Python 内置类型
    if tp in _PY_TYPE_TO_RS:
        return _PY_TYPE_TO_RS[tp]
    # 检查类名
    name = getattr(tp, '__name__', None) or getattr(tp, 'name', str(tp))
    # 先查已知的 numpy 类型名
    if name in _numpy_type_names:
        return _numpy_type_names[name]
    # 再查代码表
    if name in _RS_DTYPE_CODES:
        return _RS_DTYPE_CODES[name]
    # int_ / float_ 等 numpy 特殊名称
    name_clean = name.rstrip('_')
    if name_clean in _RS_DTYPE_CODES:
        return _RS_DTYPE_CODES[name_clean]
    return name


# 类型名 → 类型码 逆向映射（用于结构化 dtype）
_RS_NAME_TO_CODE = {}
for _code, _name in _RS_DTYPE_CODES.items():
    # 只保留短码映射，忽略 'int32': 'int32' 这样的自映射
    if len(_code) <= 4 and _code not in ('float', 'complex', 'bool') and _code != _name:
        _RS_NAME_TO_CODE[_name] = _code


class DType:
    """表示元素数据类型。"""

    def __init__(self, name, fields=None):
        self._name = name
        self._fields = fields  # [(field_name, type_code), ...] 或 None

    @property
    def name(self):
        return self._name

    def _field_to_code(self, tp):
        """将字段类型转换为短类型码。"""
        if isinstance(tp, str):
            return _RS_NAME_TO_CODE.get(_resolve_type_name(tp), tp)
        resolved = _resolve_type_name(tp)
        return _RS_NAME_TO_CODE.get(resolved, resolved)

    def __str__(self):
        if self._fields:
            parts = [f"('{f[0]}', '{self._field_to_code(f[1])}')" for f in self._fields]
            return "[" + ", ".join(parts) + "]"
        return self._name

    def __repr__(self):
        if self._fields:
            parts = [f"('{f[0]}', '{self._field_to_code(f[1])}')" for f in self._fields]
            return "dtype([" + ", ".join(parts) + "])"
        return f"dtype('{self._name}')"

    def __eq__(self, other):
        if isinstance(other, DType):
            if self._fields is not None or other._fields is not None:
                return self._fields == other._fields
            return self._name == other._name
        if isinstance(other, str):
            return self._name == other
        return NotImplemented


# 立即初始化 numpy 类型名称
_init_numpy_types()


def dtype(obj):
    """创建 dtype 对象。

    >>> dt = dtype('i4')
    >>> print(dt)
    int32

    >>> dt = dtype([('age', np.int8)])
    >>> print(dt)
    [('age', 'i1')]
    """
    if isinstance(obj, (list, tuple)):
        # 结构化 dtype
        fields = []
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                fields.append((item[0], item[1]))
        if fields:
            return DType("void", fields=fields)
    name = _resolve_type_name(obj)
    return DType(name)


# ========== 标量类型别名 ==========

int8 = type('int8', (), {})
int16 = type('int16', (), {})
int32 = type('int32', (), {})
int64 = type('int64', (), {})
uint8 = type('uint8', (), {})
uint16 = type('uint16', (), {})
uint32 = type('uint32', (), {})
uint64 = type('uint64', (), {})
float16 = type('float16', (), {})
float32 = type('float32', (), {})
float64 = type('float64', (), {})
complex64 = type('complex64', (), {})
complex128 = type('complex128', (), {})


# ========== 构造/工厂函数 ==========

def array(data, dtype=None, copy=True, order='K', subok=False, ndmin=0):
    """创建数组。"""
    _dtype = _resolve_dtype(dtype)
    arr = ndarray(data, _dtype=_dtype)
    if ndmin > arr.ndim:
        new_shape = (1,) * (ndmin - arr.ndim) + arr.shape
        arr = ndarray._wrap(arr._array.reshape(new_shape), _dtype=_dtype)
    return arr


def asarray(a, dtype=None, order=None):
    """转换输入为数组。"""
    if isinstance(a, ndarray):
        return a
    _dtype = "float64"
    if dtype is not None:
        dt_str = dtype if isinstance(dtype, str) else dtype.__name__
        if dt_str in ("complex", "complex128", "complex64", "cfloat", "cdouble"):
            _dtype = "complex128"
    return ndarray(a, _dtype=_dtype)


def asanyarray(a, dtype=None, order=None):
    """转换输入为 ndarray。"""
    return asarray(a, dtype, order)


def copy(a, order='K'):
    """返回数组副本。"""
    return ndarray(a).copy()


def _resolve_dtype(dtype):
    """解析 dtype 字符串/Types/DType 为内部表示。"""
    if dtype is None:
        return "float64"
    if isinstance(dtype, DType):
        return dtype.name
    dt_str = dtype if isinstance(dtype, str) else dtype.__name__
    if dt_str in ("complex", "complex128", "complex64", "cfloat", "cdouble"):
        return "complex128"
    return "float64"


def zeros(shape, dtype=None, order='C'):
    """返回指定形状的零数组。"""
    _dtype = _resolve_dtype(dtype)
    return ndarray(_core.zeros(shape), _dtype=_dtype)


def ones(shape, dtype=None, order='C'):
    """返回指定形状的1数组。"""
    _dtype = _resolve_dtype(dtype)
    return ndarray(_core.ones(shape), _dtype=_dtype)


def empty(shape, dtype=None, order='C'):
    """返回指定形状的空数组。"""
    _dtype = _resolve_dtype(dtype)
    return ndarray(_core.empty(shape), _dtype=_dtype)


def full(shape, fill_value, dtype=None, order='C'):
    """返回指定形状的填充数组。"""
    _dtype = _resolve_dtype(dtype)
    return ndarray(_core.full(shape, fill_value), _dtype=_dtype)


def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    """返回与输入形状相同的零数组。"""
    arr = ndarray(a)
    return zeros(arr.shape)


def ones_like(a, dtype=None, order='K', subok=True, shape=None):
    """返回与输入形状相同的1数组。"""
    arr = ndarray(a)
    return ones(arr.shape)


def empty_like(a, dtype=None, order='K', subok=True, shape=None):
    """返回与输入形状相同的空数组。"""
    arr = ndarray(a)
    return empty(arr.shape)


def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):
    """返回与输入形状相同的填充数组。"""
    arr = ndarray(a)
    return full(arr.shape, fill_value)


def eye(N, M=None, k=0, dtype=None, order='C'):
    """返回对角线为1的二维数组。"""
    _dtype = _resolve_dtype(dtype)
    return ndarray(_core.eye(N, M, k), _dtype=_dtype)


def identity(n, dtype=None):
    """返回单位矩阵。"""
    return eye(n, dtype=dtype)


def arange(start=0, stop=None, step=1, dtype=None):
    """返回给定间隔内均匀间隔的值。"""
    if stop is None:
        stop = start
        start = 0
    return ndarray(_core.arange(start, stop, step))


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """返回指定间隔内均匀间隔的数字。"""
    start_val = _scalar(_ensure(start))
    stop_val = _scalar(_ensure(stop))
    result = ndarray(_core.linspace(start_val, stop_val, num))
    if retstep:
        step = (stop_val - start_val) / (num - 1) if num > 1 else 0
        return result, step
    return result


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """返回对数刻度上均匀间隔的数字。"""
    return power(ndarray([float(base)]), linspace(start, stop, num, endpoint))


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """返回几何级数上均匀间隔的数字。"""
    import math
    if start <= 0 or stop <= 0:
        raise ValueError("geomspace requires positive start and stop values")
    log_start = math.log(start)
    log_stop = math.log(stop)
    return exp(linspace(log_start, log_stop, num, endpoint))


# ========== 索引函数 ==========

def where(condition, x=None, y=None):
    """根据条件返回元素或索引。"""
    if x is not None and y is not None:
        return ndarray(_core.where(_ensure(condition), _ensure(x), _ensure(y)))
    return ndarray(_core.nonzero(_ensure(condition)))


def take(a, indices, axis=None, out=None, mode='raise'):
    """沿指定轴取元素。"""
    arr = ndarray(a)
    return arr.take(indices, axis)


def put(a, indices, values, mode='raise'):
    """设置指定位置的元素。"""
    arr = ndarray(a)
    arr.put(indices, values)


def select(condlist, choicelist, default=0):
    """根据条件列表选择值。"""
    result = ndarray([default] * ndarray(condlist[0]).size)
    for cond, choice in zip(condlist, choicelist):
        mask = ndarray(cond)
        result = where(mask, choice, result)
    return result


def nonzero(a):
    """返回非零元素的索引，返回元组形式的数组（兼容NumPy）。"""
    raw = _core.nonzero(_ensure(a))
    # raw is Vec<Vec<usize>>, convert to tuple of 1D arrays
    result = []
    for indices in raw:
        arr = _core.ndarray(list(indices))
        result.append(ndarray(arr))
    return tuple(result)


def argwhere(a):
    """查找非零元素的索引。"""
    return ndarray(_core.argwhere(_ensure(a)))


def flatnonzero(a):
    """返回扁平化数组中非零元素的索引。"""
    return ndarray(_core.flatnonzero(_ensure(a)))


# ========== 缺失构造函数 ==========

def fromfunction(function, shape, *, dtype=None, **kwargs):
    """根据函数和形状创建数组。"""
    indices = [ndarray(list(range(d))) for d in shape]
    grid = _core.meshgrid(*indices, indexing='ij')
    return function(*grid, **kwargs)


def frombuffer(buffer, dtype=None, count=-1, offset=0, *, like=None):
    """从缓冲区创建一维数组（使用 Rust 层实现）。"""
    if isinstance(buffer, bytes):
        return ndarray(_core.bytes_to_floats(buffer, count))
    return ndarray(list(buffer))


class _RClass:
    """行连接辅助类，模拟 np.r_。"""
    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        arrays = []
        for it in item:
            arrays.append(ndarray(it))
        return concatenate([a.ravel() for a in arrays])


r_ = _RClass()


def mgrid(*ranges):
    """密集网格索引，返回密集网格数组。"""
    arrays = []
    for r in ranges:
        if isinstance(r, slice):
            arrays.append(ndarray(arange(r.start or 0, r.stop, r.step or 1)))
        elif isinstance(r, (int, float)):
            arrays.append(ndarray([float(r)]))
        else:
            arrays.append(ndarray(r))
    grids = _core.meshgrid(*arrays, indexing='ij')
    if isinstance(grids, tuple) and len(grids) > 1:
        return stack(grids, axis=0)
    return grids


def ogrid(*ranges):
    """开放网格索引，返回开放网格数组。"""
    arrays = []
    for r in ranges:
        if isinstance(r, slice):
            arrays.append(ndarray(arange(r.start or 0, r.stop, r.step or 1)))
        elif isinstance(r, (int, float)):
            arrays.append(ndarray([float(r)]))
        else:
            arrays.append(ndarray(r))
    return _core.meshgrid(*arrays, indexing='ij')


# ========== FFT 函数 ==========

def fft(a, n=None, axis=-1):
    """计算一维离散傅里叶变换。"""
    if isinstance(a, ndarray):
        a = a.tolist()
    elif isinstance(a, (list, tuple)):
        pass
    else:
        a = [float(a)]
    return _core.py_fft(a)


def ifft(a, n=None, axis=-1):
    """计算一维逆离散傅里叶变换。"""
    return _core.py_ifft(a)


def rfft(a, n=None, axis=-1):
    """计算实输入的一维离散傅里叶变换。"""
    if isinstance(a, ndarray):
        a = a.tolist()
    elif isinstance(a, (list, tuple)):
        pass
    else:
        a = [float(a)]
    return _core.py_rfft(a)


def irfft(a, n=None, axis=-1):
    """计算 rfft 的逆变换。"""
    return _core.py_irfft(a, n)


# ========== 常量 ==========
pi = 3.141592653589793
e = 2.718281828459045
euler_gamma = 0.5772156649015328606
inf = float('inf')
nan = float('nan')
newaxis = None


# ========== 判断函数 ==========

def isnan(x):
    """逐元素检测是否为 NaN。"""
    arr = ndarray(x)
    return ndarray(_core.isnan(arr._array))


def isinf(x):
    """逐元素检测是否为无穷大。"""
    arr = ndarray(x)
    return ndarray(_core.isinf(arr._array))


def isfinite(x):
    """逐元素检测是否为有限值。"""
    arr = ndarray(x)
    return ndarray(_core.isfinite(arr._array))


# 数组操作函数
reshape = _array_ops_module.reshape
ravel = _array_ops_module.ravel
moveaxis = _array_ops_module.moveaxis
rollaxis = _array_ops_module.rollaxis
broadcast_to = _array_ops_module.broadcast_to
transpose = _array_ops_module.transpose
swapaxes = _array_ops_module.swapaxes
expand_dims = _array_ops_module.expand_dims
squeeze = _array_ops_module.squeeze
concatenate = _array_ops_module.concatenate
stack = _array_ops_module.stack
vstack = _array_ops_module.vstack
hstack = _array_ops_module.hstack
dstack = _array_ops_module.dstack
column_stack = _array_ops_module.column_stack
split = _array_ops_module.split
hsplit = _array_ops_module.hsplit
vsplit = _array_ops_module.vsplit
dsplit = _array_ops_module.dsplit
tile = _array_ops_module.tile
repeat = _array_ops_module.repeat
flip = _array_ops_module.flip
fliplr = _array_ops_module.fliplr
flipud = _array_ops_module.flipud
roll = _array_ops_module.roll
rot90 = _array_ops_module.rot90

# 数学函数
sin = _math_functions_module.sin
cos = _math_functions_module.cos
tan = _math_functions_module.tan
arcsin = _math_functions_module.arcsin
arccos = _math_functions_module.arccos
arctan = _math_functions_module.arctan
arctan2 = _math_functions_module.arctan2
deg2rad = _math_functions_module.deg2rad
rad2deg = _math_functions_module.rad2deg
sinh = _math_functions_module.sinh
cosh = _math_functions_module.cosh
tanh = _math_functions_module.tanh
asinh = _math_functions_module.asinh
acosh = _math_functions_module.acosh
atanh = _math_functions_module.atanh
exp = _math_functions_module.exp
expm1 = _math_functions_module.expm1
log = _math_functions_module.log
log10 = _math_functions_module.log10
log2 = _math_functions_module.log2
log1p = _math_functions_module.log1p
around = _math_functions_module.around
floor = _math_functions_module.floor
ceil = _math_functions_module.ceil
trunc = _math_functions_module.trunc
fix = _math_functions_module.fix
sqrt = _math_functions_module.sqrt
square = _math_functions_module.square
cbrt = _math_functions_module.cbrt
abs = _math_functions_module.abs
sign = _math_functions_module.sign
clip = _math_functions_module.clip
sinc = _math_functions_module.sinc
heaviside = _math_functions_module.heaviside
add = _math_functions_module.add
subtract = _math_functions_module.subtract
multiply = _math_functions_module.multiply
divide = _math_functions_module.divide
power = _math_functions_module.power
mod = _math_functions_module.mod
greater = _math_functions_module.greater
less = _math_functions_module.less
equal = _math_functions_module.equal
not_equal = _math_functions_module.not_equal
greater_equal = _math_functions_module.greater_equal
less_equal = _math_functions_module.less_equal
logical_and = _math_functions_module.logical_and
logical_or = _math_functions_module.logical_or
logical_xor = _math_functions_module.logical_xor
isclose = _math_functions_module.isclose
allclose = _math_functions_module.allclose

# 统计函数
sum = _statistics_module.sum
mean = _statistics_module.mean
std = _statistics_module.std
var = _statistics_module.var
min = _statistics_module.min
max = _statistics_module.max
ptp = _statistics_module.ptp
median = _statistics_module.median
percentile = _statistics_module.percentile
quantile = _statistics_module.quantile
nanmedian = _statistics_module.nanmedian
nanpercentile = _statistics_module.nanpercentile
argmax = _statistics_module.argmax
argmin = _statistics_module.argmin
argsort = _statistics_module.argsort
sort = _statistics_module.sort
searchsorted = _statistics_module.searchsorted
cov = _statistics_module.cov
corrcoef = _statistics_module.corrcoef
histogram = _statistics_module.histogram
histogram2d = _statistics_module.histogram2d
histogramdd = _statistics_module.histogramdd
digitize = _statistics_module.digitize

# 子模块

linalg = _linalg_module()
random = _random_module()


# ========== 导出列表 ==========
__all__ = [
    'ndarray', 'NdArrayIter',
    'array', 'asarray', 'asanyarray', 'copy',
    'zeros', 'ones', 'empty', 'full',
    'zeros_like', 'ones_like', 'empty_like', 'full_like',
    'eye', 'identity',
    'arange', 'linspace', 'logspace', 'geomspace',
    'fromfunction', 'frombuffer', 'r_', 'mgrid', 'ogrid',
    'reshape', 'ravel', 'moveaxis', 'rollaxis', 'broadcast_to',
    'transpose', 'swapaxes', 'expand_dims', 'squeeze',
    'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',
    'split', 'hsplit', 'vsplit', 'dsplit',
    'tile', 'repeat',
    'flip', 'fliplr', 'flipud', 'roll', 'rot90',
    'where', 'take', 'put', 'select', 'nonzero', 'argwhere', 'flatnonzero',
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
    'deg2rad', 'rad2deg',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    'exp', 'expm1', 'log', 'log10', 'log2', 'log1p',
    'around', 'floor', 'ceil', 'trunc', 'fix',
    'sqrt', 'square', 'cbrt', 'abs', 'sign', 'clip', 'sinc', 'heaviside',
    'add', 'subtract', 'multiply', 'divide', 'power', 'mod',
    'greater', 'less', 'equal', 'logical_and', 'logical_or', 'isclose', 'allclose',
    'sum', 'mean', 'std', 'var', 'min', 'max', 'ptp',
    'median', 'percentile', 'quantile', 'nanmedian', 'nanpercentile',
    'argmax', 'argmin', 'argsort', 'sort', 'searchsorted',
    'cov', 'corrcoef',
    'histogram', 'histogram2d', 'histogramdd', 'digitize',
    'fft', 'ifft', 'rfft', 'irfft',
    'pi', 'e', 'euler_gamma', 'inf', 'nan', 'newaxis',
    'isnan', 'isinf', 'isfinite',
    'save', 'load', 'loadtxt', 'savetxt', 'savez',
    'Poly', 'polyval', 'polyfit', 'polyder', 'polyint', 'polyroots',
    'linalg', 'random', 'load_npz'
]
