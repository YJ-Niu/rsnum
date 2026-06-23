"""rsnumpy.matlib - 矩阵库模块"""

import rsnumpy._core as _core


def empty(shape, dtype=None, order='C'):
    """返回一个新的未初始化矩阵。"""
    from .__init__ import ndarray
    if isinstance(shape, int):
        shape = (shape, shape)
    raw = _core.random.uniform(0.0, 1e-300, shape)
    return ndarray(raw, _dtype=dtype or 'float64')


def zeros(shape, dtype=None, order='C'):
    """创建一个以 0 填充的矩阵。"""
    from .__init__ import ndarray
    if isinstance(shape, int):
        shape = (shape, shape)
    size = shape[0] * shape[1]
    raw_data = [0.0] * size
    arr = ndarray(raw_data, _dtype=dtype or 'float64')
    return arr.reshape(shape)


def ones(shape, dtype=None, order='C'):
    """创建一个以 1 填充的矩阵。"""
    from .__init__ import ndarray
    if isinstance(shape, int):
        shape = (shape, shape)
    size = shape[0] * shape[1]
    raw_data = [1.0] * size
    arr = ndarray(raw_data, _dtype=dtype or 'float64')
    return arr.reshape(shape)


def eye(n, M=None, k=0, dtype=None, order='C'):
    """返回一个矩阵，对角线元素为 1，其他位置为零。"""
    if M is None:
        M = n
    arr = zeros((n, M), dtype=dtype)
    for i in range(n):
        j = i + k
        if 0 <= j < M:
            arr[i, j] = 1.0
    return arr


def identity(n, dtype=None):
    """返回给定大小的单位矩阵。"""
    return eye(n, n, 0, dtype)


def rand(*args):
    """创建一个给定大小的矩阵，数据是随机填充的。"""
    from .__init__ import ndarray
    if len(args) == 1:
        shape = (args[0], args[0])
    else:
        shape = args
    raw = _core.random.rand(*shape)
    return ndarray(raw, _dtype='float64')


def randn(*args):
    """生成指定形状的标准正态分布随机矩阵。"""
    from .__init__ import ndarray
    if len(args) == 1:
        shape = (args[0], args[0])
    else:
        shape = args
    raw = _core.random.randn(*shape)
    return ndarray(raw, _dtype='float64')


__all__ = ['empty', 'zeros', 'ones', 'eye', 'identity', 'rand', 'randn']
