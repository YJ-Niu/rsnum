"""I/O 模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import rsnum._core as _core


def _get_ndarray():
    from .__init__ import ndarray as _n
    return _n


def _ensure_raw(a):
    if hasattr(a, '_array'):
        return a._array
    return _core.ndarray(a)


def _wrap(result):
    return _get_ndarray()(result)


def save(file, arr):
    """将数组保存为二进制文件 (.npy)。"""
    raw = _ensure_raw(arr)
    _core.save_npy(file, raw)


def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    """从二进制文件加载数组 (.npy)。"""
    return _wrap(_core.load_npy(file))


def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
    """将数组保存为文本文件。"""
    raw = _ensure_raw(X)
    _core.save_text(fname, raw, fmt, delimiter)


def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes',
            max_rows=None, *, like=None):
    """从文本文件加载数据。"""
    if delimiter is None:
        delimiter = ""
    return _wrap(_core.load_text(fname, delimiter, skiprows))


def savez(file, *args, **kwds):
    """将多个数组保存为未压缩的 .npz 文件。"""
    import zipfile
    with zipfile.ZipFile(file, 'w') as zf:
        named = {}
        for i, arg in enumerate(args):
            name = 'arr_%d' % i
            named[name] = arg
        named.update(kwds)
        for name, arr in named.items():
            if not name.endswith('.npy'):
                fname = name + '.npy'
            else:
                fname = name
            path = '/tmp/_rsnum_' + fname
            save(path, arr)
            zf.write(path, fname)


def load_npz(file):
    """从 .npz 文件加载数组（返回 dict）。"""
    import zipfile
    zf = zipfile.ZipFile(file, 'r')
    result = {}
    for name in zf.namelist():
        if name.endswith('.npy'):
            key = name[:-4]
            with zf.open(name) as f:
                content = f.read()
                path = '/tmp/_rsnum_load_' + name
                with open(path, 'wb') as tmp:
                    tmp.write(content)
                result[key] = load(path)
    return result
