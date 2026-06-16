"""run_all_tests.py - rsnumpy 全面测试入口

按顺序运行所有测试脚本，打印总览。
"""
import sys
import subprocess
import os
import time


TEST_FILES = [
    ('test_01_ndarray_basics.py', 'ndarray 基础'),
    ('test_02_array_manipulation.py', '数组操作'),
    ('test_03_indexing_broadcasting_ufunc.py', '索引/广播/ufunc'),
    ('test_04_math_functions.py', '数学函数'),
    ('test_05_statistics.py', '统计函数'),
    ('test_06_linalg.py', '线性代数'),
    ('test_07_fft.py', 'FFT'),
    ('test_08_random.py', '随机数'),
    ('test_09_polynomial.py', '多项式'),
    ('test_10_constants_io.py', '常量与 I/O'),
]


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 60)
    print("rsnumpy 全面测试套件")
    print("=" * 60)

    total_pass = 0
    total_fail = 0
    failed_files = []

    start = time.time()
    for fname, desc in TEST_FILES:
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print(f"\n[SKIP] {fname} - 文件不存在")
            continue

        print(f"\n>>> {fname}  ({desc})")
        print("-" * 60)
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True
        )
        # 打印尾部若干行以展示结果
        out = result.stdout
        for line in out.splitlines()[-6:]:
            print(line)

        if result.returncode != 0:
            failed_files.append(fname)
            print(f"\n[STDERR]\n{result.stderr[:500]}")
        else:
            # 提取 PASS / FAIL
            for line in out.splitlines():
                if '[PASS]' in line and '[FAIL]' in line:
                    try:
                        parts = line.split()
                        p_idx = parts.index('[PASS]')
                        f_idx = parts.index('[FAIL]')
                        total_pass += int(parts[p_idx + 1])
                        total_fail += int(parts[f_idx + 1])
                    except Exception:
                        pass

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"总 PASS: {total_pass}")
    print(f"总 FAIL: {total_fail}")
    print(f"失败文件: {len(failed_files)}")
    for f in failed_files:
        print(f"  - {f}")
    print(f"耗时: {elapsed:.2f}s")
    return 0 if total_fail == 0 and not failed_files else 1


if __name__ == '__main__':
    sys.exit(main())
