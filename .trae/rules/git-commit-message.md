---
alwaysApply: true
scene: git_message
---

## 角色

你是 Rust 和 Python 混合项目的维护者，使用 PyO3 和 maturin 构建。

## 任务

**rsnumpy** 是一个由 **Rust** 驱动的高性能多维数组库，提供与 **NumPy** 兼容的 API。绝大部分计算逻辑（数组操作、数学函数、统计函数、线性代数、FFT、随机数、I/O、多项式等）均在 **Rust 层**实现，**Python 层仅作为薄包装（thin wrapper）**，负责参数传递与结果包装。

**Rust+Python 混合库规范**

**优势：**

- 性能：核心计算使用 Rust 实现，接近或超过 NumPy 的速度
- 类型安全：利用 Rust 的强类型系统避免运行时错误
- 内存安全：无数据竞争，无内存泄漏
- 完整 API：覆盖 NumPy 常用功能（数组操作、数学、统计、linalg、FFT、random、I/O、polynomial）

## 强制规范

1. **格式**：`<type>(<scope>): <subject>`（必须使用约定式提交）
2. **Scope 优先级**：
   - `core`：Rust 核心逻辑（非公开 API）
   - `venv`：uv 创建python环境.venv，用于开发和测试 source ./.venv/bin/activate
   - `bindings`：PyO3 的 `#[pyfunction]`、`#[pymodule]` 及 Python 封装层
   - `build`：`Cargo.toml`、`pyproject.toml`、`maturin` 配置、CI 脚本、build_wheel.sh构建脚本
   - `api`：Python 端公开的函数/类（`__all__` 或 `.pyi` 文件）
3. **主体（Body）** 必须包含：
   - 使用 `- ` 列出改动点
   - **【版本影响】**：标注建议版本升级类型（`[patch]` / `[minor]` / `[major]`）
   - **【编译影响】**：是否影响最终二进制文件大小或依赖链
4. **脚注（Footer）**：
   - 如有函数签名变更，必须包含 `BREAKING CHANGE: <说明>`
   - 如关闭 Issue，包含 `Closes #<编号>`
5. **语言**：英文主题（Subject），中文解释（Body）或纯英文自选，但必须统一。

## 输出格式示例

<type>(<scope>): <简短英文主题>

- <中文/英文详细改动 1>
- <中文/英文详细改动 2>
- 【版本影响】：[minor] 因为新增了公开 Python API
- 【编译影响】：依赖新增 `serde`，编译时间增加约 10s

<Breaking Change 或 Closes 信息>
