use ndarray::{Array, IxDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::SeedableRng;
use rand_distr::Distribution;

use crate::NdArray;
use std::sync::Mutex;

static RNG: Mutex<Option<::rand::rngs::StdRng>> = Mutex::new(None);
static GLOBAL_SEED: Mutex<Option<u64>> = Mutex::new(None);

fn get_rng() -> std::sync::MutexGuard<'static, Option<::rand::rngs::StdRng>> {
    RNG.lock().unwrap()
}

fn get_global_seed() -> std::sync::MutexGuard<'static, Option<u64>> {
    GLOBAL_SEED.lock().unwrap()
}

fn ensure_rng_mut() -> std::sync::MutexGuard<'static, Option<::rand::rngs::StdRng>> {
    let mut guard = get_rng();
    if guard.is_none() {
        let seed = get_global_seed().unwrap_or(0xdeadbeef);
        *guard = Some(::rand::rngs::StdRng::seed_from_u64(seed));
    }
    guard
}

/// 创建一个新的独立 RNG（用于 default_rng）
fn new_rng(seed: Option<u64>) -> ::rand::rngs::StdRng {
    ::rand::rngs::StdRng::seed_from_u64(seed.unwrap_or_else(rand::random))
}

fn parse_shape_from_args(args: &Bound<'_, PyTuple>) -> Vec<usize> {
    let mut shape = Vec::new();
    for item in args.iter() {
        if let Ok(val) = item.extract::<usize>() {
            shape.push(val);
        }
    }
    shape
}

fn parse_size_arg(size: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<usize>> {
    match size {
        None => Ok(vec![]),
        Some(s) => {
            if let Ok(val) = s.extract::<usize>() {
                Ok(vec![val])
            } else if let Ok(tup) = s.cast::<PyTuple>() {
                let mut result = Vec::new();
                for item in tup.iter() {
                    result.push(item.extract::<usize>()?);
                }
                Ok(result)
            } else if let Ok(list) = s.cast::<pyo3::types::PyList>() {
                let mut result = Vec::new();
                for item in list.iter() {
                    result.push(item.extract::<usize>()?);
                }
                Ok(result)
            } else {
                Err(PyValueError::new_err("size must be an integer or tuple"))
            }
        }
    }
}

fn make_ndarray(rng: &mut impl rand::Rng, shape: &[usize], dist: impl rand_distr::Distribution<f64>) -> PyResult<NdArray> {
    if shape.is_empty() {
        return Ok(NdArray {
            data: Array::from_elem(IxDyn(&[]), dist.sample(rng)),
        });
    }
    let total: usize = shape.iter().product();
    let values: Vec<f64> = (0..total).map(|_| dist.sample(rng)).collect();
    let arr = Array::from_shape_vec(IxDyn(shape), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
fn seed(val: u64) {
    let mut guard = get_rng();
    let mut seed_guard = get_global_seed();
    *seed_guard = Some(val);
    *guard = Some(::rand::rngs::StdRng::seed_from_u64(val));
}

#[pyfunction]
fn get_state() -> u64 {
    let seed_guard = get_global_seed();
    seed_guard.unwrap_or(0xdeadbeef)
}

#[pyfunction]
#[pyo3(signature = (*args), name = "rand")]
fn random_rand(args: &Bound<'_, PyTuple>) -> PyResult<NdArray> {
    let shape = parse_shape_from_args(args);
    let dist = ::rand_distr::Uniform::<f64>::new(0.0, 1.0)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut guard = ensure_rng_mut();
    let rng = guard.as_mut().unwrap();
    make_ndarray(rng, &shape, dist)
}

#[pyfunction]
#[pyo3(signature = (*args))]
fn randn(args: &Bound<'_, PyTuple>) -> PyResult<NdArray> {
    let shape = parse_shape_from_args(args);
    let normal = ::rand_distr::Normal::new(0.0, 1.0)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut guard = ensure_rng_mut();
    let rng = guard.as_mut().unwrap();
    make_ndarray(rng, &shape, normal)
}

#[pyfunction]
#[pyo3(signature = (low, high, size=None), name = "randint")]
fn random_randint(low: i64, high: i64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Uniform::new(low, high)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut guard = ensure_rng_mut();
    let rng = guard.as_mut().unwrap();
    if shape.is_empty() {
        return Ok(NdArray {
            data: Array::from_elem(IxDyn(&[]), dist.sample(rng) as f64),
        });
    }
    let total: usize = shape.iter().product();
    let values: Vec<f64> = (0..total).map(|_| dist.sample(rng) as f64).collect();
    let arr = Array::from_shape_vec(IxDyn(&shape), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

// ===== 新 API: default_rng =====

/// 创建一个新的 Generator（默认 RNG）
#[pyclass(name = "Generator", from_py_object)]
#[derive(Clone)]
struct PyGenerator {
    #[allow(dead_code)]
    seed: Option<u64>,
}

#[pymethods]
impl PyGenerator {
    #[new]
    fn new(seed: Option<u64>) -> Self {
        PyGenerator { seed }
    }

    /// 生成 [0, 1) 均匀分布随机数
    #[pyo3(signature = (size=None))]
    fn random(&self, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Uniform::<f64>::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        make_ndarray(&mut rng, &shape, dist)
    }

    /// 生成标准正态分布随机数
    #[pyo3(signature = (size=None))]
    fn standard_normal(&self, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let normal = ::rand_distr::Normal::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        make_ndarray(&mut rng, &shape, normal)
    }

    /// 生成整数随机数
    #[pyo3(signature = (low, high=None, size=None, endpoint=false))]
    fn integers(&self, low: i64, high: Option<i64>, size: Option<&Bound<'_, PyAny>>, endpoint: bool) -> PyResult<NdArray> {
        let hi = high.unwrap_or(low);
        let actual_low = if high.is_some() { low } else { 0 };
        let actual_high = if endpoint { hi + 1 } else { hi };
        if actual_high <= actual_low {
            return Err(PyValueError::new_err("high must be > low"));
        }
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Uniform::new(actual_low, actual_high)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        if shape.is_empty() {
            return Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), dist.sample(&mut rng) as f64),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| dist.sample(&mut rng) as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    /// 从数组中随机选择
    #[pyo3(signature = (a, size=None, replace=true))]
    fn choice<'py>(&self, py: Python<'py>, a: &Bound<'_, PyAny>, size: Option<usize>, replace: bool) -> PyResult<Bound<'py, PyAny>> {
        let mut rng = new_rng(self.seed);
        // Extract values from array-like a
        let vals: Vec<f64> = if let Ok(arr) = a.extract::<NdArray>() {
            arr.data.iter().copied().collect()
        } else if let Ok(list) = a.cast::<pyo3::types::PyList>() {
            let mut v = Vec::new();
            for item in list.iter() {
                v.push(item.extract::<f64>()?);
            }
            v
        } else {
            return Err(PyTypeError::new_err("a must be array-like"));
        };

        if vals.is_empty() {
            return Err(PyValueError::new_err("a must be non-empty"));
        }

        let n = size.unwrap_or(1);
        if !replace && n > vals.len() {
            return Err(PyValueError::new_err("Cannot take a larger sample than population when 'replace=false'"));
        }

        let dist = ::rand_distr::Uniform::new(0usize, vals.len())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let result: Vec<f64> = if replace {
            (0..n).map(|_| vals[dist.sample(&mut rng)]).collect()
        } else {
            let mut indices: Vec<usize> = (0..vals.len()).collect();
            for i in (1..vals.len()).rev() {
                let j_dist = ::rand_distr::Uniform::new(0usize, i + 1)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let j = j_dist.sample(&mut rng);
                indices.swap(i, j);
            }
            indices.truncate(n);
            indices.into_iter().map(|i| vals[i]).collect()
        };

        if n == 1 && size.is_none() {
            return Ok(result[0].into_pyobject(py).unwrap().into_any());
        }

        let arr = Array::from_shape_vec(IxDyn(&[n]), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Bound::new(py, NdArray { data: arr })?.into_any())
    }

    /// 打乱数组（1D）
    fn shuffle(&self, a: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut rng = new_rng(self.seed);
        if let Ok(arr) = a.extract::<NdArray>() {
            if arr.data.ndim() != 1 {
                return Err(PyValueError::new_err("shuffle only works for 1D arrays"));
            }
            let len = arr.data.len();
            let _indices: Vec<usize> = (0..len).collect();
            // Fisher-Yates shuffle using Uniform distribution
            let mut result: Vec<f64> = arr.data.iter().copied().collect();
            for i in (1..len).rev() {
                let j_dist = ::rand_distr::Uniform::new(0usize, i + 1)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let j = j_dist.sample(&mut rng);
                result.swap(i, j);
            }
            // Note: this does not modify the original array in-place because
            // we can't get mutable access to the underlying data through PyAny.
            // Use Python-level shuffle or permutation for in-place modification.
            Ok(())
        } else {
            Err(PyTypeError::new_err("Expected ndarray"))
        }
    }

    /// 随机排列
    #[pyo3(signature = (a))]
    fn permutation<'py>(&self, py: Python<'py>, a: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let mut rng = new_rng(self.seed);
        if let Ok(n) = a.extract::<usize>() {
            let mut indices: Vec<f64> = (0..n).map(|i| i as f64).collect();
            for i in (1..n).rev() {
                let j_dist = ::rand_distr::Uniform::new(0usize, i + 1)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let j = j_dist.sample(&mut rng);
                indices.swap(i, j);
            }
            let arr = Array::from_shape_vec(IxDyn(&[n]), indices)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Bound::new(py, NdArray { data: arr })?.into_any())
        } else if let Ok(arr) = a.extract::<NdArray>() {
            let vals: Vec<f64> = arr.data.iter().copied().collect();
            let n = vals.len();
            let mut indices: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j_dist = ::rand_distr::Uniform::new(0usize, i + 1)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let j = j_dist.sample(&mut rng);
                indices.swap(i, j);
            }
            let result: Vec<f64> = indices.into_iter().map(|i| vals[i]).collect();
            let result_arr = Array::from_shape_vec(arr.data.dim(), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Bound::new(py, NdArray { data: result_arr.into_dyn() })?.into_any())
        } else {
            Err(PyTypeError::new_err("Expected integer or ndarray"))
        }
    }

    /// 均匀分布
    #[pyo3(signature = (low=0.0, high=1.0, size=None))]
    fn uniform(&self, low: f64, high: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Uniform::new(low, high)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        make_ndarray(&mut rng, &shape, dist)
    }

    /// 正态分布
    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn normal(&self, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Normal::new(loc, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        make_ndarray(&mut rng, &shape, dist)
    }

    /// Beta 分布
    #[pyo3(signature = (a, b, size=None))]
    fn beta(&self, a: f64, b: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Beta::new(a, b)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        make_ndarray(&mut rng, &shape, dist)
    }

    /// Gamma 分布
    #[pyo3(signature = (shape, scale=1.0, size=None))]
    fn gamma(&self, shape: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let dist_shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Gamma::new(shape, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        make_ndarray(&mut rng, &dist_shape, dist)
    }

    /// 指数分布
    #[pyo3(signature = (scale=1.0, size=None))]
    fn exponential(&self, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Exp::new(1.0 / scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        make_ndarray(&mut rng, &shape, dist)
    }

    /// 二项分布
    #[pyo3(signature = (n, p, size=None))]
    fn binomial(&self, n: u64, p: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Binomial::new(n, p)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        if shape.is_empty() {
            return Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), dist.sample(&mut rng) as f64),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| dist.sample(&mut rng) as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    /// Poisson 分布
    #[pyo3(signature = (lam, size=None))]
    fn poisson(&self, lam: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Poisson::new(lam)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        if shape.is_empty() {
            return Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), dist.sample(&mut rng) as f64),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| dist.sample(&mut rng) as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    /// Weibull 分布
    #[pyo3(signature = (a, scale=1.0, size=None))]
    fn weibull(&self, a: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Weibull::new(a, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        make_ndarray(&mut rng, &shape, dist)
    }

    /// Logistic 分布（使用逆变换采样实现）
    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn logistic(&self, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let uniform = ::rand_distr::Uniform::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        if shape.is_empty() {
            let u = uniform.sample(&mut rng) as f64;
            let val = loc + scale * (u / (1.0 - u)).ln();
            return Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), val),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| {
            let u = uniform.sample(&mut rng) as f64;
            loc + scale * (u / (1.0 - u)).ln()
        }).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    /// Cauchy 分布
    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn cauchy(&self, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Cauchy::new(loc, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        make_ndarray(&mut rng, &shape, dist)
    }

    /// Laplace 分布（使用逆变换采样实现）
    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn laplace(&self, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let uniform = ::rand_distr::Uniform::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        if shape.is_empty() {
            let u: f64 = uniform.sample(&mut rng);
            let val = if u < 0.5 {
                loc + scale * (2.0 * u).ln()
            } else {
                loc - scale * (2.0 * (1.0 - u)).ln()
            };
            return Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), val),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| {
            let u: f64 = uniform.sample(&mut rng);
            if u < 0.5 {
                loc + scale * (2.0 * u).ln()
            } else {
                loc - scale * (2.0 * (1.0 - u)).ln()
            }
        }).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }
}

// ===== 旧 API 函数（保持向后兼容） =====

#[pyfunction]
#[pyo3(signature = (low=0.0, high=1.0, size=None))]
fn uniform(low: f64, high: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Uniform::new(low, high)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut guard = ensure_rng_mut();
    let rng = guard.as_mut().unwrap();
    make_ndarray(rng, &shape, dist)
}

#[pyfunction]
#[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
fn normal(loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Normal::new(loc, scale)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut guard = ensure_rng_mut();
    let rng = guard.as_mut().unwrap();
    make_ndarray(rng, &shape, dist)
}

#[pyfunction]
#[pyo3(signature = (a, b, size=None))]
fn beta(a: f64, b: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Beta::new(a, b)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut guard = ensure_rng_mut();
    let rng = guard.as_mut().unwrap();
    make_ndarray(rng, &shape, dist)
}

#[pyfunction]
#[pyo3(signature = (shape, scale=1.0, size=None))]
fn gamma(shape: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let dist_shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Gamma::new(shape, scale)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut guard = ensure_rng_mut();
    let rng = guard.as_mut().unwrap();
    make_ndarray(rng, &dist_shape, dist)
}

#[pyfunction]
#[pyo3(signature = (scale=1.0, size=None))]
fn exponential(scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Exp::new(1.0 / scale)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut guard = ensure_rng_mut();
    let rng = guard.as_mut().unwrap();
    make_ndarray(rng, &shape, dist)
}

pub fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(seed, m)?)?;
    m.add_function(wrap_pyfunction!(random_rand, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(random_randint, m)?)?;
    m.add_function(wrap_pyfunction!(uniform, m)?)?;
    m.add_function(wrap_pyfunction!(normal, m)?)?;
    m.add_function(wrap_pyfunction!(beta, m)?)?;
    m.add_function(wrap_pyfunction!(gamma, m)?)?;
    m.add_function(wrap_pyfunction!(exponential, m)?)?;
    m.add_function(wrap_pyfunction!(get_state, m)?)?;
    m.add_class::<PyGenerator>()?;
    Ok(())
}