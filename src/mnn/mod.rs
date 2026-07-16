//! MNN Inference Engine FFI Binding Layer
//!
//! This module encapsulates the low-level interfaces of the MNN C++ inference framework, providing safe Rust APIs.

// Use stub implementation when building on docs.rs
#[cfg(feature = "docsrs")]
mod docsrs_stub;

#[cfg(feature = "docsrs")]
pub use docsrs_stub::*;

// Use complete implementation for normal builds
#[cfg(not(feature = "docsrs"))]
mod normal_impl {

    use ndarray::{ArrayD, ArrayViewD, IxDyn};
    use std::ffi::CStr;
    use std::ptr::NonNull;

    #[allow(non_camel_case_types)]
    #[allow(non_upper_case_globals)]
    #[allow(non_snake_case)]
    #[allow(dead_code)]
    mod ffi {
        include!(concat!(env!("OUT_DIR"), "/mnn_bindings.rs"));
    }

    // ============== Error Types ==============

    /// MNN related errors
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum MnnError {
        /// Invalid parameter
        InvalidParameter(String),
        /// Out of memory
        OutOfMemory,
        /// Runtime error
        RuntimeError(String),
        /// Unsupported operation
        Unsupported,
        /// Model loading failed
        ModelLoadFailed(String),
        /// Requested inference backend is not available in the linked MNN build
        BackendUnavailable(String),
        /// Null pointer error
        NullPointer,
        /// Shape mismatch
        ShapeMismatch {
            expected: Vec<usize>,
            got: Vec<usize>,
        },
    }

    impl std::fmt::Display for MnnError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                MnnError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
                MnnError::OutOfMemory => write!(f, "Out of memory"),
                MnnError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
                MnnError::Unsupported => write!(f, "Unsupported operation"),
                MnnError::ModelLoadFailed(msg) => write!(f, "Model loading failed: {}", msg),
                MnnError::BackendUnavailable(backend) => {
                    write!(f, "Backend unavailable: {}", backend)
                }
                MnnError::NullPointer => write!(f, "Null pointer"),
                MnnError::ShapeMismatch { expected, got } => {
                    write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
                }
            }
        }
    }

    impl std::error::Error for MnnError {}

    pub type Result<T> = std::result::Result<T, MnnError>;

    // ============== Configuration Types ==============

    /// Precision mode
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[repr(i32)]
    pub enum PrecisionMode {
        /// Normal precision
        #[default]
        Normal = 0,
        /// Low precision (faster)
        Low = 1,
        /// High precision (more accurate)
        High = 2,
    }

    /// Data format
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[repr(i32)]
    pub enum DataFormat {
        /// NCHW format (Caffe/PyTorch/ONNX)
        #[default]
        NCHW = 0,
        /// NHWC format (TensorFlow)
        NHWC = 1,
        /// Auto detect
        Auto = 2,
    }

    /// Inference backend type
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum Backend {
        /// CPU backend
        #[default]
        CPU,
        /// Metal GPU (macOS/iOS)
        Metal,
        /// OpenCL GPU
        OpenCL,
        /// OpenGL GPU
        OpenGL,
        /// Vulkan GPU
        Vulkan,
        /// CUDA GPU (NVIDIA)
        CUDA,
        /// CoreML (macOS/iOS)
        CoreML,
    }

    impl Backend {
        /// Stable backend name used in errors, logs, and command-line options.
        pub const fn as_str(self) -> &'static str {
            match self {
                Backend::CPU => "cpu",
                Backend::Metal => "metal",
                Backend::OpenCL => "opencl",
                Backend::OpenGL => "opengl",
                Backend::Vulkan => "vulkan",
                Backend::CUDA => "cuda",
                Backend::CoreML => "coreml",
            }
        }

        /// Return whether the linked MNN library registered this backend.
        pub fn is_available(self) -> bool {
            unsafe { ffi::mnnr_is_backend_available(self.to_forward_type()) }
        }

        fn ensure_available(self) -> Result<()> {
            if self.is_available() {
                Ok(())
            } else {
                Err(MnnError::BackendUnavailable(self.as_str().to_string()))
            }
        }

        /// Convert to MNNForwardType integer value
        fn to_forward_type(self) -> i32 {
            match self {
                Backend::CPU => 0,    // MNN_FORWARD_CPU
                Backend::Metal => 1,  // MNN_FORWARD_METAL
                Backend::CUDA => 2,   // MNN_FORWARD_CUDA
                Backend::OpenCL => 3, // MNN_FORWARD_OPENCL
                Backend::CoreML => 5, // MNN_FORWARD_NN
                Backend::OpenGL => 6, // MNN_FORWARD_OPENGL
                Backend::Vulkan => 7, // MNN_FORWARD_VULKAN
            }
        }
    }

    /// Inference configuration
    #[derive(Debug, Clone)]
    pub struct InferenceConfig {
        /// Thread count (0 means auto, default is 4)
        pub thread_count: i32,
        /// Precision mode
        pub precision_mode: PrecisionMode,
        /// Whether to use cache
        pub use_cache: bool,
        /// Data format
        pub data_format: DataFormat,
        /// Inference backend
        pub backend: Backend,
    }

    impl Default for InferenceConfig {
        fn default() -> Self {
            InferenceConfig {
                thread_count: 4,
                precision_mode: PrecisionMode::Normal,
                use_cache: false,
                data_format: DataFormat::NCHW,
                backend: Backend::CPU,
            }
        }
    }

    impl InferenceConfig {
        /// Create new inference configuration
        pub fn new() -> Self {
            Self::default()
        }

        /// Set thread count
        pub fn with_threads(mut self, threads: i32) -> Self {
            self.thread_count = threads;
            self
        }

        /// Set precision mode
        pub fn with_precision(mut self, precision: PrecisionMode) -> Self {
            self.precision_mode = precision;
            self
        }

        /// Set backend
        pub fn with_backend(mut self, backend: Backend) -> Self {
            self.backend = backend;
            self
        }

        /// Set data format
        pub fn with_data_format(mut self, format: DataFormat) -> Self {
            self.data_format = format;
            self
        }

        fn to_ffi(&self) -> ffi::MNNR_Config {
            ffi::MNNR_Config {
                thread_count: self.thread_count,
                precision_mode: self.precision_mode as i32,
                use_cache: self.use_cache,
                data_format: self.data_format as i32,
                forward_type: self.backend.to_forward_type(),
            }
        }
    }

    // ============== Shared Runtime ==============

    /// Shared runtime for sharing resources among multiple engines
    pub struct SharedRuntime {
        ptr: NonNull<ffi::MNN_SharedRuntime>,
    }

    impl SharedRuntime {
        /// Create new shared runtime
        pub fn new(config: &InferenceConfig) -> Result<Self> {
            config.backend.ensure_available()?;
            let c_config = config.to_ffi();
            let runtime_ptr = unsafe { ffi::mnnr_create_runtime(&c_config) };

            let ptr = NonNull::new(runtime_ptr).ok_or_else(|| {
                MnnError::RuntimeError("Create shared runtime failed".to_string())
            })?;

            Ok(SharedRuntime { ptr })
        }

        pub(crate) fn as_ptr(&self) -> *mut ffi::MNN_SharedRuntime {
            self.ptr.as_ptr()
        }
    }

    impl Drop for SharedRuntime {
        fn drop(&mut self) {
            unsafe {
                ffi::mnnr_destroy_runtime(self.ptr.as_ptr());
            }
        }
    }

    unsafe impl Send for SharedRuntime {}
    unsafe impl Sync for SharedRuntime {}

    // ============== Helper Functions ==============

    fn get_last_error_message(engine: Option<*const ffi::MNN_InferenceEngine>) -> String {
        match engine {
            Some(ptr) => unsafe {
                let c_str = ffi::mnnr_get_last_error(ptr);
                if c_str.is_null() {
                    "Unknown error".to_string()
                } else {
                    CStr::from_ptr(c_str).to_string_lossy().into_owned()
                }
            },
            None => "Engine creation failed".to_string(),
        }
    }

    // ============== Inference Engine ==============

    /// MNN inference engine
    ///
    /// Encapsulates MNN model loading and inference functionality
    pub struct InferenceEngine {
        ptr: NonNull<ffi::MNN_InferenceEngine>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    }

    impl InferenceEngine {
        /// Create inference engine from model byte data
        ///
        /// # Parameters
        /// - `model_buffer`: Model file byte data
        /// - `config`: Optional inference configuration
        ///
        /// # Example
        /// ```ignore
        /// let model_data = std::fs::read("model.mnn")?;
        /// let engine = InferenceEngine::from_buffer(&model_data, None)?;
        /// ```
        pub fn from_buffer(model_buffer: &[u8], config: Option<InferenceConfig>) -> Result<Self> {
            if model_buffer.is_empty() {
                return Err(MnnError::InvalidParameter(
                    "Model data is empty".to_string(),
                ));
            }

            let cfg = config.unwrap_or_default();
            cfg.backend.ensure_available()?;
            let c_config = cfg.to_ffi();

            let engine_ptr = unsafe {
                ffi::mnnr_create_engine(
                    model_buffer.as_ptr() as *const _,
                    model_buffer.len(),
                    &c_config,
                )
            };

            let ptr = NonNull::new(engine_ptr)
                .ok_or_else(|| MnnError::ModelLoadFailed(get_last_error_message(None)))?;

            let (input_shape, output_shape) = unsafe { Self::get_shapes(ptr.as_ptr())? };

            Ok(InferenceEngine {
                ptr,
                input_shape,
                output_shape,
            })
        }

        /// Create inference engine from model file
        pub fn from_file(
            model_path: impl AsRef<std::path::Path>,
            config: Option<InferenceConfig>,
        ) -> Result<Self> {
            let model_buffer = std::fs::read(model_path.as_ref()).map_err(|e| {
                MnnError::ModelLoadFailed(format!("Failed to read model file: {}", e))
            })?;
            Self::from_buffer(&model_buffer, config)
        }

        /// Create inference engine from model byte data using shared runtime
        pub fn from_buffer_with_runtime(
            model_buffer: &[u8],
            runtime: &SharedRuntime,
        ) -> Result<Self> {
            if model_buffer.is_empty() {
                return Err(MnnError::InvalidParameter(
                    "Model data is empty".to_string(),
                ));
            }

            let engine_ptr = unsafe {
                ffi::mnnr_create_engine_with_runtime(
                    model_buffer.as_ptr() as *const _,
                    model_buffer.len(),
                    runtime.as_ptr(),
                )
            };

            let ptr = NonNull::new(engine_ptr)
                .ok_or_else(|| MnnError::ModelLoadFailed(get_last_error_message(None)))?;

            let (input_shape, output_shape) = unsafe { Self::get_shapes(ptr.as_ptr())? };

            Ok(InferenceEngine {
                ptr,
                input_shape,
                output_shape,
            })
        }

        unsafe fn get_shapes(
            ptr: *mut ffi::MNN_InferenceEngine,
        ) -> Result<(Vec<usize>, Vec<usize>)> {
            let mut input_shape_vec = vec![0usize; 8];
            let mut input_ndims = 0;
            let mut output_shape_vec = vec![0usize; 8];
            let mut output_ndims = 0;

            if ffi::mnnr_get_input_shape(ptr, input_shape_vec.as_mut_ptr(), &mut input_ndims)
                != ffi::MNNR_ErrorCode_MNNR_SUCCESS
            {
                return Err(MnnError::RuntimeError(
                    "Failed to get input shape".to_string(),
                ));
            }
            input_shape_vec.truncate(input_ndims);

            if ffi::mnnr_get_output_shape(ptr, output_shape_vec.as_mut_ptr(), &mut output_ndims)
                != ffi::MNNR_ErrorCode_MNNR_SUCCESS
            {
                return Err(MnnError::RuntimeError(
                    "Failed to get output shape".to_string(),
                ));
            }
            output_shape_vec.truncate(output_ndims);

            Ok((input_shape_vec, output_shape_vec))
        }

        /// Get input tensor shape
        pub fn input_shape(&self) -> &[usize] {
            &self.input_shape
        }

        /// Get output tensor shape
        pub fn output_shape(&self) -> &[usize] {
            &self.output_shape
        }

        /// Execute inference
        ///
        /// # Parameters
        /// - `input_data`: Input data, shape must match model input shape
        ///
        /// # Returns
        /// Inference result array
        pub fn run(&self, input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
            if input_data.shape() != self.input_shape.as_slice() {
                return Err(MnnError::ShapeMismatch {
                    expected: self.input_shape.clone(),
                    got: input_data.shape().to_vec(),
                });
            }

            let input_slice = input_data.as_slice().ok_or_else(|| {
                MnnError::InvalidParameter("Input data must be contiguous".to_string())
            })?;

            let output_size: usize = self.output_shape.iter().product();
            let mut output_buffer = vec![0.0f32; output_size];

            let error_code = unsafe {
                ffi::mnnr_run_inference(
                    self.ptr.as_ptr(),
                    input_slice.as_ptr(),
                    input_slice.len(),
                    output_buffer.as_mut_ptr(),
                    output_buffer.len(),
                )
            };

            match error_code {
                ffi::MNNR_ErrorCode_MNNR_SUCCESS => {
                    ArrayD::from_shape_vec(IxDyn(&self.output_shape), output_buffer).map_err(|e| {
                        MnnError::RuntimeError(format!("Failed to create output array: {}", e))
                    })
                }
                ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => Err(
                    MnnError::InvalidParameter(get_last_error_message(Some(self.ptr.as_ptr()))),
                ),
                ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
                ffi::MNNR_ErrorCode_MNNR_ERROR_UNSUPPORTED => Err(MnnError::Unsupported),
                _ => Err(MnnError::RuntimeError(get_last_error_message(Some(
                    self.ptr.as_ptr(),
                )))),
            }
        }

        /// Execute inference (using raw slices)
        ///
        /// This is a low-level API, suitable for scenarios requiring maximum performance
        pub fn run_raw(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
            let expected_input: usize = self.input_shape.iter().product();
            let expected_output: usize = self.output_shape.iter().product();

            if input.len() != expected_input {
                return Err(MnnError::ShapeMismatch {
                    expected: vec![expected_input],
                    got: vec![input.len()],
                });
            }

            if output.len() != expected_output {
                return Err(MnnError::ShapeMismatch {
                    expected: vec![expected_output],
                    got: vec![output.len()],
                });
            }

            let error_code = unsafe {
                ffi::mnnr_run_inference(
                    self.ptr.as_ptr(),
                    input.as_ptr(),
                    input.len(),
                    output.as_mut_ptr(),
                    output.len(),
                )
            };

            match error_code {
                ffi::MNNR_ErrorCode_MNNR_SUCCESS => Ok(()),
                ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => Err(
                    MnnError::InvalidParameter(get_last_error_message(Some(self.ptr.as_ptr()))),
                ),
                ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
                _ => Err(MnnError::RuntimeError(get_last_error_message(Some(
                    self.ptr.as_ptr(),
                )))),
            }
        }

        pub(crate) fn as_ptr(&self) -> NonNull<ffi::MNN_InferenceEngine> {
            self.ptr
        }

        /// Check if model has dynamic shape (contains -1 dimension)
        pub fn has_dynamic_shape(&self) -> bool {
            // When shape contains very large values, it indicates dynamic shape (-1 converted to usize becomes very large)
            self.input_shape.iter().any(|&d| d > 100000)
                || self.output_shape.iter().any(|&d| d > 100000)
        }

        /// Execute dynamic shape inference
        ///
        /// Suitable for models where input shape changes at runtime (such as detection models).
        /// This function adjusts model input tensor shape before running.
        ///
        /// # Parameters
        /// - `input_data`: Input data array
        ///
        /// # Returns
        /// Inference result array, shape dynamically determined by model
        pub fn run_dynamic(&self, input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
            let input_shape: Vec<usize> = input_data.shape().to_vec();
            let input_slice = input_data.as_slice().ok_or_else(|| {
                MnnError::InvalidParameter("Input data must be contiguous".to_string())
            })?;

            let mut output_data: *mut f32 = std::ptr::null_mut();
            let mut output_size: usize = 0;
            let mut output_dims = [0usize; 8];
            let mut output_ndims: usize = 0;

            let error_code = unsafe {
                ffi::mnnr_run_inference_dynamic(
                    self.ptr.as_ptr(),
                    input_slice.as_ptr(),
                    input_shape.as_ptr(),
                    input_shape.len(),
                    &mut output_data,
                    &mut output_size,
                    output_dims.as_mut_ptr(),
                    &mut output_ndims,
                )
            };

            if error_code != ffi::MNNR_ErrorCode_MNNR_SUCCESS {
                return match error_code {
                    ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => Err(
                        MnnError::InvalidParameter(get_last_error_message(Some(self.ptr.as_ptr()))),
                    ),
                    ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
                    ffi::MNNR_ErrorCode_MNNR_ERROR_UNSUPPORTED => Err(MnnError::Unsupported),
                    _ => Err(MnnError::RuntimeError(get_last_error_message(Some(
                        self.ptr.as_ptr(),
                    )))),
                };
            }

            // Copy output data and free C buffer
            let output_shape: Vec<usize> = output_dims[..output_ndims].to_vec();
            let output_buffer = unsafe {
                let slice = std::slice::from_raw_parts(output_data, output_size);
                let buffer = slice.to_vec();
                ffi::mnnr_free_output(output_data);
                buffer
            };

            ArrayD::from_shape_vec(IxDyn(&output_shape), output_buffer).map_err(|e| {
                MnnError::RuntimeError(format!("Failed to create output array: {}", e))
            })
        }

        /// Execute dynamic shape inference (using raw slices)
        ///
        /// Low-level API, caller is responsible for managing output buffer
        pub fn run_dynamic_raw(
            &self,
            input: &[f32],
            input_shape: &[usize],
        ) -> Result<(Vec<f32>, Vec<usize>)> {
            let mut output_data: *mut f32 = std::ptr::null_mut();
            let mut output_size: usize = 0;
            let mut output_dims = [0usize; 8];
            let mut output_ndims: usize = 0;

            let error_code = unsafe {
                ffi::mnnr_run_inference_dynamic(
                    self.ptr.as_ptr(),
                    input.as_ptr(),
                    input_shape.as_ptr(),
                    input_shape.len(),
                    &mut output_data,
                    &mut output_size,
                    output_dims.as_mut_ptr(),
                    &mut output_ndims,
                )
            };

            if error_code != ffi::MNNR_ErrorCode_MNNR_SUCCESS {
                return match error_code {
                    ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => Err(
                        MnnError::InvalidParameter(get_last_error_message(Some(self.ptr.as_ptr()))),
                    ),
                    ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
                    _ => Err(MnnError::RuntimeError(get_last_error_message(Some(
                        self.ptr.as_ptr(),
                    )))),
                };
            }

            // Copy output and free C buffer
            let output_shape = output_dims[..output_ndims].to_vec();
            let output_buffer = unsafe {
                let slice = std::slice::from_raw_parts(output_data, output_size);
                let buffer = slice.to_vec();
                ffi::mnnr_free_output(output_data);
                buffer
            };

            Ok((output_buffer, output_shape))
        }
    }

    impl Drop for InferenceEngine {
        fn drop(&mut self) {
            unsafe {
                ffi::mnnr_destroy_engine(self.ptr.as_ptr());
            }
        }
    }

    unsafe impl Send for InferenceEngine {}
    unsafe impl Sync for InferenceEngine {}

    // ============== Session Pool ==============

    /// Session pool for high-concurrency inference scenarios
    pub struct SessionPool {
        ptr: NonNull<ffi::MNN_SessionPool>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    }

    impl SessionPool {
        /// Create session pool
        ///
        /// # Parameters
        /// - `engine`: Inference engine
        /// - `pool_size`: Number of sessions in pool
        /// - `config`: Optional inference configuration
        pub fn new(
            engine: &InferenceEngine,
            pool_size: usize,
            config: Option<InferenceConfig>,
        ) -> Result<Self> {
            if pool_size == 0 {
                return Err(MnnError::InvalidParameter(
                    "Pool size cannot be 0".to_string(),
                ));
            }

            let cfg = config.unwrap_or_default();
            cfg.backend.ensure_available()?;
            let c_config = cfg.to_ffi();

            let pool_ptr = unsafe {
                ffi::mnnr_create_session_pool(engine.as_ptr().as_ptr(), pool_size, &c_config)
            };

            let ptr = NonNull::new(pool_ptr)
                .ok_or_else(|| MnnError::RuntimeError("Create session pool failed".to_string()))?;

            Ok(SessionPool {
                ptr,
                input_shape: engine.input_shape.clone(),
                output_shape: engine.output_shape.clone(),
            })
        }

        /// Execute inference (thread-safe)
        pub fn run(&self, input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
            if input_data.shape() != self.input_shape.as_slice() {
                return Err(MnnError::ShapeMismatch {
                    expected: self.input_shape.clone(),
                    got: input_data.shape().to_vec(),
                });
            }

            let input_slice = input_data.as_slice().ok_or_else(|| {
                MnnError::InvalidParameter("Input data must be contiguous".to_string())
            })?;

            let output_size: usize = self.output_shape.iter().product();
            let mut output_buffer = vec![0.0f32; output_size];

            let error_code = unsafe {
                ffi::mnnr_session_pool_run(
                    self.ptr.as_ptr(),
                    input_slice.as_ptr(),
                    input_slice.len(),
                    output_buffer.as_mut_ptr(),
                    output_buffer.len(),
                )
            };

            match error_code {
                ffi::MNNR_ErrorCode_MNNR_SUCCESS => {
                    ArrayD::from_shape_vec(IxDyn(&self.output_shape), output_buffer).map_err(|e| {
                        MnnError::RuntimeError(format!("Failed to create output array: {}", e))
                    })
                }
                _ => Err(MnnError::RuntimeError(
                    "Session pool inference failed".to_string(),
                )),
            }
        }

        /// Get available session count
        pub fn available(&self) -> usize {
            unsafe { ffi::mnnr_session_pool_available(self.ptr.as_ptr()) }
        }
    }

    impl Drop for SessionPool {
        fn drop(&mut self) {
            unsafe {
                ffi::mnnr_destroy_session_pool(self.ptr.as_ptr());
            }
        }
    }

    unsafe impl Send for SessionPool {}
    unsafe impl Sync for SessionPool {}

    // ============== Utility Functions ==============

    /// Get MNN version number
    pub fn get_version() -> String {
        unsafe {
            let c_str = ffi::mnnr_get_version();
            if c_str.is_null() {
                "unknown".to_string()
            } else {
                CStr::from_ptr(c_str).to_string_lossy().into_owned()
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_config_default() {
            let config = InferenceConfig::default();
            assert_eq!(config.thread_count, 4);
            assert_eq!(config.precision_mode, PrecisionMode::Normal);
        }

        #[test]
        fn test_config_builder() {
            let config = InferenceConfig::new()
                .with_threads(8)
                .with_precision(PrecisionMode::High)
                .with_backend(Backend::Metal);

            assert_eq!(config.thread_count, 8);
            assert_eq!(config.precision_mode, PrecisionMode::High);
            assert_eq!(config.backend, Backend::Metal);
        }

        #[test]
        fn cpu_backend_is_always_available() {
            assert!(Backend::CPU.is_available());
        }

        #[test]
        fn backend_unavailable_error_names_the_requested_backend() {
            let backend = [
                Backend::CUDA,
                Backend::Vulkan,
                Backend::OpenCL,
                Backend::OpenGL,
                Backend::CoreML,
                Backend::Metal,
            ]
            .into_iter()
            .find(|backend| !backend.is_available())
            .expect("at least one optional backend should be unavailable in a platform build");

            let error = backend.ensure_available().unwrap_err();
            assert_eq!(
                error.to_string(),
                format!("Backend unavailable: {}", backend.as_str())
            );
        }

        #[test]
        fn engine_rejects_an_unavailable_backend_before_loading_the_model() {
            let backend = [
                Backend::CUDA,
                Backend::Vulkan,
                Backend::OpenCL,
                Backend::OpenGL,
                Backend::CoreML,
                Backend::Metal,
            ]
            .into_iter()
            .find(|backend| !backend.is_available())
            .expect("at least one optional backend should be unavailable in a platform build");
            let config = InferenceConfig::new().with_backend(backend);

            let result = InferenceEngine::from_buffer(&[0], Some(config));
            assert!(matches!(
                result,
                Err(MnnError::BackendUnavailable(name)) if name == backend.as_str()
            ));
        }
    }
} // end of normal_impl module

// Re-export types from normal implementation
#[cfg(not(feature = "docsrs"))]
pub use normal_impl::*;
