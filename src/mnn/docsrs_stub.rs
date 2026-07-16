//! docsrs stub module - for documentation generation
//!
//! This module is used during docs.rs build, providing type definitions without actual implementations

use ndarray::{ArrayD, ArrayViewD};
use std::path::Path;

// ============== Error Types ==============

/// MNN-related errors
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
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for MnnError {}

/// MNN Result type
pub type Result<T> = std::result::Result<T, MnnError>;

// ============== Backend Types ==============

/// Computation backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// CPU backend (default)
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
    /// CUDA GPU
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

    /// Runtime availability cannot be inspected while generating docs.
    pub const fn is_available(self) -> bool {
        matches!(self, Backend::CPU)
    }
}

/// Precision mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrecisionMode {
    /// Normal precision
    #[default]
    Normal,
    /// Low precision
    Low,
    /// High precision
    High,
    /// Low memory usage
    LowMemory,
}

/// Data format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataFormat {
    /// NCHW format
    #[default]
    NCHW,
    /// NHWC format
    NHWC,
}

// ============== Configuration Types ==============

/// Inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub thread_count: i32,
    pub precision_mode: PrecisionMode,
    pub backend: Backend,
    pub use_cache: bool,
    pub data_format: DataFormat,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            thread_count: 4,
            precision_mode: PrecisionMode::Normal,
            backend: Backend::CPU,
            use_cache: true,
            data_format: DataFormat::NCHW,
        }
    }
}

impl InferenceConfig {
    /// Create a new inference configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of threads
    pub fn with_threads(mut self, threads: i32) -> Self {
        self.thread_count = threads;
        self
    }

    /// Set the precision mode
    pub fn with_precision(mut self, precision: PrecisionMode) -> Self {
        self.precision_mode = precision;
        self
    }

    /// Set the backend
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the data format
    pub fn with_data_format(mut self, format: DataFormat) -> Self {
        self.data_format = format;
        self
    }
}

// ============== Shared Runtime ==============

/// Shared runtime for sharing resources between multiple engines
pub struct SharedRuntime {
    _private: (),
}

impl SharedRuntime {
    /// Create a new shared runtime
    pub fn new(_config: &InferenceConfig) -> Result<Self> {
        unimplemented!(
            "This feature is only available at runtime, not available during documentation build"
        )
    }
}

// ============== Inference Engine ==============

/// MNN inference engine
pub struct InferenceEngine {
    _input_shape: Vec<usize>,
    _output_shape: Vec<usize>,
}

impl InferenceEngine {
    /// Create inference engine from file
    pub fn from_file(
        _model_path: impl AsRef<Path>,
        _config: Option<InferenceConfig>,
    ) -> Result<Self> {
        unimplemented!(
            "This feature is only available at runtime, not available during documentation build"
        )
    }

    /// Create inference engine from memory
    pub fn from_buffer(_data: &[u8], _config: Option<InferenceConfig>) -> Result<Self> {
        unimplemented!(
            "This feature is only available at runtime, not available during documentation build"
        )
    }

    /// Create inference engine from model bytes using shared runtime
    pub fn from_buffer_with_runtime(
        _model_buffer: &[u8],
        _runtime: &SharedRuntime,
    ) -> Result<Self> {
        unimplemented!(
            "This feature is only available at runtime, not available during documentation build"
        )
    }

    /// Get input shape
    pub fn input_shape(&self) -> &[usize] {
        &self._input_shape
    }

    /// Get output shape
    pub fn output_shape(&self) -> &[usize] {
        &self._output_shape
    }

    /// Perform inference
    pub fn infer(&self, _input: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        unimplemented!()
    }

    /// Perform inference (variable input shape)
    pub fn infer_dynamic(&self, _input: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        unimplemented!()
    }

    /// Perform inference (variable input shape) - alias
    pub fn run_dynamic(&self, _input: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        unimplemented!()
    }

    /// Perform inference (raw interface)
    pub fn run_dynamic_raw(
        &self,
        _input_data: &[f32],
        _input_shape: &[usize],
        _output_data: &mut [f32],
    ) -> Result<Vec<usize>> {
        unimplemented!()
    }
}

// ============== Helper Functions ==============

/// Get MNN version
pub fn get_version() -> String {
    "unknown (docs.rs build)".to_string()
}
