#ifndef MNN_WRAPPER_H
#define MNN_WRAPPER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Opaque handles
    typedef struct MNN_InferenceEngine MNN_InferenceEngine;
    typedef struct MNN_SessionPool MNN_SessionPool;
    typedef struct MNN_SharedRuntime MNN_SharedRuntime;

    // Error codes (compatible with original medivh-mnn)
    typedef enum
    {
        MNNR_SUCCESS = 0,
        MNNR_ERROR_INVALID_PARAMETER = 1,
        MNNR_ERROR_OUT_OF_MEMORY = 2,
        MNNR_ERROR_RUNTIME_ERROR = 3,
        MNNR_ERROR_UNSUPPORTED = 4,
        MNNR_ERROR_MODEL_LOAD_FAILED = 5
    } MNNR_ErrorCode;

    // Data format for input/output tensors
    typedef enum
    {
        MNNR_DATA_FORMAT_NCHW = 0, // Caffe/PyTorch/ONNX: [batch, channels, height, width]
        MNNR_DATA_FORMAT_NHWC = 1, // TensorFlow: [batch, height, width, channels]
        MNNR_DATA_FORMAT_AUTO = 2  // Auto-detect from model
    } MNNR_DataFormat;

    // Configuration for inference engine
    typedef struct
    {
        int32_t thread_count;   // Number of threads (0 for auto, -1 to use MNN default thread pool)
        int32_t precision_mode; // 0=Normal, 1=Low(faster), 2=High(accurate)
        bool use_cache;         // Whether to use cache file
        int32_t data_format;    // Input/Output data format
        int32_t forward_type;   // MNNForwardType: 0=CPU, 1=Metal, 2=CUDA, 3=OpenCL, 6=OpenGL, 7=Vulkan, 5=CoreML/NNAPI
    } MNNR_Config;

    // ============== Version & Info ==============

    // Get MNN version string
    const char *mnnr_get_version(void);

    // Check whether the requested MNN backend is registered.
    bool mnnr_is_backend_available(int32_t forward_type);

    // ============== Shared Runtime API ==============

    // Create a shared runtime for resource sharing across engines
    // Returns NULL on failure
    MNN_SharedRuntime *mnnr_create_runtime(const MNNR_Config *config);

    // Destroy a shared runtime
    // Warning: All engines using this runtime must be destroyed first
    void mnnr_destroy_runtime(MNN_SharedRuntime *runtime);

    // ============== Inference Engine API ==============

    // Create an inference engine from model buffer
    // Returns NULL on failure
    MNN_InferenceEngine *mnnr_create_engine(
        const void *buffer,
        size_t size,
        const MNNR_Config *config);

    // Create an inference engine using a shared runtime
    // This allows multiple engines to share thread pool and memory pool
    MNN_InferenceEngine *mnnr_create_engine_with_runtime(
        const void *buffer,
        size_t size,
        MNN_SharedRuntime *runtime);

    // Destroy an inference engine
    void mnnr_destroy_engine(MNN_InferenceEngine *engine);

    // Get input tensor shape
    // dims: output array (must have space for at least 8 elements)
    // out_ndims: output for number of dimensions
    MNNR_ErrorCode mnnr_get_input_shape(
        const MNN_InferenceEngine *engine,
        size_t *dims,
        size_t *out_ndims);

    // Get output tensor shape
    MNNR_ErrorCode mnnr_get_output_shape(
        const MNN_InferenceEngine *engine,
        size_t *dims,
        size_t *out_ndims);

    // Run single inference (thread-safe but serialized)
    // This uses the default session and is suitable for simple use cases
    MNNR_ErrorCode mnnr_run_inference(
        MNN_InferenceEngine *engine,
        const float *input_data,
        size_t input_size,
        float *output_data,
        size_t output_size);

    // Get last error message
    const char *mnnr_get_last_error(const MNN_InferenceEngine *engine);

    // ============== Session Pool API (Recommended for Production) ==============

    // Create a session pool with multiple sessions for concurrent inference
    // pool_size: number of sessions (also max concurrent inferences)
    // Uses MNN's internal thread pool for optimal performance
    MNN_SessionPool *mnnr_create_session_pool(
        MNN_InferenceEngine *engine,
        size_t pool_size,
        const MNNR_Config *config);

    // Destroy a session pool
    void mnnr_destroy_session_pool(MNN_SessionPool *pool);

    // Run inference using the session pool (blocking, thread-safe)
    // Automatically queues if all sessions are busy
    MNNR_ErrorCode mnnr_session_pool_run(
        MNN_SessionPool *pool,
        const float *input_data,
        size_t input_size,
        float *output_data,
        size_t output_size);

    // Get number of available (idle) sessions
    size_t mnnr_session_pool_available(const MNN_SessionPool *pool);

    // Get last error message from session pool
    const char *mnnr_session_pool_get_last_error(const MNN_SessionPool *pool);

    // ============== Single Session API ==============

    typedef struct MNN_SingleSession MNN_SingleSession;

    // Create a single session for manual management
    MNN_SingleSession *mnnr_create_session(
        MNN_InferenceEngine *engine,
        const MNNR_Config *config);

    // Destroy a single session
    void mnnr_destroy_session(MNN_SingleSession *session);

    // Run inference with a specific session (NOT thread-safe for same session)
    MNNR_ErrorCode mnnr_run_inference_with_session(
        MNN_SingleSession *session,
        const float *input_data,
        size_t input_size,
        float *output_data,
        size_t output_size);

    // Get last error message from session
    const char *mnnr_session_get_last_error(const MNN_SingleSession *session);

    // ============== Dynamic Shape API ==============

    // Run inference with dynamic input shape
    // input_dims: array of input dimensions
    // input_ndims: number of input dimensions
    // output_dims: output array for result dimensions (at least 8 elements)
    // output_ndims: output for number of result dimensions
    MNNR_ErrorCode mnnr_run_inference_dynamic(
        MNN_InferenceEngine *engine,
        const float *input_data,
        const size_t *input_dims,
        size_t input_ndims,
        float **output_data,
        size_t *output_size,
        size_t *output_dims,
        size_t *output_ndims);

    // Free output buffer allocated by mnnr_run_inference_dynamic
    void mnnr_free_output(float *output_data);

#ifdef __cplusplus
}
#endif

#endif // MNN_WRAPPER_H
