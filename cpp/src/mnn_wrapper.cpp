#include "mnn_wrapper.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>

#include <cstring>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <memory>

namespace MNN
{
    class RuntimeCreator;
    MNN_PUBLIC const RuntimeCreator *MNNGetExtraRuntimeCreator(MNNForwardType type);
}

// C++11 compatible make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique_ptr(Args &&...args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Global mutex to serialize MNN inference calls
// MNN's internal thread pool has a limit of MNN_THREAD_POOL_MAX_TASKS (default=2)
// This mutex ensures only one inference runs at a time, avoiding thread pool exhaustion
static std::mutex g_mnn_inference_mutex;

// ============== Internal Structures ==============

struct MNN_SharedRuntime
{
    MNN::BackendConfig backend_config;
    MNN::ScheduleConfig schedule_config;
    int thread_count;
    int precision_mode;
};

struct MNN_InferenceEngine
{
    std::unique_ptr<MNN::Interpreter> interpreter;
    MNN::Session *default_session;
    std::mutex mutex;
    std::string last_error;

    std::vector<int> input_shape;
    std::vector<int> output_shape;
    MNN::Tensor *input_tensor;
    MNN::Tensor *output_tensor;

    MNN_SharedRuntime *runtime; // Optional shared runtime
    bool owns_runtime;

    MNN_InferenceEngine() : default_session(nullptr), input_tensor(nullptr),
                            output_tensor(nullptr), runtime(nullptr), owns_runtime(false) {}
};

struct MNN_SingleSession
{
    MNN::Session *session;
    MNN_InferenceEngine *engine;
    std::string last_error;
    MNN::Tensor *input_tensor;
    MNN::Tensor *output_tensor;

    MNN_SingleSession() : session(nullptr), engine(nullptr),
                          input_tensor(nullptr), output_tensor(nullptr) {}
};

struct MNN_SessionPool
{
    MNN_InferenceEngine *engine;
    std::vector<MNN::Session *> sessions;
    std::vector<MNN::Tensor *> input_tensors;
    std::vector<MNN::Tensor *> output_tensors;

    std::mutex mutex;
    std::condition_variable cv;
    std::queue<size_t> available_sessions;
    std::string last_error;
};

// ============== Helper Functions ==============

// Initialize schedule and backend configs from MNNR_Config.
// Caller must ensure `schedule` and `backend` outlive any use of schedule.backendConfig.
static void init_schedule_config(MNN::ScheduleConfig &schedule, MNN::BackendConfig &backend, const MNNR_Config *config)
{
    schedule.type = (config) ? static_cast<MNNForwardType>(config->forward_type) : MNN_FORWARD_CPU;
    schedule.numThread = config ? config->thread_count : 4;
    if (schedule.numThread <= 0)
    {
        schedule.numThread = 4;
    }

    if (config)
    {
        switch (config->precision_mode)
        {
        case 1:
            backend.precision = MNN::BackendConfig::Precision_Low;
            break;
        case 2:
            backend.precision = MNN::BackendConfig::Precision_High;
            break;
        default:
            backend.precision = MNN::BackendConfig::Precision_Normal;
            break;
        }
    }
    schedule.backendConfig = &backend;
}

static bool init_engine_tensors(MNN_InferenceEngine *engine)
{
    if (!engine->interpreter || !engine->default_session)
    {
        return false;
    }

    // Get input tensor
    auto input_map = engine->interpreter->getSessionInputAll(engine->default_session);
    if (input_map.empty())
    {
        engine->last_error = "No input tensors found";
        return false;
    }

    engine->input_tensor = input_map.begin()->second;
    auto input_shape_vec = engine->input_tensor->shape();
    engine->input_shape.assign(input_shape_vec.begin(), input_shape_vec.end());

    // Get output tensor
    auto output_map = engine->interpreter->getSessionOutputAll(engine->default_session);
    if (output_map.empty())
    {
        engine->last_error = "No output tensors found";
        return false;
    }

    engine->output_tensor = output_map.begin()->second;
    auto output_shape_vec = engine->output_tensor->shape();
    engine->output_shape.assign(output_shape_vec.begin(), output_shape_vec.end());

    return true;
}

// ============== Version & Info ==============

const char *mnnr_get_version(void)
{
    return MNN_VERSION;
}

bool mnnr_is_backend_available(int32_t forward_type)
{
    if (forward_type < MNN_FORWARD_CPU || forward_type >= MNN_FORWARD_ALL)
    {
        return false;
    }
    return MNN::MNNGetExtraRuntimeCreator(static_cast<MNNForwardType>(forward_type)) != nullptr;
}

// ============== Shared Runtime API ==============

MNN_SharedRuntime *mnnr_create_runtime(const MNNR_Config *config)
{
    auto runtime = new MNN_SharedRuntime();

    runtime->thread_count = config ? config->thread_count : 4;
    if (runtime->thread_count <= 0)
    {
        runtime->thread_count = 4;
    }

    runtime->precision_mode = config ? config->precision_mode : 0;

    runtime->schedule_config.type = (config) ? static_cast<MNNForwardType>(config->forward_type) : MNN_FORWARD_CPU;
    runtime->schedule_config.numThread = runtime->thread_count;

    switch (runtime->precision_mode)
    {
    case 1:
        runtime->backend_config.precision = MNN::BackendConfig::Precision_Low;
        break;
    case 2:
        runtime->backend_config.precision = MNN::BackendConfig::Precision_High;
        break;
    default:
        runtime->backend_config.precision = MNN::BackendConfig::Precision_Normal;
        break;
    }
    runtime->schedule_config.backendConfig = &runtime->backend_config;

    return runtime;
}

void mnnr_destroy_runtime(MNN_SharedRuntime *runtime)
{
    delete runtime;
}

// ============== Inference Engine API ==============

MNN_InferenceEngine *mnnr_create_engine(
    const void *buffer,
    size_t size,
    const MNNR_Config *config)
{
    if (!buffer || size == 0)
    {
        return nullptr;
    }

    auto engine = new MNN_InferenceEngine();

    // Create interpreter from buffer
    engine->interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    if (!engine->interpreter)
    {
        engine->last_error = "Failed to create interpreter from buffer";
        delete engine;
        return nullptr;
    }

    // Create default session
    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);
    engine->default_session = engine->interpreter->createSession(schedule);
    if (!engine->default_session)
    {
        engine->last_error = "Failed to create default session";
        delete engine;
        return nullptr;
    }

    // Initialize tensors
    if (!init_engine_tensors(engine))
    {
        delete engine;
        return nullptr;
    }

    return engine;
}

MNN_InferenceEngine *mnnr_create_engine_with_runtime(
    const void *buffer,
    size_t size,
    MNN_SharedRuntime *runtime)
{
    if (!buffer || size == 0 || !runtime)
    {
        return nullptr;
    }

    auto engine = new MNN_InferenceEngine();
    engine->runtime = runtime;
    engine->owns_runtime = false;

    // Create interpreter from buffer
    engine->interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    if (!engine->interpreter)
    {
        engine->last_error = "Failed to create interpreter from buffer";
        delete engine;
        return nullptr;
    }

    // Create session using shared runtime config
    engine->default_session = engine->interpreter->createSession(runtime->schedule_config);
    if (!engine->default_session)
    {
        engine->last_error = "Failed to create session with shared runtime";
        delete engine;
        return nullptr;
    }

    // Initialize tensors
    if (!init_engine_tensors(engine))
    {
        delete engine;
        return nullptr;
    }

    return engine;
}

void mnnr_destroy_engine(MNN_InferenceEngine *engine)
{
    if (engine)
    {
        if (engine->default_session && engine->interpreter)
        {
            engine->interpreter->releaseSession(engine->default_session);
        }
        delete engine;
    }
}

MNNR_ErrorCode mnnr_get_input_shape(
    const MNN_InferenceEngine *engine,
    size_t *dims,
    size_t *out_ndims)
{
    if (!engine || !dims || !out_ndims)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    *out_ndims = engine->input_shape.size();
    for (size_t i = 0; i < engine->input_shape.size() && i < 8; i++)
    {
        dims[i] = static_cast<size_t>(engine->input_shape[i]);
    }

    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_get_output_shape(
    const MNN_InferenceEngine *engine,
    size_t *dims,
    size_t *out_ndims)
{
    if (!engine || !dims || !out_ndims)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    *out_ndims = engine->output_shape.size();
    for (size_t i = 0; i < engine->output_shape.size() && i < 8; i++)
    {
        dims[i] = static_cast<size_t>(engine->output_shape[i]);
    }

    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_run_inference(
    MNN_InferenceEngine *engine,
    const float *input_data,
    size_t input_size,
    float *output_data,
    size_t output_size)
{
    if (!engine || !input_data || !output_data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Use global lock to serialize MNN inference (thread pool limit)
    std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);
    std::lock_guard<std::mutex> lock(engine->mutex);

    // Calculate expected sizes
    size_t expected_input = 1;
    for (int dim : engine->input_shape)
    {
        expected_input *= dim;
    }

    size_t expected_output = 1;
    for (int dim : engine->output_shape)
    {
        expected_output *= dim;
    }

    if (input_size != expected_input || output_size != expected_output)
    {
        engine->last_error = "Input/output size mismatch";
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Create host tensor and copy input data
    auto input_host = make_unique_ptr<MNN::Tensor>(engine->input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, input_size * sizeof(float));
    engine->input_tensor->copyFromHostTensor(input_host.get());

    // Run inference
    MNN::ErrorCode code = engine->interpreter->runSession(engine->default_session);
    if (code != MNN::NO_ERROR)
    {
        engine->last_error = "Inference failed";
        return MNNR_ERROR_RUNTIME_ERROR;
    }

    // Copy output data
    auto output_host = make_unique_ptr<MNN::Tensor>(engine->output_tensor, MNN::Tensor::CAFFE);
    engine->output_tensor->copyToHostTensor(output_host.get());
    std::memcpy(output_data, output_host->host<float>(), output_size * sizeof(float));

    return MNNR_SUCCESS;
}

const char *mnnr_get_last_error(const MNN_InferenceEngine *engine)
{
    if (!engine)
    {
        return "Engine is null";
    }
    return engine->last_error.c_str();
}

// ============== Session Pool API ==============

MNN_SessionPool *mnnr_create_session_pool(
    MNN_InferenceEngine *engine,
    size_t pool_size,
    const MNNR_Config *config)
{
    if (!engine || pool_size == 0)
    {
        return nullptr;
    }

    auto pool = new MNN_SessionPool();
    pool->engine = engine;

    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);

    // Create sessions
    for (size_t i = 0; i < pool_size; i++)
    {
        MNN::Session *session = engine->interpreter->createSession(schedule);
        if (!session)
        {
            // Cleanup on failure
            for (auto s : pool->sessions)
            {
                engine->interpreter->releaseSession(s);
            }
            // Note: input/output tensors are owned by MNN sessions, not by us.
            // They will be freed when sessions are released above.
            delete pool;
            return nullptr;
        }

        pool->sessions.push_back(session);
        pool->available_sessions.push(i);

        // Get input/output tensors for this session
        auto input_map = engine->interpreter->getSessionInputAll(session);
        auto output_map = engine->interpreter->getSessionOutputAll(session);

        pool->input_tensors.push_back(input_map.begin()->second);
        pool->output_tensors.push_back(output_map.begin()->second);
    }

    return pool;
}

void mnnr_destroy_session_pool(MNN_SessionPool *pool)
{
    if (pool)
    {
        for (auto session : pool->sessions)
        {
            if (pool->engine && pool->engine->interpreter)
            {
                pool->engine->interpreter->releaseSession(session);
            }
        }
        delete pool;
    }
}

MNNR_ErrorCode mnnr_session_pool_run(
    MNN_SessionPool *pool,
    const float *input_data,
    size_t input_size,
    float *output_data,
    size_t output_size)
{
    if (!pool || !input_data || !output_data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Acquire a session (this will block if all sessions are busy)
    size_t session_idx;
    {
        std::unique_lock<std::mutex> lock(pool->mutex);
        pool->cv.wait(lock, [pool]
                      { return !pool->available_sessions.empty(); });
        session_idx = pool->available_sessions.front();
        pool->available_sessions.pop();
    }

    // Run inference with global lock to serialize MNN thread pool access
    MNNR_ErrorCode result = MNNR_SUCCESS;

    auto *session = pool->sessions[session_idx];
    auto *input_tensor = pool->input_tensors[session_idx];
    auto *output_tensor = pool->output_tensors[session_idx];

    // Create host tensor and copy input (can be done outside the global lock)
    auto input_host = make_unique_ptr<MNN::Tensor>(input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, input_size * sizeof(float));

    {
        // Global lock for MNN inference to avoid thread pool exhaustion
        std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);

        input_tensor->copyFromHostTensor(input_host.get());

        // Run inference
        MNN::ErrorCode code = pool->engine->interpreter->runSession(session);
        if (code != MNN::NO_ERROR)
        {
            pool->last_error = "Session pool inference failed";
            result = MNNR_ERROR_RUNTIME_ERROR;
        }
        else
        {
            // Copy output
            auto output_host = make_unique_ptr<MNN::Tensor>(output_tensor, MNN::Tensor::CAFFE);
            output_tensor->copyToHostTensor(output_host.get());
            std::memcpy(output_data, output_host->host<float>(), output_size * sizeof(float));
        }
    }

    // Release session
    {
        std::lock_guard<std::mutex> lock(pool->mutex);
        pool->available_sessions.push(session_idx);
    }
    pool->cv.notify_one();

    return result;
}

size_t mnnr_session_pool_available(const MNN_SessionPool *pool)
{
    if (!pool)
    {
        return 0;
    }
    std::lock_guard<std::mutex> lock(const_cast<MNN_SessionPool *>(pool)->mutex);
    return pool->available_sessions.size();
}

const char *mnnr_session_pool_get_last_error(const MNN_SessionPool *pool)
{
    if (!pool)
    {
        return "Pool is null";
    }
    return pool->last_error.c_str();
}

// ============== Single Session API ==============

MNN_SingleSession *mnnr_create_session(
    MNN_InferenceEngine *engine,
    const MNNR_Config *config)
{
    if (!engine)
    {
        return nullptr;
    }

    auto session = new MNN_SingleSession();
    session->engine = engine;

    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);
    session->session = engine->interpreter->createSession(schedule);

    if (!session->session)
    {
        delete session;
        return nullptr;
    }

    // Get tensors
    auto input_map = engine->interpreter->getSessionInputAll(session->session);
    auto output_map = engine->interpreter->getSessionOutputAll(session->session);

    if (input_map.empty() || output_map.empty())
    {
        engine->interpreter->releaseSession(session->session);
        delete session;
        return nullptr;
    }

    session->input_tensor = input_map.begin()->second;
    session->output_tensor = output_map.begin()->second;

    return session;
}

void mnnr_destroy_session(MNN_SingleSession *session)
{
    if (session)
    {
        if (session->session && session->engine && session->engine->interpreter)
        {
            session->engine->interpreter->releaseSession(session->session);
        }
        delete session;
    }
}

MNNR_ErrorCode mnnr_run_inference_with_session(
    MNN_SingleSession *session,
    const float *input_data,
    size_t input_size,
    float *output_data,
    size_t output_size)
{
    if (!session || !input_data || !output_data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Create host tensor and copy input (outside lock)
    auto input_host = make_unique_ptr<MNN::Tensor>(session->input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, input_size * sizeof(float));

    {
        // Global lock for MNN inference to avoid thread pool exhaustion
        std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);

        session->input_tensor->copyFromHostTensor(input_host.get());

        // Run inference
        MNN::ErrorCode code = session->engine->interpreter->runSession(session->session);
        if (code != MNN::NO_ERROR)
        {
            session->last_error = "Session inference failed";
            return MNNR_ERROR_RUNTIME_ERROR;
        }

        // Copy output
        auto output_host = make_unique_ptr<MNN::Tensor>(session->output_tensor, MNN::Tensor::CAFFE);
        session->output_tensor->copyToHostTensor(output_host.get());
        std::memcpy(output_data, output_host->host<float>(), output_size * sizeof(float));
    }

    return MNNR_SUCCESS;
}

const char *mnnr_session_get_last_error(const MNN_SingleSession *session)
{
    if (!session)
    {
        return "Session is null";
    }
    return session->last_error.c_str();
}

// ============== Dynamic Shape API ==============

MNNR_ErrorCode mnnr_run_inference_dynamic(
    MNN_InferenceEngine *engine,
    const float *input_data,
    const size_t *input_dims,
    size_t input_ndims,
    float **output_data,
    size_t *output_size,
    size_t *output_dims,
    size_t *output_ndims)
{
    if (!engine || !input_data || !input_dims || !output_data || !output_size || !output_dims || !output_ndims)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);
    std::lock_guard<std::mutex> lock(engine->mutex);

    // Build new input shape
    std::vector<int> new_shape(input_ndims);
    size_t total_input_size = 1;
    for (size_t i = 0; i < input_ndims; i++)
    {
        new_shape[i] = static_cast<int>(input_dims[i]);
        total_input_size *= input_dims[i];
    }

    // Resize input tensor
    engine->interpreter->resizeTensor(engine->input_tensor, new_shape);
    engine->interpreter->resizeSession(engine->default_session);

    // Get the updated input tensor after resize
    auto input_map = engine->interpreter->getSessionInputAll(engine->default_session);
    if (input_map.empty())
    {
        engine->last_error = "No input tensors found after resize";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    engine->input_tensor = input_map.begin()->second;

    // Create host tensor and copy input data
    auto input_host = make_unique_ptr<MNN::Tensor>(engine->input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, total_input_size * sizeof(float));
    engine->input_tensor->copyFromHostTensor(input_host.get());

    // Run inference
    MNN::ErrorCode code = engine->interpreter->runSession(engine->default_session);
    if (code != MNN::NO_ERROR)
    {
        engine->last_error = "Dynamic inference failed";
        return MNNR_ERROR_RUNTIME_ERROR;
    }

    // Get output tensor after inference
    auto output_map = engine->interpreter->getSessionOutputAll(engine->default_session);
    if (output_map.empty())
    {
        engine->last_error = "No output tensors found";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    engine->output_tensor = output_map.begin()->second;

    // Get output shape
    auto output_shape = engine->output_tensor->shape();
    *output_ndims = output_shape.size();
    size_t total_output_size = 1;
    for (size_t i = 0; i < output_shape.size() && i < 8; i++)
    {
        output_dims[i] = static_cast<size_t>(output_shape[i]);
        total_output_size *= output_shape[i];
    }
    *output_size = total_output_size;

    // Allocate output buffer
    *output_data = new float[total_output_size];

    // Copy output data
    auto output_host = make_unique_ptr<MNN::Tensor>(engine->output_tensor, MNN::Tensor::CAFFE);
    engine->output_tensor->copyToHostTensor(output_host.get());
    std::memcpy(*output_data, output_host->host<float>(), total_output_size * sizeof(float));

    return MNNR_SUCCESS;
}

void mnnr_free_output(float *output_data)
{
    delete[] output_data;
}
