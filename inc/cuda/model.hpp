#pragma once

#include "corekit/core.hpp"

#include <NvInfer.h>
#include <format>
#include <vector>
#include <vector_types.h>

namespace cuda {
namespace model {

    using namespace corekit::types;
    using namespace corekit::utils;

    struct IOBinding {
        using List = std::vector<IOBinding>;

        Name           name  = Name();
        size_t         size  = 1;
        size_t         dsize = 1;
        nvinfer1::Dims dims;
        void*          ptr;
    };

    struct NvLogger : public nvinfer1::ILogger {

        NvLogger(const Name& name, Severity severity = Severity::kWARNING)
            : logger(name)
            , level(severity)
        {
        }

        void setLevel(Severity severity) noexcept
        {
            this->level = severity;
        }

        void log(Severity severity, const char* msg) noexcept override
        {
            if (this->level < severity) {
                return;
            }

            switch (severity) {
                case Severity::kINTERNAL_ERROR: logger.fatal() << msg; break;
                case Severity::kERROR: logger.error() << msg; break;
                case Severity::kWARNING: logger.warn() << msg; break;
                case Severity::kINFO: logger.info() << msg; break;
                default: logger.debug() << msg; break;
            }
        }

        Severity               level;
        corekit::utils::Logger logger;
    };

    struct Model : public Device {
        Model(const Path& engine)
            : Device(engine.stem())
            , nvlog(engine.stem())
            , file(engine)
        {
        }

        ~Model()
        {
            unload();
        }

        void info() const
        {
            for (const IOBinding& input : inputs) {
                logger.info() << std::format("Input: {} - size: {} - dsize: {}", input.name, input.size, input.dsize);

                for (uint i = 0; i < input.dims.nbDims; i++) {
                    logger.info() << std::format("  dim[{}]: {}", i, input.dims.d[i]);
                }
            }

            for (const IOBinding& output : outputs) {
                logger.info() << std::format(
                    "Output: {} - size: {} - dsize: {}", output.name, output.size, output.dsize);

                for (uint i = 0; i < output.dims.nbDims; i++) {
                    logger.info() << std::format("  dim[{}]: {}", i, output.dims.d[i]);
                }
            }
        }

        cudaStream_t stream = 0;

      protected:
        virtual bool prepare() override
        {
            cudaStreamCreate(&stream);

            const Code& content  = File::loadTxt(file);
            const char* trtModel = content.data();

            runtime = nvinfer1::createInferRuntime(nvlog);
            if (!runtime) {
                logger.error() << "Failed to create TensorRT runtime.";
                return false;
            }

            engine = runtime->deserializeCudaEngine(trtModel, content.size());
            if (!engine) {
                logger.error() << "Failed to create TensorRT engine.";
                return false;
            }

            context = engine->createExecutionContext();
            if (!context) {
                logger.error() << "Failed to create TensorRT execution context.";
                return false;
            }

            auto num_bindings = engine->getNbIOTensors();

            for (int i = 0; i < num_bindings; ++i) {

                IOBinding binding;

                const char*        name  = this->engine->getIOTensorName(i);
                nvinfer1::DataType dtype = this->engine->getTensorDataType(name);

                binding.name  = name;
                binding.dsize = type_to_size(dtype);

                switch (engine->getTensorIOMode(name)) {
                    case nvinfer1::TensorIOMode::kINPUT: {
                        binding.dims = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
                        binding.size = get_size_by_dims(binding.dims);

                        cudaMallocAsync(&binding.ptr, binding.size * binding.dsize, stream);

                        inputs.push_back(binding);
                        context->setInputShape(name, binding.dims);
                        context->setTensorAddress(name, binding.ptr);
                    } break;

                    case nvinfer1::TensorIOMode::kOUTPUT: {
                        binding.dims = this->context->getTensorShape(name);
                        binding.size = get_size_by_dims(binding.dims);

                        cudaMallocAsync(&binding.ptr, binding.size * binding.dsize, stream);

                        outputs.push_back(binding);
                        context->setTensorAddress(name, binding.ptr);
                    } break;

                    default: throw std::runtime_error("Unsupported TensorIOMode"); break;
                }
            }

            this->info();
            return true;
        }

        virtual bool cleanup() override
        {
            if (context) {
                delete context;
            }
            if (engine) {
                delete engine;
            }
            if (runtime) {
                delete runtime;
            }
            if (stream) {
                cudaStreamDestroy(stream);
                stream = 0;
            }

            return true;
        }

        void exec() const
        {
            corecheck(this->isLoaded(), "Model is not loaded.");
            context->enqueueV3(stream);
        }

        static int get_size_by_dims(const nvinfer1::Dims& dims)
        {
            int size = 1;
            for (int i = 0; i < dims.nbDims; i++) {
                size *= dims.d[i];
            }
            return size;
        }

        static int type_to_size(const nvinfer1::DataType& dataType)
        {
            switch (dataType) {
                case nvinfer1::DataType::kINT32: return 4;
                case nvinfer1::DataType::kFLOAT: return 4;
                case nvinfer1::DataType::kHALF: return 2;
                case nvinfer1::DataType::kINT8: return 1;
                case nvinfer1::DataType::kBOOL: return 1;
                default: return 4;
            }
        }

        IOBinding::List inputs;
        IOBinding::List outputs;

        Path     file;
        NvLogger nvlog;

        nvinfer1::IRuntime*          runtime = nullptr;
        nvinfer1::ICudaEngine*       engine  = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
    };

} // namespace model
} // namespace cuda
