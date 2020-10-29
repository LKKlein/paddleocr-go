#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>
#include <iostream>
#include "paddle_api.h"
#include "c_api_internal.h"
#include "paddle_c_api.h"

using paddle::ConvertToACPrecision;
using paddle::ConvertToPaddleDType;
using paddle::ConvertToPDDataType;

extern "C" {
PD_Predictor* PD_NewPredictor(const PD_AnalysisConfig* config) {
  PD_Predictor* predictor = new PD_Predictor;
  predictor->predictor = paddle::CreatePaddlePredictor(config->config);
  return predictor;
}

void PD_DeletePredictor(PD_Predictor* predictor) {
  if (predictor) {
    predictor->predictor = nullptr;
    delete predictor;
    predictor = nullptr;
  }
}

int PD_GetInputNum(const PD_Predictor* predictor) {
  return static_cast<int>(predictor->predictor->GetInputNames().size());
}

int PD_GetOutputNum(const PD_Predictor* predictor) {
  return static_cast<int>(predictor->predictor->GetOutputNames().size());
}

const char* PD_GetInputName(const PD_Predictor* predictor, int n) {
  std::vector<std::string> names = predictor->predictor->GetInputNames();
  return names[n].c_str();
}

const char* PD_GetOutputName(const PD_Predictor* predictor, int n) {
  std::vector<std::string> names =
      predictor->predictor->GetOutputNames();
  return names[n].c_str();
}

void PD_SetZeroCopyInput(PD_Predictor* predictor,
                         const PD_ZeroCopyTensor* tensor) {
  auto input = predictor->predictor->GetInputTensor(tensor->name);
  auto* shape_ptr = static_cast<int*>(tensor->shape.data);
  std::vector<int> shape(shape_ptr,
                         shape_ptr + tensor->shape.length / sizeof(int));
  input->Reshape(std::move(shape));
  switch (tensor->dtype) {
    case PD_FLOAT32:
      input->copy_from_cpu(static_cast<float*>(tensor->data.data));
      break;
    case PD_INT32:
      input->copy_from_cpu(static_cast<int32_t*>(tensor->data.data));
      break;
    case PD_INT64:
      input->copy_from_cpu(static_cast<int64_t*>(tensor->data.data));
      break;
    case PD_UINT8:
      input->copy_from_cpu(static_cast<uint8_t*>(tensor->data.data));
      break;
    default:
      break;
  }

  if (tensor->lod.length) {
    auto* lod_ptr = reinterpret_cast<size_t*>(tensor->lod.data);
    std::vector<size_t> lod;
    lod.assign(lod_ptr, lod_ptr + tensor->lod.length / sizeof(size_t));
    input->SetLoD({std::move(lod)});
  }
}

void PD_GetZeroCopyOutput(PD_Predictor* predictor, PD_ZeroCopyTensor* tensor) {
  auto output = predictor->predictor->GetOutputTensor(tensor->name);
  tensor->dtype = ConvertToPDDataType(output->type());
  auto shape = output->shape();
  size_t shape_size = shape.size();
  if (tensor->shape.capacity < shape_size * sizeof(int)) {
    if (tensor->shape.data || tensor->shape.capacity) {
      std::free(tensor->shape.data);
    }
    tensor->shape.data = std::malloc(shape_size * sizeof(int));
    tensor->shape.capacity = shape_size * sizeof(int);
  }
  tensor->shape.length = shape_size * sizeof(int);
  std::copy(shape.begin(), shape.end(), static_cast<int*>(tensor->shape.data));

  int n =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  size_t length = n * paddle::PaddleDtypeSize(output->type());
  if (tensor->data.capacity < length) {
    if (tensor->data.data) {
      std::free(tensor->data.data);
    }
    tensor->data.data = std::malloc(length);
    tensor->data.capacity = std::move(length);
  }
  tensor->data.length = length;

  auto lod = output->lod();
  if (!lod.empty()) {
    tensor->lod.length = lod.front().size() * sizeof(size_t);
    if (tensor->lod.capacity < lod.front().size()) {
      if (tensor->lod.data) {
        std::free(tensor->lod.data);
      }

      tensor->lod.data = std::malloc(lod.front().size() * sizeof(size_t));
      tensor->lod.capacity = lod.front().size() * sizeof(size_t);
    }
    std::copy(lod.front().begin(), lod.front().end(),
              reinterpret_cast<size_t*>(tensor->lod.data));
  }
  switch (tensor->dtype) {
    case PD_FLOAT32:
      output->copy_to_cpu(reinterpret_cast<float*>(tensor->data.data));
      break;
    case PD_INT32:
      output->copy_to_cpu(reinterpret_cast<int32_t*>(tensor->data.data));
      break;
    case PD_INT64:
      output->copy_to_cpu(reinterpret_cast<int64_t*>(tensor->data.data));
      break;
    case PD_UINT8:
      output->copy_to_cpu(reinterpret_cast<uint8_t*>(tensor->data.data));
      break;
    default:
      break;
  }
}

void PD_ZeroCopyRun(PD_Predictor* predictor) {
  predictor->predictor->ZeroCopyRun();
}
}  // extern "C"
