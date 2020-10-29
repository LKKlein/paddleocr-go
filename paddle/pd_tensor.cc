#include <cstdlib>
#include <cstring>
#include <memory>
#include "c_api_internal.h"
#include "paddle_c_api.h"

using paddle::ConvertToACPrecision;
using paddle::ConvertToPaddleDType;
using paddle::ConvertToPDDataType;

extern "C" {

PD_ZeroCopyTensor* PD_NewZeroCopyTensor() {
  auto* tensor = new PD_ZeroCopyTensor;
  PD_InitZeroCopyTensor(tensor);
  return tensor;
}
void PD_DeleteZeroCopyTensor(PD_ZeroCopyTensor* tensor) {
  if (tensor) {
    PD_DestroyZeroCopyTensor(tensor);
    delete tensor;
  }
  tensor = nullptr;
}

void PD_InitZeroCopyTensor(PD_ZeroCopyTensor* tensor) {
  std::memset(tensor, 0, sizeof(PD_ZeroCopyTensor));
}

void PD_DestroyZeroCopyTensor(PD_ZeroCopyTensor* tensor) {
  std::free(tensor->data.data);
  std::free(tensor->shape.data);
  std::free(tensor->lod.data);
}

}  // extern "C"
