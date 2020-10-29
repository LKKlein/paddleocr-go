#include <algorithm>
#include <vector>
#include "c_api_internal.h"
#include "paddle_c_api.h"

using paddle::ConvertToACPrecision;
using paddle::ConvertToPaddleDType;
using paddle::ConvertToPDDataType;

extern "C" {

PD_PaddleBuf* PD_NewPaddleBuf() { return new PD_PaddleBuf; }

void PD_DeletePaddleBuf(PD_PaddleBuf* buf) {
  if (buf) {
    delete buf;
    buf = nullptr;
  }
}

void PD_PaddleBufResize(PD_PaddleBuf* buf, size_t length) {
  buf->buf.Resize(length);
}

void PD_PaddleBufReset(PD_PaddleBuf* buf, void* data, size_t length) {
  buf->buf.Reset(data, length);
}

bool PD_PaddleBufEmpty(PD_PaddleBuf* buf) {
  return buf->buf.empty();
}

void* PD_PaddleBufData(PD_PaddleBuf* buf) {
  return buf->buf.data();
}

size_t PD_PaddleBufLength(PD_PaddleBuf* buf) {
  return buf->buf.length();
}

}  // extern "C"

namespace paddle {
paddle::PaddleDType ConvertToPaddleDType(PD_DataType dtype) {
  switch (dtype) {
    case PD_FLOAT32:
      return PD_PaddleDType::FLOAT32;
    case PD_INT32:
      return PD_PaddleDType::INT32;
    case PD_INT64:
      return PD_PaddleDType::INT64;
    case PD_UINT8:
      return PD_PaddleDType::UINT8;
    default:
      return PD_PaddleDType::FLOAT32;
  }
}

PD_DataType ConvertToPDDataType(PD_PaddleDType dtype) {
  switch (dtype) {
    case PD_PaddleDType::FLOAT32:
      return PD_DataType::PD_FLOAT32;
    case PD_PaddleDType::INT32:
      return PD_DataType::PD_INT32;
    case PD_PaddleDType::INT64:
      return PD_DataType::PD_INT64;
    case PD_PaddleDType::UINT8:
      return PD_DataType::PD_UINT8;
    default:
      return PD_DataType::PD_UNKDTYPE;
  }
}

PD_ACPrecision ConvertToACPrecision(Precision dtype) {
  switch (dtype) {
    case Precision::kFloat32:
      return PD_ACPrecision::kFloat32;
    case Precision::kInt8:
      return PD_ACPrecision::kInt8;
    case Precision::kHalf:
      return PD_ACPrecision::kHalf;
    default:
      return PD_ACPrecision::kFloat32;
  }
}
}  // namespace paddle
