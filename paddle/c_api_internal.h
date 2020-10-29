#pragma once

#include <memory>
#include "paddle_analysis_config.h"
#include "paddle_api.h"
#include "paddle_c_api.h"

using PD_PaddleDType = paddle::PaddleDType;
using PD_ACPrecision = paddle::AnalysisConfig::Precision;

struct PD_AnalysisConfig {
  paddle::AnalysisConfig config;
};

struct PD_Tensor {
  paddle::PaddleTensor tensor;
};

struct PD_PaddleBuf {
  paddle::PaddleBuf buf;
};

struct PD_Predictor {
  std::unique_ptr<paddle::PaddlePredictor> predictor;
//   paddle::PaddlePredictor* predictor;
};

namespace paddle {
paddle::PaddleDType ConvertToPaddleDType(PD_DataType dtype);

PD_DataType ConvertToPDDataType(PD_PaddleDType dtype);

PD_ACPrecision ConvertToACPrecision(Precision dtype);
}  // namespace paddle
