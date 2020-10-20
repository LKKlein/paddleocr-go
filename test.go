package main

import "paddleocr-go/core"

func main() {
	args := make(map[string]interface{})
	args["det_model_dir"] = "/home/lvkun/.paddleocr/det"
	args["cls_model_dir"] = "/home/lvkun/.paddleocr/cls"
	args["rec_model_dir"] = "/home/lvkun/.paddleocr/rec/ch"
	args["use_angle_cls"] = true
	args["use_gpu"] = false
	sys := core.NewTextPredictSystem(args)
	img := core.ReadImage("/home/lvkun/projects/PaddleOCR/0.jpg")
	sys.Run(img)
}
