package main

import "paddleocr-go/ocr"

func main() {
	args := make(map[string]interface{})
	args["det_model_dir"] = "/home/lvkun/.paddleocr/det"
	args["cls_model_dir"] = "/home/lvkun/.paddleocr/cls"
	args["rec_model_dir"] = "/home/lvkun/.paddleocr/rec/ch"
	args["use_angle_cls"] = false
	args["use_gpu"] = true
	sys := ocr.NewTextPredictSystem(args)
	img := ocr.ReadImage("test.jpg")
	sys.Run(img)
}
