package main

import "paddleocr-go/ocr"

func main() {
	sys := ocr.NewTextPredictSystem("config/conf.yaml")
	img := ocr.ReadImage("test.jpg")
	sys.Run(img)
}
