package core

import "github.com/LKKlein/gocv"

type TextRecognizer struct {
}

func NewTextRecognizer(modelDir string, args map[string]interface{}) *TextRecognizer {
	reco := &TextRecognizer{}
	return reco
}

func (t *TextRecognizer) loadModel(modelDir string) {

}

func (t *TextRecognizer) Run(imgs []gocv.Mat) {

}
