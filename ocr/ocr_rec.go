package ocr

import (
	"log"
	"reflect"
	"time"

	"github.com/LKKlein/gocv"
)

type TextRecognizer struct {
	*PaddleModel
	batchNum int
	textLen  int
	shape    []int
	charType string
}

func NewTextRecognizer(modelDir string, args map[string]interface{}) *TextRecognizer {
	shapes := []int{3, 32, 320}
	if v, ok := args["rec_image_shape"]; ok {
		shapes = v.([]int)
	}
	rec := &TextRecognizer{
		PaddleModel: NewPaddleModel(args),
		batchNum:    getInt(args, "rec_batch_num", 1),
		textLen:     getInt(args, "max_text_len", 25),
		charType:    getString(args, "rec_char_type", "ch"),
		shape:       shapes,
	}
	rec.LoadModel(modelDir)
	return rec
}

func (rec *TextRecognizer) Run(imgs []gocv.Mat) []OCRText {
	recResult := make([]OCRText, 0, len(imgs))
	batch := rec.batchNum
	var recTime int64 = 0
	c, h, w := rec.shape[0], rec.shape[1], rec.shape[2]
	for i := 0; i < len(imgs); i += batch {
		j := i + batch
		if len(imgs) < j {
			j = len(imgs)
		}

		maxwhratio := 0.0
		for k := i; k < j; k++ {
			h, w := imgs[k].Rows(), imgs[k].Cols()
			ratio := float64(w) / float64(h)
			if ratio > maxwhratio {
				maxwhratio = ratio
			}
		}

		normimgs := make([]float32, (j-i)*c*h*w)
		for k := i; k < j; k++ {
			img := crnnResize(imgs[k], rec.shape, maxwhratio, rec.charType)
			data := normPermute(img, []float32{0.5, 0.5, 0.5}, []float32{0.5, 0.5, 0.5}, 255.0)
			copy(normimgs[(k-i)*c*h*w:], data)
		}

		st := time.Now()
		rec.input.SetValue(normimgs)
		rec.input.Reshape([]int32{int32(j - i), int32(c), int32(w), int32(w)})

		rec.predictor.SetZeroCopyInput(rec.input)
		rec.predictor.ZeroCopyRun()
		rec.predictor.GetZeroCopyOutput(rec.outputs[0])
		rec.predictor.GetZeroCopyOutput(rec.outputs[1])

		outputVal0 := rec.outputs[0].Value()
		value0 := reflect.ValueOf(outputVal0)
		var recIdxBatch [][]int64 = value0.Interface().([][]int64)

		outputVal1 := rec.outputs[1].Value()
		value1 := reflect.ValueOf(outputVal1)
		var predictBatch [][]float32 = value1.Interface().([][]float32)
		recTime += int64(time.Since(st).Milliseconds())
		log.Println(recIdxBatch, predictBatch)
	}
	log.Println("rec num: ", len(recResult), ", rec time elapse: ", recTime, "ms")
	return recResult
}
