package ocr

import (
	"log"
	"time"

	"github.com/LKKlein/gocv"
)

type TextRecognizer struct {
	*PaddleModel
	batchNum int
	textLen  int
	shape    []int
	charType string
	labels   []string
}

func NewTextRecognizer(modelDir string, args map[string]interface{}) *TextRecognizer {
	shapes := []int{3, 32, 320}
	if v, ok := args["rec_image_shape"]; ok {
		shapes = v.([]int)
	}
	labelpath := getString(args, "rec_char_dict_path", "./config/ppocr_keys_v1.txt")
	rec := &TextRecognizer{
		PaddleModel: NewPaddleModel(args),
		batchNum:    getInt(args, "rec_batch_num", 1),
		textLen:     getInt(args, "max_text_len", 25),
		charType:    getString(args, "rec_char_type", "ch"),
		shape:       shapes,
		labels:      append(readLines2StringSlice(labelpath), " "),
	}
	rec.LoadModel(modelDir)
	return rec
}

func (rec *TextRecognizer) Run(imgs []gocv.Mat, bboxes [][][]int) []OCRText {
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
		// normimgs := make([]float32, 0, (j-i)*c*h*w)
		for k := i; k < j; k++ {
			img := crnnResize(imgs[k], rec.shape, maxwhratio, rec.charType)
			data := normPermute(img, []float32{0.5, 0.5, 0.5}, []float32{0.5, 0.5, 0.5}, 255.0)
			copy(normimgs[(k-i)*c*h*w:], data)
			// normimgs = append(normimgs, data...)
		}

		st := time.Now()
		rec.input.SetValue(normimgs)
		rec.input.Reshape([]int32{int32(j - i), int32(c), int32(w), int32(w)})

		rec.predictor.SetZeroCopyInput(rec.input)
		rec.predictor.ZeroCopyRun()
		rec.predictor.GetZeroCopyOutput(rec.outputs[0])
		rec.predictor.GetZeroCopyOutput(rec.outputs[1])

		recIdxBatch := rec.outputs[0].Value().([][]int64)
		recIdxLod := rec.outputs[0].Lod()

		predictBatch := rec.outputs[1].Value().([][]float32)
		predictLod := rec.outputs[1].Lod()
		recTime += int64(time.Since(st).Milliseconds())
		log.Println(recIdxBatch, predictBatch, recIdxLod, predictLod)

		for rno := 0; rno < len(recIdxLod)-1; rno++ {
			predIdx := make([]int, 0, 2)
			for beg := recIdxLod[rno]; beg < recIdxLod[rno+1]; beg++ {
				predIdx = append(predIdx, int(recIdxBatch[beg][0]))
			}
			if len(predIdx) == 0 {
				continue
			}
			words := ""
			for n := 0; n < len(predIdx); n++ {
				words += rec.labels[predIdx[n]]
			}

			score := 0.0
			count := 0
			blankPosition := int(rec.outputs[1].Shape()[1])
			for beg := predictLod[rno]; beg < predictLod[rno+1]; beg++ {
				argMaxID, maxVal := argmax(predictBatch[beg])
				if blankPosition-1-argMaxID > 0 {
					score += float64(maxVal)
					count++
				}
			}
			score = score / float64(count)
			recResult = append(recResult, OCRText{
				BBox:  bboxes[i+rno],
				Text:  words,
				Score: score,
			})
		}
	}
	log.Println("rec num: ", len(recResult), ", rec time elapse: ", recTime, "ms")
	return recResult
}
