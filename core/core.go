package core

import (
	"image"
	"log"
	"math"
	"sort"

	"github.com/LKKlein/gocv"
)

type OCRText struct {
	bbox  [][]int
	text  string
	score float64
}

type TextPredictSystem struct {
	detector *DBDetector
	cls      *TextClassifier
	rec      *TextRecognizer
}

func NewTextPredictSystem(args map[string]interface{}) *TextPredictSystem {
	sys := &TextPredictSystem{
		detector: NewDBDetector(getString(args, "det_model_dir", ""), args),
		rec:      NewTextRecognizer(getString(args, "rec_model_dir", ""), args),
	}
	if getBool(args, "use_angle_cls", false) {
		sys.cls = NewTextClassifier(getString(args, "cls_model_dir", ""), args)
	}
	return sys
}

func (t *TextPredictSystem) sortBoxes(boxes [][][]int) [][][]int {
	sort.Slice(boxes, func(i, j int) bool {
		if boxes[i][0][1] < boxes[j][0][1] {
			return true
		}
		if boxes[i][0][1] > boxes[j][0][1] {
			return false
		}
		return boxes[i][0][0] < boxes[j][0][0]
	})

	for i := 0; i < len(boxes)-1; i++ {
		if math.Abs(float64(boxes[i+1][0][1]-boxes[i][0][1])) < 10 && boxes[i+1][0][0] < boxes[i][0][0] {
			boxes[i], boxes[i+1] = boxes[i+1], boxes[i]
		}
	}
	return boxes
}

func (t *TextPredictSystem) getRotateCropImage(img gocv.Mat, box [][]int) gocv.Mat {
	boxX := []int{box[0][0], box[1][0], box[2][0], box[3][0]}
	boxY := []int{box[0][1], box[1][1], box[2][1], box[3][1]}

	left, right, top, bottom := mini(boxX), maxi(boxX), mini(boxY), maxi(boxY)
	cropimg := img.Region(image.Rect(left, top, right, bottom))
	for i := 0; i < len(box); i++ {
		box[i][0] -= left
		box[i][1] -= top
	}

	cropW := int(math.Sqrt(math.Pow(float64(box[0][0]-box[1][0]), 2) + math.Pow(float64(box[0][1]-box[1][1]), 2)))
	cropH := int(math.Sqrt(math.Pow(float64(box[0][0]-box[3][0]), 2) + math.Pow(float64(box[0][1]-box[3][1]), 2)))
	ptsstd := make([]image.Point, 4)
	ptsstd[0] = image.Pt(0, 0)
	ptsstd[1] = image.Pt(cropW, 0)
	ptsstd[2] = image.Pt(cropW, cropH)
	ptsstd[3] = image.Pt(0, cropH)

	points := make([]image.Point, 4)
	points[0] = image.Pt(box[0][0], box[0][1])
	points[1] = image.Pt(box[1][0], box[1][1])
	points[2] = image.Pt(box[2][0], box[2][1])
	points[3] = image.Pt(box[3][0], box[3][1])

	M := gocv.GetPerspectiveTransform(points, ptsstd)
	defer M.Close()
	dstimg := gocv.NewMat()
	gocv.WarpPerspective(cropimg, &dstimg, M, image.Pt(cropW, cropH))

	if float64(dstimg.Rows()) >= float64(dstimg.Cols())*1.5 {
		srcCopy := gocv.NewMat()
		gocv.Transpose(dstimg, &srcCopy)
		defer dstimg.Close()
		gocv.Flip(srcCopy, &srcCopy, 0)
		return srcCopy
	}
	return dstimg
}

func (t *TextPredictSystem) Run(img gocv.Mat) []OCRText {
	result := make([]OCRText, 0, 10)

	srcimg := gocv.NewMat()
	img.CopyTo(&srcimg)
	boxes := t.detector.Run(img)
	if len(boxes) == 0 {
		return result
	}

	boxes = t.sortBoxes(boxes)
	cropimages := make([]gocv.Mat, len(boxes))
	for i := 0; i < len(boxes); i++ {
		tmpbox := make([][]int, len(boxes[i]))
		copy(tmpbox, boxes[i])
		cropimg := t.getRotateCropImage(srcimg, tmpbox)
		// log.Println(cropimg.Rows(), cropimg.Cols(), cropimg.Channels())
		cropimages[i] = cropimg
	}
	// cropimages = make([]gocv.Mat, 1)
	// cropimages[0] = img
	if t.cls != nil {
		var clsOut []ClsResult
		var clsPredictTime int64
		cropimages, clsOut, clsPredictTime = t.cls.Run(cropimages)
		log.Println("cls num: ", len(clsOut), ", cls time elapse: ", clsPredictTime, "ms")
	}

	return nil
}
