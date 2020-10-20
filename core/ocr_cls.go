package core

import (
	"paddleocr-go/paddle"
	"reflect"
	"time"

	"github.com/LKKlein/gocv"
)

type TextClassifier struct {
	predictor *paddle.Predictor
	input     *paddle.ZeroCopyTensor
	outputs   []*paddle.ZeroCopyTensor

	batchNum int
	thresh   float64
	shape    []int
	labels   []string

	useGPU     bool
	deviceID   int
	initGPUMem int
	numThreads int
	useMKLDNN  bool
}

type ClsResult struct {
	Score float32
	Label int64
}

func NewTextClassifier(modelDir string, args map[string]interface{}) *TextClassifier {
	shapes := []int{3, 48, 192}
	if v, ok := args["cls_image_shape"]; ok {
		shapes = v.([]int)
	}
	cls := &TextClassifier{
		batchNum: getInt(args, "cls_batch_num", 1),
		thresh:   getFloat64(args, "cls_thresh", 0.9),
		shape:    shapes,

		useGPU:     getBool(args, "use_gpu", false),
		deviceID:   getInt(args, "gpu_id", 0),
		initGPUMem: getInt(args, "gpu_mem", 1000),
		numThreads: getInt(args, "num_threads", 6),
		useMKLDNN:  getBool(args, "use_mkldnn", false),
	}
	cls.loadModel(modelDir)
	return cls
}

func (t *TextClassifier) loadModel(modelDir string) {
	config := paddle.NewAnalysisConfig()
	config.SetModel(modelDir+"/model", modelDir+"/params")
	if t.useGPU {
		config.EnableUseGpu(t.initGPUMem, t.deviceID)
	} else {
		config.DisableGpu()
		config.SetCpuMathLibraryNumThreads(t.numThreads)
		if t.useMKLDNN {
			config.EnableMkldnn()
		}
	}

	config.EnableMemoryOptim()
	// config.DisableGlogInfo()
	config.SwitchIrOptim(true)

	// false for zero copy tensor
	config.SwitchUseFeedFetchOps(false)
	config.SwitchSpecifyInputNames(true)

	t.predictor = paddle.NewPredictor(config)
	t.input = t.predictor.GetInputTensors()[0]
	t.outputs = t.predictor.GetOutputTensors()
}

func (t *TextClassifier) Run(imgs []gocv.Mat) ([]gocv.Mat, []ClsResult, int64) {
	batch := t.batchNum
	var clsTime int64 = 0
	clsout := make([]ClsResult, len(imgs))
	srcimgs := make([]gocv.Mat, len(imgs))
	c, h, w := t.shape[0], t.shape[1], t.shape[2]
	for i := 0; i < len(imgs); i += batch {
		j := i + batch
		if len(imgs) < j {
			j = len(imgs)
		}

		normImgs := make([]float32, (j-i)*c*h*w)
		for k := i; k < j; k++ {
			tmp := gocv.NewMat()
			imgs[k].CopyTo(&tmp)
			srcimgs[k] = tmp
			img := clsResize(imgs[k], t.shape)
			data := normPermute(img, []float32{0.5, 0.5, 0.5}, []float32{0.5, 0.5, 0.5}, 255.0)
			copy(normImgs[(k-i)*c*h*w:], data)
		}

		st := time.Now()
		t.input.SetValue(normImgs)
		t.input.Reshape([]int32{int32(j - i), int32(c), int32(w), int32(w)})

		t.predictor.SetZeroCopyInput(t.input)
		t.predictor.ZeroCopyRun()
		t.predictor.GetZeroCopyOutput(t.outputs[0])
		t.predictor.GetZeroCopyOutput(t.outputs[1])

		var probout [][]float32
		var labelout []int64
		outputVal0 := t.outputs[0].Value()
		value0 := reflect.ValueOf(outputVal0)
		if len(t.outputs[0].Shape()) == 2 {
			probout = value0.Interface().([][]float32)
		} else {
			labelout = value0.Interface().([]int64)
		}

		outputVal1 := t.outputs[1].Value()
		value1 := reflect.ValueOf(outputVal1)
		if len(t.outputs[1].Shape()) == 2 {
			probout = value1.Interface().([][]float32)
		} else {
			labelout = value1.Interface().([]int64)
		}
		clsTime += int64(time.Since(st).Milliseconds())

		for no, label := range labelout {
			score := probout[no][label]
			clsout[i+no] = ClsResult{
				Score: score,
				Label: label,
			}

			if label%2 == 1 && float64(score) > t.thresh {
				gocv.Rotate(srcimgs[i+no], &srcimgs[i+no], gocv.Rotate180Clockwise)
			}
		}
	}
	return srcimgs, clsout, clsTime
}
