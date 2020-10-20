package core

import (
	"log"
	"paddleocr-go/paddle"
	"time"

	"github.com/LKKlein/gocv"
)

type DBDetector struct {
	preProcess  DetPreProcess
	postProcess DetPostProcess
	predictor   *paddle.Predictor
	input       *paddle.ZeroCopyTensor
	output      *paddle.ZeroCopyTensor

	useGPU     bool
	deviceID   int
	initGPUMem int
	numThreads int
	useMKLDNN  bool
}

func NewDBDetector(modelDir string, args map[string]interface{}) *DBDetector {
	maxSideLen := getInt(args, "det_max_side_len", 960)
	thresh := getFloat64(args, "det_db_thresh", 0.3)
	boxThresh := getFloat64(args, "det_db_box_thresh", 0.5)
	unClipRatio := getFloat64(args, "det_db_unclip_ratio", 2.0)

	detector := &DBDetector{
		useGPU:      getBool(args, "use_gpu", false),
		deviceID:    getInt(args, "gpu_id", 0),
		initGPUMem:  getInt(args, "gpu_mem", 1000),
		numThreads:  getInt(args, "num_threads", 6),
		useMKLDNN:   getBool(args, "use_mkldnn", false),
		preProcess:  NewDBProcess(make([]int, 0), maxSideLen),
		postProcess: NewDBPostProcess(thresh, boxThresh, unClipRatio),
	}
	detector.loadModel(modelDir)
	return detector
}

func (d *DBDetector) loadModel(modelDir string) {
	config := paddle.NewAnalysisConfig()
	config.SetModel(modelDir+"/model", modelDir+"/params")
	if d.useGPU {
		config.EnableUseGpu(d.initGPUMem, d.deviceID)
	} else {
		config.DisableGpu()
		config.SetCpuMathLibraryNumThreads(d.numThreads)
		if d.useMKLDNN {
			config.EnableMkldnn()
		}
	}

	config.EnableMemoryOptim()
	// config.DisableGlogInfo()
	config.SwitchIrOptim(true)

	// false for zero copy tensor
	config.SwitchUseFeedFetchOps(false)
	config.SwitchSpecifyInputNames(true)

	d.predictor = paddle.NewPredictor(config)
	d.input = d.predictor.GetInputTensors()[0]
	d.output = d.predictor.GetOutputTensors()[0]
}

func (d *DBDetector) Run(img gocv.Mat) [][][]int {
	oriH := img.Rows()
	oriW := img.Cols()
	data, resizeH, resizeW := d.preProcess.Run(img)
	st := time.Now()
	d.input.SetValue(data)
	d.input.Reshape([]int32{1, 3, int32(resizeH), int32(resizeW)})

	d.predictor.SetZeroCopyInput(d.input)
	d.predictor.ZeroCopyRun()
	d.predictor.GetZeroCopyOutput(d.output)

	log.Println("predict time: ", time.Since(st))

	ratioH, ratioW := float64(resizeH)/float64(oriH), float64(resizeW)/float64(oriW)
	boxes := d.postProcess.Run(d.output, oriH, oriW, ratioH, ratioW)
	elapse := time.Since(st)
	log.Println("det_box num: ", len(boxes), ", time elapse: ", elapse)
	return boxes
}
