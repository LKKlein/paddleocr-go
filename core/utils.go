package core

import (
	"log"
	"os"

	"github.com/LKKlein/gocv"
)

func getString(args map[string]interface{}, key string, dv string) string {
	if f, ok := args[key]; ok {
		return f.(string)
	}
	return dv
}

func getFloat64(args map[string]interface{}, key string, dv float64) float64 {
	if f, ok := args[key]; ok {
		return f.(float64)
	}
	return dv
}

func getInt(args map[string]interface{}, key string, dv int) int {
	if i, ok := args[key]; ok {
		return i.(int)
	}
	return dv
}

func getBool(args map[string]interface{}, key string, dv bool) bool {
	if b, ok := args[key]; ok {
		return b.(bool)
	}
	return dv
}

func ReadImage(image_path string) gocv.Mat {
	img := gocv.IMRead(image_path, gocv.IMReadColor)
	if img.Empty() {
		log.Printf("Could not read image %s\n", image_path)
		os.Exit(1)
	}
	return img
}

func clip(value, min, max int) int {
	if value <= min {
		return min
	} else if value >= max {
		return max
	}
	return value
}

func minf(data []float32) float32 {
	v := data[0]
	for _, val := range data {
		if val < v {
			v = val
		}
	}
	return v
}

func maxf(data []float32) float32 {
	v := data[0]
	for _, val := range data {
		if val > v {
			v = val
		}
	}
	return v
}

func mini(data []int) int {
	v := data[0]
	for _, val := range data {
		if val < v {
			v = val
		}
	}
	return v
}

func maxi(data []int) int {
	v := data[0]
	for _, val := range data {
		if val > v {
			v = val
		}
	}
	return v
}
