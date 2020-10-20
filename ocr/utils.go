package ocr

import (
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

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

func DownloadFile(filepath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	log.Println("[download_file] from: ", url, " to: ", filepath)
	return err
}

func IsPathExist(path string) bool {
	if _, err := os.Stat(path); err == nil {
		return true
	} else if os.IsNotExist(err) {
		return false
	}
	return false
}

func DownloadModel(modelDir string, modelPath string) (string, error) {
	if modelPath != "" && (strings.HasPrefix(modelPath, "http://") ||
		strings.HasPrefix(modelPath, "ftp://") || strings.HasPrefix(modelPath, "https://")) {
		reg := regexp.MustCompile("^(http|https|ftp)://[^/]+/(.+)")
		suffix := reg.FindStringSubmatch(modelPath)[2]
		if strings.HasPrefix(suffix, "tpflow/") {
			suffix = suffix[7:]
		}
		outPath := filepath.Join(modelDir, suffix)
		outDir := filepath.Dir(outPath)
		if !IsPathExist(outDir) {
			os.MkdirAll(outDir, os.ModePerm)
		}

		err := DownloadFile(outPath, modelPath)
		if err != nil {
			return "", err
		}
		return outPath, nil
	}
	return modelPath, nil
}