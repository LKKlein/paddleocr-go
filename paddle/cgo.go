// +build !customenv

package paddle

/*
#cgo CXXFLAGS: -std=c++11 -w -fPIC -I../paddle_cxx/paddle/include
#cgo LDFLAGS: -L${SRCDIR}/../paddle_cxx/paddle/lib -Wl,-rpath,$ORIGIN/paddle_cxx/paddle/lib -lpaddle_fluid -ldl -lrt -lz -lm -lpthread
*/
import "C"
