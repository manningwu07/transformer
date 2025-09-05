//go:build accelerate

package main

// #cgo LDFLAGS: -framework Accelerate
import "C"
import (
    "gonum.org/v1/gonum/blas/blas64"
    "gonum.org/v1/gonum/blas/cgo"
)

// This file forces linking against Apple's Accelerate framework
// when you build with `-tags accelerate`.
func init() {
    blas64.Use(cgo.Implementation{})
}