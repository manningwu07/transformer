//go:build accelerate

package main

// #cgo LDFLAGS: -framework Accelerate
import "C"

// This file forces linking against Apple's Accelerate framework
// when you build with `-tags accelerate`.