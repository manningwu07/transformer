# 

export PATH=$PATH:$(go env GOPATH)/bin
# Link against Accelerate
export CGO_LDFLAGS="-framework Accelerate"
# Use all CPU cores
export GOMAXPROCS=$(sysctl -n hw.ncpu)
# Tell Accelerate to use all threads
export VECLIB_MAXIMUM_THREADS=$(sysctl -n hw.ncpu)
# Parallelize attention heads
export HEAD_PAR=1
export GOMAXPROCS=$(sysctl -n hw.ncpu)