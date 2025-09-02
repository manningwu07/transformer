# Link against Accelerate
export CGO_LDFLAGS="-framework Accelerate"

# Use all CPU cores
export GOMAXPROCS=$(sysctl -n hw.ncpu)

# Tell Accelerate to use all threads
export VECLIB_MAXIMUM_THREADS=$(sysctl -n hw.ncpu)

# Parallelize attention heads in your code
export HEAD_PAR=1