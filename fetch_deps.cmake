# Eigen is used for feedback matrix operations
find_package(
    Eigen3
    3.4
    REQUIRED
    NO_MODULE
)

# PFFFT is used in the partitioned convolution code
find_package(pffft REQUIRED)

# PFFFT does not support small FFT sizes (< 32) so KISSFFT is used for those cases. So far, this is only used for
# building Circulant Matrix.
find_package(kissfft REQUIRED)
