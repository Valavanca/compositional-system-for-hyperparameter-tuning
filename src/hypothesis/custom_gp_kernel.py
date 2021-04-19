from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
# [1] Kernel with parameters given in GPML book. Mauna Loa CO2 data
k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = 2.4**2 * RBF(length_scale=90.0) \
    * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
k3 = 0.66**2 \
    * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18**2 * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)  # noise terms

KERNEL_GPML = k1 + k2 + k3 + k4

# [2]
KERNEL_SIMPLE = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))


# [3] GPR on Mauna Loa CO2 data
# The objective is to model the CO2 concentration as a function of the time t.
KERNEL_MAUNA = 34.4**2 * RBF(length_scale=41.8) \
            + 3.27**2 * RBF(length_scale=180) * ExpSineSquared(length_scale=1.44, periodicity=1) \
            + 0.446**2 * RationalQuadratic(alpha=17.7, length_scale=0.957) \
            + 0.197**2 * RBF(length_scale=0.138) + WhiteKernel(noise_level=0.0336)
