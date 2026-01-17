"""
Usually last step before predictor, try features decomposition or aggregation.
Example : PCA
"""
from .act_pca import ActPCA
from .act_kernel_pca import ActKernelPCA
from .act_feature_agglomeration import ActFeatureAgglomeration
from .act_nystroem import ActNystroem
# # from .act_polynomial_features import ActPolynomialFeatures # Disable -> Use so much memory
from .act_rbf_sampler import ActRBFSampler
from .act_select_percentile import ActSelectPercentile
from .act_power_transformer import ActPowerTransformer
from .act_quantile_transformer import ActQuantileTransformer
from .act_truncated_svd import ActTruncatedSVD
from .act_fast_ica import ActFastICA
from .act_sparse_random_projection import ActSparseRandomProjection
from .act_k_bins_discretizer import ActKBinsDiscretizer
from .act_log_transformer import ActLogTransformer
from .act_k_means_features import ActKMeansFeatures
