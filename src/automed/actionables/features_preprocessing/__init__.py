"""
Usually last step before predictor, try features decomposition or aggregation.
Example : PCA
"""
from .act_pca import ActPCA
from .act_kernel_pca import ActKernelPCA
from .act_feature_agglomeration import ActFeatureAgglomeration
# from .act_nystroem import ActNystroem
# from .act_polynomial_features import ActPolynomialFeatures
from .act_rbf_sampler import ActRBFSampler
from .act_select_percentile import ActSelectPercentile
