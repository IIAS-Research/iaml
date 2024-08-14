from src.iaml.iaml.reference import Reference
from src.iaml.iaml.actionables.predictors.act_ada_boost_regressor import ActAdaBoostRegressor
from src.iaml.iaml.actionables.predictors.act_catboost_regressor import ActCatBoostRegressor
from src.iaml.iaml.actionables.predictors.act_ard_regression import ActARDRegression
from src.iaml.iaml.actionables.features_preprocessing.act_power_transformer import ActPowerTransformer
from src.iaml.iaml.actionables.features_preprocessing.act_rbf_sampler import ActRBFSampler
from src.iaml.iaml.iaml_pipeline import IAMLPipeline

# properties = {
#     'year': 1995,
#     'name': (
#         'A desicion-theoretic generalization of on-line learning '
#         'and an application to boosting'
#     ),
#     'authors': [
#         'Yoav Freund',
#         'Robert E. Schapire'
#     ],
#     'doi': 'https://doi.org/10.1007/3-540-59119-2_166',
#     'publisher': 'Springer, Berlin, Heidelberg'
# }

# ref = Reference(properties)

# properties = {
#     'year': 2017,
#     'name': 'CatBoost: unbiased boosting with categorical features',
#     'authors': [
#         'Liudmila Prokhorenkova',
#         'Gleb Gusev',
#         'Aleksandr Vorobev',
#         'Anna Veronika Dorogush',
#         'Andrey Gulin'
#     ],
#     'doi': 'https://doi.org/10.48550/arXiv.1706.09516',
#     'publisher': 'Advances in Neural Information Processing Systems'
# }
# ref2 = Reference(properties)


# references = [ref, ref2] * 30
# for i, refe in enumerate(references):
#     print(f"[{i+1:>{len(str(len(references)))}}]  {refe}")


# a = ActAdaBoostRegressor()
# # c = ActCatBoostRegressor()
# # b = ActARDRegression()

# print(a.citation()[0])

# print(c.citation()[0])
# print(b.citation()[0])

pipeline = IAMLPipeline(
    steps=[
        ('1', ActAdaBoostRegressor()),
        ('2', ActPowerTransformer()),
        ('3', ActRBFSampler()),
    ],
    estimator_type='regressor'
)


print("----------")
# print(pipeline.last_stage_candidates)



print(pipeline.citation())


# a = ActAdaBoostRegressor()
# print(a.citation()[0])
# b = ActPowerTransformer()
# print(b.citation()[0])