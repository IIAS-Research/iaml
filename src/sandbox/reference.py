import sys
sys.path.insert(0, '../')

from iaml.iaml.reference import Reference
from iaml.iaml.actionables.predictors.act_ada_boost_regressor import ActAdaBoostRegressor
from iaml.iaml.actionables.predictors.act_catboost_regressor import ActCatBoostRegressor
from iaml.iaml.actionables.predictors.act_ard_regression import ActARDRegression
from iaml.iaml.actionables.features_preprocessing.act_power_transformer import ActPowerTransformer
from iaml.iaml.actionables.features_preprocessing.act_rbf_sampler import ActRBFSampler
from iaml.iaml.iaml_pipeline import IAMLPipeline
from iaml.iaml.dataset import Dataset
from iaml.iaml import splitters

# # properties = {
# #     'year': 1995,
# #     'name': (
# #         'A desicion-theoretic generalization of on-line learning '
# #         'and an application to boosting'
# #     ),
# #     'authors': [
# #         'Yoav Freund',
# #         'Robert E. Schapire'
# #     ],
# #     'doi': 'https://doi.org/10.1007/3-540-59119-2_166',
# #     'publisher': 'Springer, Berlin, Heidelberg'
# # }

# # ref = Reference(properties)

# # properties = {
# #     'year': 2017,
# #     'name': 'CatBoost: unbiased boosting with categorical features',
# #     'authors': [
# #         'Liudmila Prokhorenkova',
# #         'Gleb Gusev',
# #         'Aleksandr Vorobev',
# #         'Anna Veronika Dorogush',
# #         'Andrey Gulin'
# #     ],
# #     'doi': 'https://doi.org/10.48550/arXiv.1706.09516',
# #     'publisher': 'Advances in Neural Information Processing Systems'
# # }
# # ref2 = Reference(properties)


# # references = [ref, ref2] * 30
# # for i, refe in enumerate(references):
# #     print(f"[{i+1:>{len(str(len(references)))}}]  {refe}")


# # a = ActAdaBoostRegressor()
# # # c = ActCatBoostRegressor()
# # # b = ActARDRegression()

# # print(a.references[0])

# # print(c.references[0])
# # print(b.references[0])

# pipeline = IAMLPipeline(
#     steps=[
#         ('1', ActAdaBoostRegressor()),
#         ('2', ActPowerTransformer()),
#         ('3', ActRBFSampler()),
#     ],
#     estimator_type='regressor'
# )


# print("----------")
# # print(pipeline.last_stage_candidates)



# print(pipeline.bibliography(structured=False))

# print(pipeline.bibliography(structured=True))

import pandas as pd

file = '/stockage/hr/automl/src/perf_logger/tests_data/body_signal_of_smoking.csv'
try:
    df1 = pd.read_csv(file, sep=",")
except:
    df1 = pd.DataFrame()

# print(df1.head())

target = 'label'

X = df1.drop('label', axis=1)
y = df1['label']
# Test with groups_columns

# print(X.head())
print(len(X))

dataset = Dataset(X, y, groups_columns=['age'])
# dataset = Dataset(X, y)
print(len(dataset.X))

# print(dataset.X.head())


X = splitters.random_splitter(dataset)
for train, test in X:
    print(train.y)
    print(train.X.head())
    
X_train = train.X
X_train['label'] = train.y
print(X_train.head())