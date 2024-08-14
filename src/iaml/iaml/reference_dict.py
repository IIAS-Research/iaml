"""
REF is a dictionnary containing all references for all steps available.
It is called inside the Step constructor to add each references to specific step
"""


REF = {
    'ActAdaBoostRegressor': [
        {
            'year': 1995,
            'name': (
                'A desicion-theoretic generalization of on-line learning '
                'and an application to boosting'
            ),
            'authors': [
                'Yoav Freund',
                'Robert E. Schapire'
            ],
            'doi': 'https://doi.org/10.1007/3-540-59119-2_166',
            'publisher': 'Springer, Berlin, Heidelberg'
        }
    ],
    'ActARDRegression': [
        {
            'year': 1996,
            'name': 'Bayesian Non-Linear Modeling for the Prediction Competition',
            'authors': ['David J. C. MacKay'],
            'doi': 'https://doi.org/10.1007/978-94-015-8729-7_18',
            'publisher': 'Springer, Dordrecht'
        }
    ],
    'ActAutoSKLearn': [
        {
            'year': 2015,
            'name': 'Efficient and Robust Automated Machine Learning',
            'authors': [
                'Matthias Feurer',
                'Aaron Klein',
                'Katharina Eggensperger',
                'Jost Tobias Springenberg',
                'Manuel Blum',
                'Frank Hutter'
            ],
            'doi': 'https://doi.org/10.1007/978-3-030-05318-5_6 (V2)',
            # Store pages for book ? (2962--2970), Not always present in NeurIPS doc'
            'publisher': 'Advances in Neural Information Processing Systems 28 (NeurIPS 2015)'
        },
        {
            'year': 2020,
            'name': 'Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning',
            'authors': [
                'Matthias Feurer',
                'Katharina Eggensperger',
                'Stefan Falkner',
                'Marius Lindauer',
                'Frank Hutter'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.2007.04074',
            'publisher': 'Journal of Machine Learning Research 23(261), 2022'
        }
    ],
    
    'ActCatBoostRegressor': [
        {
            'year': 2017,
            'name': 'CatBoost: unbiased boosting with categorical features',
            'authors': [
                'Liudmila Prokhorenkova',
                'Gleb Gusev',
                'Aleksandr Vorobev',
                'Anna Veronika Dorogush',
                'Andrey Gulin'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.1706.09516',
            'publisher': 'Advances in Neural Information Processing Systems 31 (NeurIPS 2018)'
        }
    ]
    'ActPowerTransformer': [],
    'ActRBFSampler': [],
}
