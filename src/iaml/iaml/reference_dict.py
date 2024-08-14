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
            'publisher': (
                'Advances in Neural Information Processing Systems 28 (NeurIPS 2015) '
                'page 2962--2970'
            )
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
    'ActBernoulliNb': [
        {
            'year': 1998,
            'name': 'A Comparison of Event Models for Naive Bayes Text Classification',
            'authors': [
                'Andrew McCallum',
                'Kamal Nigam'
            ],
            'doi': '',
            'publisher': (
                'AAAI-98 workshop on learning for text categorization, '
                '752, page 41--48. (1998)'
            )
        },
        {
            'year': 2006,
            'name': 'Spam Filtering with Naive Bayes - Which Naive Bayes?',
            'authors': [
                'Vangelis Metsis',
                'Ion Androutsopoulos',
                'Georgios Paliouras'
            ],
            'doi': '',
            'publisher': (
                'The Third Conference on Email and Anti-Spam 2006 (CEAS)'
            )
        }
    ],
    'ActCatBoost': [
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
    ],
    'ActExtraTreesClassifier': [
        {
            'year': 2006,
            'name': 'Extremely randomized trees',
            'authors': [
                'Pierre Geurts',
                'Damien Ernst',
                'Wehenkel'
            ],
            'doi': 'https://doi.org/10.1007/s10994-006-6226-1',
            'publisher': 'Machine Learning Vol. 63 page 3--42'
        }
    ],
    'ActExtraTreesRegressor': [
        {
            'year': 2006,
            'name': 'Extremely randomized trees',
            'authors': [
                'Pierre Geurts',
                'Damien Ernst',
                'Wehenkel'
            ],
            'doi': 'https://doi.org/10.1007/s10994-006-6226-1',
            'publisher': 'Machine Learning Vol. 63 page 3--42'
        }
    ],
    'ActGaussianNb': [
        {
            'year': 1763,
            'name': 'No info provided',
            'authors': [
            ],
            'doi': '',
            'publisher': ''
        }
    ],
    'ActGaussianProcessRegressor': [
        {
            'year': 2006,
            'name': 'Gaussian Processes for Machine Learning',
            'authors': [
                'Carl Edward Rasmussen',
                'Christopher K. I. Williams'
            ],
            'doi': '',
            'publisher': 'MIT Press 2006'
        }
    ],
    'ActHistGradientBoostingRegressor': [
        {
            'year': 2006,
            'name': 'Gaussian Processes for Machine Learning',
            'authors': [
                'Carl Edward Rasmussen',
                'Christopher K. I. Williams'
            ],
            'doi': '',
            'publisher': 'MIT Press 2006'
        }
    ],
    'ActKNNRegressor': [
        {
            'year': 1951,
            'name': 'Discriminatory Analysis, Nonparametric Discrimination: Consistency Properties',
            'authors': [
                'Evelyn Fix',
                'Joseph Lawson Hodges Jr.'
            ],
            'doi': '',
            'publisher': 'Technical Report 4, USAF School of Aviation Medicine, Randolph Field'
        },
        {
            'year': 1967,
            'name': 'Nearest neighbor pattern classification',
            'authors': [
                'Thomas M. Cover',
                'Peter E. Hart'
            ],
            'doi': 'https://doi.org/10.1109/TIT.1967.1053964',
            'publisher': 'IEEE Transactions on Information Theory. 13: page 21--27'
        }
    ],
    'ActKNN': [
        {
            'year': 1951,
            'name': 'Discriminatory Analysis, Nonparametric Discrimination: Consistency Properties',
            'authors': [
                'Evelyn Fix',
                'Joseph Lawson Hodges Jr.'
            ],
            'doi': '',
            'publisher': 'Technical Report 4, USAF School of Aviation Medicine, Randolph Field'
        },
        {
            'year': 1967,
            'name': 'Nearest neighbor pattern classification',
            'authors': [
                'Thomas M. Cover',
                'Peter E. Hart'
            ],
            'doi': 'https://doi.org/10.1109/TIT.1967.1053964',
            'publisher': 'IEEE Transactions on Information Theory. 13: page 21--27'
        }
    ]
}
