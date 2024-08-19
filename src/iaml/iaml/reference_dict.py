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
            'doi': None,
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
                'Louis Wehenkel'
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
                'Louis Wehenkel'
            ],
            'doi': 'https://doi.org/10.1007/s10994-006-6226-1',
            'publisher': 'Machine Learning Vol. 63 page 3--42'
        }
    ],
    'ActGaussianNb': [
        {
            'year': 1763,
            'name': None,
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
    ],
    'ActLinearDiscriminantAnalysis': [
        {
            'year': 1936,
            'name': 'The Use of Multiple Measurements in Taxonomic Problems',
            'authors': [
                'Sir Ronald Aylmer Fisher'
            ],
            'doi': '',
            'publisher': 'Annals of Eugenics Vol.7 page 179--188'
        }
    ],
    'ActLinearRegression' : [
        {
            'year': 1875,
            'name': 'No information provided',
            'authors': [
                'Sir Francis Galton'
            ],
            'doi': '',
            'publisher': ''
        }
    ],
    'ActLogisticRegression': [
        {
            'year': 1944,
            'name': 'Application of the Logistic Function to Bio-Essay',
            'authors': [
                'Joseph Berkson'
            ],
            'doi': 'https://doi.org/10.2307/2280041',
            'publisher': (
                'Journal of the American Statistical Association '
                'Vol. 39, No. 227, page 357--365'
            )
        },
        {
            'year': 1951,
            'name': 'Why I Prefer Logits to Probits',
            'authors': [
                'Joseph Berkson'
            ],
            'doi': 'https://doi.org/10.2307/3001655',
            'publisher': (
                'Biometrics '
                'Vol. 7, No. 4, page 327--339'
            )
        }
    ],
    'ActMLPClassifier': [
        {
            'year': 1989,
            'name': 'Connectionist Learning Procedures',
            'authors': [
                'Geoffrey E. Hinton'
            ],
            'doi': 'https://doi.org/10.1016/0004-3702(89)90049-0',
            'publisher': 'Artificial intelligence Vol. 40.1 page 185--234'
        },
        {
            'year': 2010,
            'name': 'Understanding the difficulty of training deep feedforward neural networks',
            'authors': [
                'Xavier Glorot',
                'Yoshua Bengio'
            ],
            'doi': '',
            'publisher': (
                'Proceedings of the Thirteenth International Conference on '
                'Artificial Intelligence and Statistics page 249--256'
            )
        }
    ],
    'ActMLPRegressor': [
        {
            'year': 1989,
            'name': 'Connectionist Learning Procedures',
            'authors': [
                'Geoffrey E. Hinton'
            ],
            'doi': 'https://doi.org/10.1016/0004-3702(89)90049-0',
            'publisher': 'Artificial intelligence Vol. 40.1 page 185--234'
        },
        {
            'year': 2010,
            'name': 'Understanding the difficulty of training deep feedforward neural networks',
            'authors': [
                'Xavier Glorot',
                'Yoshua Bengio'
            ],
            'doi': '',
            'publisher': (
                'Proceedings of the Thirteenth International Conference on '
                'Artificial Intelligence and Statistics page 249--256'
            )
        }
    ],
    'ActMultinomialNB': [
        {
            'year': 2008,
            'name': 'Introduction to Information Retrieval',
            'authors': [
                'Christopher D. Manning',
                'Prabhakar Raghaban',
                'Hinrich Schütze'
            ],
            'doi': '',
            'publisher': 'Cambridge University Press'
        }
    ],
    'ActQuadraticDiscriminantAnalysis': [
        {
            'year': 1965,
            'name': 'Geometrical and Statistical Properties of Systems of Linear Inequalities with Applications in Pattern Recognition',
            'authors': ['Thomas M. Cover'],
            'doi': 'https://doi.org/10.1109/PGEC.1965.264137',
            'publisher': 'IEEE Transactions on Electronic Computers Vol.EC-14 page 326--334'
            
        },
        {
            'year': 2016,
            'name': 'Linear vs. quadratic discriminant analysis classifier: a tutorial',
            'authors': ['Alaa Tharwat'],
            'doi': 'https://www.inderscienceonline.com/doi/abs/10.1504/IJAPR.2016.079050',
            'publisher': 'International Journal of Applied Pattern Recognition Vol.3, No.2 page 145--180'
        }
    ],
    'ActRandomForestRegressor': [
        {
            'year': 2001,
            'name': 'Random Forests',
            'authors': ['Leo Breiman'],
            'doi': 'https://doi.org/10.1023/A:1010933404324',
            'publisher': 'Machine Learning Vol.45 page 5--32'
        },
        {
            'year': 2006,
            'name': 'Extremely Randomized Trees',
            'authors': ['Pierre Geurts', 'Damien Ernst', 'Lous Wehenkel'],
            'doi': 'https://doi.org/10.1007/s10994-006-6226-1',
            'publisher': 'Machine Learning Vol.63 page 5--42'
        }
    ],
    'ActRandomForestRegressor': [
        {
            'year': 2001,
            'name': 'Random Forests',
            'authors': ['Leo Breiman'],
            'doi': 'https://doi.org/10.1023/A:1010933404324',
            'publisher': 'Machine Learning Vol.45 page 5--32'
        }
    ],
    'ActSGDRegressor': [
        {
            'year': 1951,
            'name': 'A Stochastic Approximation Method',
            'authors': [
                'Herbert Robbins',
                'Sutton Monro'    
            ],
            'doi': 'https://doi.org/10.1214/aoms/1177729586',
            'publisher': 'The annals of Mathematical Statistics Vol.22 No.3 page 400--407'
        }
    ],
    'ActSVMSVC': [
        {
            'year': 1999,
            'name': 'Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods',
            'authors': [
                'John C. Platt'
            ],
            'doi': None,
            'publisher': 'Microsoft Research'
        },
        {
            'year': 2001,
            'name': 'LIBSVM: A Library for Support Vector Machines',
            'authors': [
                'Chih-Chung Chang',
                'Chih-Jen Lin'
            ],
            'doi': 'https://doi.org/10.1145/1961189.1961199',
            'publisher': 'ACM Transactions on Intelligen Systems and Technology Vol.2 page 1--27'
        },
    ],
    'ActSVMSVR': [
        {
            'year': 1999,
            'name': 'Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods',
            'authors': [
                'John C. Platt'
            ],
            'doi': None,
            'publisher': 'Microsoft Research'
        },
        {
            'year': 2001,
            'name': 'LIBSVM: A Library for Support Vector Machines',
            'authors': [
                'Chih-Chung Chang',
                'Chih-Jen Lin'
            ],
            'doi': 'https://doi.org/10.1145/1961189.1961199',
            'publisher': 'ACM Transactions on Intelligen Systems and Technology Vol.2 page 1--27'
        },
    ]
}



# {
#             'year': 1763,
#             'name': 'No information provided',
#             'authors': [
#             ],
#             'doi': '',
#             'publisher': ''
#         }