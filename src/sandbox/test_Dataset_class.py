import pandas as pd
import numpy as np
import inspect
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from iaml import Dataset

# Lecture Base de données
df = pd.read_csv('automl/python_project/src/perf_logger/tests_data/test.csv', sep=';')
print(df.shape)

# Instance de la classe Dataset
#dataset = Dataset(df, label_name = 'label')
dataset = Dataset(df)

#print(f"Nom de la classe : {dataset.__class__.__name__}")

membres = dir(dataset)
#print(len(membres))

# Filtrer les methodes de la classe
methodes = [membre for membre in membres if inspect.ismethod(getattr(dataset, membre))]

# Affichage de l'ensemble de methodes
print(" Les méthodes de la classe:")
for methode in methodes:
    print(f" - {methode}")


print('Affichage des colonnes actives:')
print(dataset.active_columns())

print('Dataset__find_differencies(df):')
print(dataset._Dataset__find_differencies(df))

#print('Acces aux paramères du constructeurs:')
#for cle, valeur in dataset._Dataset__data.items():
#    print(f' - {cle}: {valeur}')

#print('Accéder au train features de la dataset:')
#print(dataset._Dataset__data['train']['features'])

#print('Accéder au train disabled:')
#print(dataset._Dataset__data['train']['disabled'])

#print('Acéder au test features:')
#print(dataset._Dataset__data['test']['features'])

# Afficher le train data
#X_train = dataset._Dataset__data['train']['features'].copy(deep=True)
#print('X_train :')
#print(X_train)
#X_test = dataset._Dataset__data['test']['features']
#print('X_test :')
#print(X_test)
#y_train = dataset._Dataset__data['train']['labels']
#print('y_train :')
#print(y_train)
#y_test = dataset._Dataset__data['test']['labels']
#print('y_test : ')
#print(y_test)


print('Le type de données dans chaque colonne de la dataset chatgée:')
print(dataset.data_types(dataset._Dataset__data['train']['features']))

#print('Datatset.copy():')
#new_dataset = dataset.copy()

###--------------------------------------------------------------------------------------------------------------------------------------------
# Test des fonctions disable_columns et enable_columns
print(dataset._Dataset__data['train']['features'])
print('Suppression des colonnes Name, Sex, Age et Ticket :')
#dataset.disable_column('Name')
#dataset.disable_column('Sex')
#dataset.disable_column('Age')
#dataset.disable_column('Ticket')
##dataset.dc('Name')
##dataset.dc('Sex')
##dataset.dc('Age')
##dataset.dc('Ticket')
##print(dataset._Dataset__data['train']['features'])
##print(dataset.types_de_données(dataset._Dataset__data['train']['features']))
###
##print('Les colonnes désactivées:')
##print(dataset._Dataset__data['train']['disabled'])
###
##print('activation des colonnes disactivées :')
###dataset.enable_column('Name')
###dataset.enable_column('Sex')
###dataset.enable_column('Age')
###dataset.enable_column('Ticket')
##dataset.ec('Name')
##dataset.ec('Sex')
##dataset.ec('Age')
##dataset.ec('Ticket')
##### Petit problème dans l'affichage--------------------------------------------------------------
print(dataset._Dataset__data['train']['features'])
###----------------------------------------------------------------------------------------------------
##print(dataset._Dataset__data['train']['features']['Name'])
#
## Propriété train_data
#print('Propriété data_train', dataset.train_data)
#
## Propriété X_train
#print('Propriété X_train : ', dataset.X_train)
#
## Propriété __test_data
##print('Propriété __test_data', dataset._Dataset__test_data)
#
## Propriété __X_test
#print('Propriété __X_test :', dataset._Dataset__X_test)
#
## Propriété train_labels
##print('Propriété train_labels :', dataset.train_labels)
#
## Propriété y_train
#print('Propriété y-train :', dataset.y_train)
#
## Propriété __y_test (Prp privée)
#print('Propriéte __y_test :', dataset._Dataset__y_test)

#--------------------------------------------------------------------------------------------------------------
# A faire : Test des fonction reset_label sans la fonction _merge()
print("Le dataset._Dataset__data['train]['features']  avant la fonction reset_label :")
print(dataset._Dataset__data['train']['features'])
print("Le dataset._Dataset__data['train']['labels]  avant la fonction reset_label:")
print(dataset._Dataset__data['train']['labels'])
print("La fonction reset_labels :")
dataset.res_label()
print("Le dataset apres la fonction rl :")
print(dataset._Dataset__data['train']['features'])
print(dataset._Dataset__data['train']['labels'])
dataset.set_label('label')
print("Le dataset apres nouvel ajout du label :")
print(dataset._Dataset__data['train']['features'])
print(dataset._Dataset__data['train']['labels'])

# -------------------------------------------------------------------------------------------


