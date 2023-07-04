# Agagou
Documentation template Agagou

***
## 🔧 1 - Installation

Vous aurez besoin des prérequis des outils suivant :

- Quasar CLI ([lien](https://quasar.dev/start/quasar-cli))
- Ruby on Rails ([lien](https://guides.rubyonrails.org/getting_started.html))
- PostgreSQL ([lien](https://www.postgresql.org/download/))

Une fois que tout est installé et configuré :

- Lancer `git init`
- Cloner le projet `git clone https://gitlab.example.invalid/
- Dans le répertoire client/ : `yarn` ou `npm install`
- Dans le répertoire serveur/ : `bundle`
- Créer la base de données pour l'environement de developpement avec postgresql : `createdb agagou_dev`
- Appliquer les migrations (dans le répertoire serveur/) : `rails db:migrate`
- Ajouter les clées secrètes suivantes : `EDITOR="nano" rails credentials:edit` : 
- Ajouter une clé `jwt_secret`


***
## 🚀 2 - Lancement

- Lancer le server : dans le répertoire server/ : `rails s`
- Lance le client : dans le répertoire client/ : `quasar d`

***
## 🎥 3 - Explications spécifiques

TODO

***
## 📌 4 - Outils utilisés / Infos utiles

Système d'authentification avec le serveur CAS.

Client:
- Quasar 2.6 (core) ([lien](https://quasar.dev/))
- Pinia 2.0.11 (store) ([lien](https://pinia.vuejs.org/))
- Axios 0.21.1 (request) ([lien](https://axios-http.com/fr/docs/intro))
- Vue-router 4.0 (router) ([lien](https://router.vuejs.org/))

Serveur:
- Ruby on Rails 7.0.4 (core) ([lien](https://rubyonrails.org/))
- Pundit (policies) ([lien](https://github.com/varvet/pundit#pundit))
- JWT (Jwt token) ([lien](https://github.com/jwt/ruby-jwt#jwt))

Serveur de production:
- Pas encore !

Extensions Visual Studio Code recommendées:
Volar ([lien](https://marketplace.visualstudio.com/items?itemName=Vue.volar))

Schéma base de données :
TODO