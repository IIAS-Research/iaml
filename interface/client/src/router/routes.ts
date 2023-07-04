import { RouteRecordRaw } from 'vue-router'
import { gardeConnexion, gardeTableauDeBord } from 'src/router/gardes'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'racine',
    beforeEnter: gardeConnexion,
    children: [
      {
        path: '',
        name: 'racine-redirect',
        redirect: { name: 'connexion' },
      },
      {
        path: 'connexion',
        component: () => import('pages/autres/ConnexionPage.vue'),
        name: 'connexion'
      }
    ]
  },
  {
    path: '/tableau-de-bord',
    beforeEnter: gardeTableauDeBord,
    redirect: { name: 'pops' },
    children: [
      {
        name: 'tableau-de-bord', path: '', component: () => import('pages/IndexPage.vue')
      }
    ],
  },
  {
    path: '/:catchAll(.*)*',
    component: () => import('src/pages/autres/ErrorNotFound.vue'),
    name: 'erreur404'
  },
]

export default routes
