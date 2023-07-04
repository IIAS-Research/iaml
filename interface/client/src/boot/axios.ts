import { boot } from 'quasar/wrappers'
import axios, { AxiosInstance } from 'axios'
import { useUserStore } from 'stores/user-store'

declare module '@vue/runtime-core' {
  interface ComponentCustomProperties {
    $axios: AxiosInstance
  }
}
const apiUrl = process.env.api
const EXCEPTION_PAS_CONNECTE = 'PasConnecté'
const NOM_ROUTE_CONNEXION = 'connexion'
const NOM_ROUTE_ERREUR500 = 'erreur500'
const NOM_ROUTE_ERREUR401 = 'erreur401'
const api = axios.create({ baseURL: apiUrl })

export default boot(({ router }) => {
  const userStore = useUserStore()

  // Avant chaque requête
  api.interceptors.request.use(
    function (config) {
      if (userStore.utilisateur !== null) {
        if (userStore.tokenExpire === true) {
          userStore.purge()
          return config
        }
        config.headers.Authorization = userStore.token
      }
      return config
    },
    function (error) {
      console.error(error.message)
    }
  )

  // Réponses
  api.interceptors.response.use(
    function (response) {
      return response
    },
    function (error) {
      console.error(error)
      if (error.response?.data.exception === EXCEPTION_PAS_CONNECTE) {
        userStore.purge()
        router.push({ name: NOM_ROUTE_CONNEXION })
        return
      }
      if (error.response?.status === 500) {
          router.push({ name: NOM_ROUTE_ERREUR500 })
      }
      if (error.response?.status === 401) {
          router.push({ name: NOM_ROUTE_ERREUR401 })
          return
        
      }
    }
  )

  // Router
  router.beforeEach((to, from, next) => {
    if (userStore.tokenDansLesCookies() === true) {
      userStore.mettreEnPlaceTokenDepuisCookies()
    }
    if (userStore.tokenExpire === true) {
      userStore.purge()
    }
    next()
  })
})

export { api }
