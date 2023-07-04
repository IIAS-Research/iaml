import { defineStore } from 'pinia'
import { ITokenDecode } from 'components/models'
import { Cookies } from 'quasar'
import { api } from 'src/boot/axios'
import jwt_decode from 'jwt-decode'

const COOKIE_TOKEN = 'agagou-accessToken'
const COOKIE_ROUTE_RETOUR = 'agagou-nomRouteRetour'
const CAS_URL = 'https://sso.example.invalid/login?service='
const SAME_SITE = 'Lax'
const COOKIE_CHEMIN = '/'


export const useUserStore = defineStore('user', {
  state: () => ({
    utilisateur: <string | null>null,
    token: <string | null>null,
    nomRouteRetour: <string | null>null,
    tokenDecode: <ITokenDecode | null>null,
    estAdministrateur: false,
    nomComplet: <string | null>null,
    privileges: <string[]>[],
  }),
  getters: {
    tokenExpire(): boolean {
      if (this.tokenDecode !== null) {
        return new Date > new Date(((this.tokenDecode.exp) * 1000))
      }
      return false
    }
  },
  actions: {
    tokenDansLesCookies(): boolean {
      return Cookies.has(COOKIE_TOKEN)
    },
    demandeTicketCas(nomRouteRetour: string) {
      this.nomRouteRetour = nomRouteRetour
      Cookies.set(COOKIE_ROUTE_RETOUR, this.nomRouteRetour, { sameSite: SAME_SITE, path: COOKIE_CHEMIN })
      window.location.href = CAS_URL + window.location.origin
    },
    mettreEnPlaceTokenDepuisCookies() {
      const token: string = Cookies.get(COOKIE_TOKEN)
      this.authentification(token)
    },
    purge() {
      this.utilisateur = null
      this.token = null
      this.tokenDecode = null
      this.estAdministrateur = false
      this.nomRouteRetour = null
      this.nomComplet = null
      Cookies.remove(COOKIE_TOKEN, { path: COOKIE_CHEMIN })
      Cookies.remove(COOKIE_ROUTE_RETOUR, { path: COOKIE_CHEMIN })
    },
    authentification(token: string) {
      this.token = token
      this.tokenDecode = jwt_decode(token)
      this.utilisateur = this.tokenDecode?.id_res ?? null
      this.estAdministrateur = this.tokenDecode?.admin ?? false
      this.nomComplet = this.tokenDecode?.nom_complet ?? null
      Cookies.set(COOKIE_TOKEN, token, { sameSite: SAME_SITE, path: COOKIE_CHEMIN })
    },
    async demanderToken(casTicket: string) {
      try {
        const reponseAuthentification = await api.post('login.json', { ticket: casTicket, service: window.location.origin })
        this.authentification(reponseAuthentification.data.token)
      } catch (error) {
        console.error(error)
      }
    }
  }
})
