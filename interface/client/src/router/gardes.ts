import { useUserStore } from 'src/stores/user-store'

export async function gardeConnexion() {
  const urlAvecTicket = window.location.search.match(/ST.+/)
  const userStore = useUserStore()
  if (userStore.token === null && urlAvecTicket?.length || 0 > 0 && urlAvecTicket !== null) {
    const ticketCas = urlAvecTicket[0]
    await userStore.demanderToken(ticketCas)
  }
  if (userStore.token !== null) {
    return { name: 'tableau-de-bord'}
  }
}

export function gardeTableauDeBord() {
  const userStore = useUserStore()
  if (userStore.token === null) {
    return { name: 'connexion' }
  }
}
