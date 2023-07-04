<script setup lang="ts">
import { useUserStore } from 'stores/user-store'
import { useRoute } from 'vue-router'

// consts
const userStore = useUserStore()
const route = useRoute()

// functions
function connexion() {
  userStore.demandeTicketCas(route.name as string)
}
</script>

<template>
    <div class="flex flex-center column">
      <h6>Vous devez être connecté pour accéder aux fonctionnalités de <i>Trucmush</i></h6>
      <q-btn color="secondary" label="Connexion" @click="connexion" />
    </div>
</template>
