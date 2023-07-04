<script setup>
import { ref } from 'vue';

// refs
const user = ref({
  nom: 'Moulard',
  prenom: 'Didier'
})

// functions
function modifierNom() {
  user.value.nom = 'Rudy'
}
</script>

<template>
  {{ user }}
  <q-btn label="click me" @click="modifierNom" />
</template>
