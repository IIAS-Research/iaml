<script setup>
import { ref, onMounted } from 'vue'


// Props
// const props = defineProps({
//   id: Number,
// })
// const { id } = toRefs(props)

// refs
// const user = ref({
//   nom: 'Moulard',
//   prenom: 'Didier'
// })

// functions
// function modifierNom() {
//   user.value.nom = 'Rudy'
// }


const el = ref()

onMounted(() => {
  el.value // <div>
})

</script>

<template>
  <div>
    <p>Coucou tout le monde</p>
    <p>{{ $route.params.id }}</p>
    <!-- <q-btn label="click me" @click="modifierNom" /> -->
  </div>
</template>
