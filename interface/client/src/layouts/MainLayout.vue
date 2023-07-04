<script setup lang="ts">
import { computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { useUserStore } from 'src/stores/user-store';

// consts
const route = useRoute();
const router = useRouter();
const userStore = useUserStore();

// computeds
const disabled = computed(() => {
  return userStore.tokenDecode === null;
});

// functions
function deconnexion() {
  userStore.purge();
  router.push({ name: 'connexion' });
}
</script>

<template>
  <q-layout view="lHh Lpr fff" class="bg-grey-1">
    <q-header elevated class="bg-primary" height-hint="64">
      <q-toolbar class="GPL__toolbar" style="height: 64px">
        <q-toolbar-title
          v-if="$q.screen.gt.sm"
          shrink
          class="row items-center no-wrap"
        >
          <img src="/images/chu-logo.png" id="icon-chu" />
          <span class="q-ml-sm text-weight-bold">HomePOP</span>
        </q-toolbar-title>

        <q-space />


        <div class="q-gutter-sm row items-center no-wrap">
          <q-btn-dropdown :disable="disabled" round flat>
            <template v-slot:label>
              <q-avatar size="26px" icon="person" color="secondary" />
            </template>

            <q-list>
              <q-item>
                <q-item-section>
                  {{ userStore.nomComplet }}
                </q-item-section>
              </q-item>

              <q-separator />

              <q-item clickable v-close-popup :to="{ name: 'majs' }">
                <q-item-section avatar>
                  <q-icon name="update" />
                </q-item-section>
                <q-item-section>
                  <q-item-label>Mises à jour</q-item-label>
                </q-item-section>
              </q-item>
              <q-item clickable v-close-popup :to="{ name: 'settings' }">
                <q-item-section avatar>
                  <q-icon name="settings" />
                </q-item-section>
                <q-item-section>
                  <q-item-label>Préférences</q-item-label>
                </q-item-section>
              </q-item>
              <q-item clickable v-close-popup @click="deconnexion()">
                <q-item-section avatar>
                  <q-icon name="logout" />
                </q-item-section>
                <q-item-section>
                  <q-item-label>Déconnexion</q-item-label>
                </q-item-section>
              </q-item>
            </q-list>
          </q-btn-dropdown>
        </div>
      </q-toolbar>
    </q-header>

    <q-page-container class="GPL__page-container">
      <q-page class="column q-px-md q-py-sm">
        <router-view :key="route.path" v-slot="{ Component }">
          <transition
            appear
            enter-active-class="animated fadeIn"
            leave-active-class="animated fadeOut"
          >
            <component :is="Component" />
          </transition>
        </router-view>
      </q-page>
    </q-page-container>
  </q-layout>
</template>

<style lang="sass">
.GPL
  &__toolbar
    height: 64px
  &__toolbar-input
    width: 35%
  &__drawer-item
    line-height: 24px
    border-radius: 0 24px 24px 0
    margin-right: 12px
    .q-item__section--avatar
      padding-left: 12px
      .q-icon
        color: #5f6368
    .q-item__label:not(.q-item__label--caption)
      color: #3c4043
      letter-spacing: .01785714em
      font-size: .875rem
      font-weight: 500
      line-height: 1.25rem
    &--storage
      border-radius: 0
      margin-right: 0
      padding-top: 24px
      padding-bottom: 24px
  &__side-btn
    &__label
      font-size: 12px
      line-height: 24px
      letter-spacing: .01785714em
      font-weight: 500
</style>
