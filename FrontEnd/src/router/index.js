import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'home',
    component: () => import(/* webpackChunkName: "about" */ '../views/HomeView.vue')
  },
  {
    path: '/demo',
    name: 'demo',
    component: () => import(/* webpackChunkName: "about" */ '../views/DemoView.vue')
  },
  {
    path: '/upload',
    name: 'upload',
    component: () => import(/* webpackChunkName: "about" */ '../views/UploadView.vue')
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router
