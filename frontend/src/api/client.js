import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

// Token stored in memory — never in localStorage
let _token = null
export const setToken = (t) => { _token = t }
export const clearToken = () => { _token = null }

api.interceptors.request.use((config) => {
  if (_token) config.headers.Authorization = `Bearer ${_token}`
  return config
})

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      clearToken()
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export default api
