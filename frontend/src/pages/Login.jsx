import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const FEATURES = [
  'OCR local — Tesseract + OpenCV',
  'Validação SEFAZ em tempo real',
  'Zero envio de dados a terceiros',
]

// Dot grid rows × cols for the brand panel background
const GRID_DOTS = Array.from({ length: 8 * 14 }, (_, i) => i)

export default function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(email, password)
      navigate('/')
    } catch {
      setError('E-mail ou senha incorretos')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex">
      {/* Brand panel */}
      <div className="hidden lg:flex w-[600px] flex-shrink-0 bg-accent flex-col justify-between p-16 relative overflow-hidden">
        {/* Animated dot grid */}
        <div
          className="absolute inset-0 grid pointer-events-none"
          style={{ gridTemplateColumns: 'repeat(14, 1fr)', padding: '24px', gap: '28px' }}
        >
          {GRID_DOTS.map((i) => (
            <div
              key={i}
              className="brand-dot w-1 h-1 rounded-full bg-white"
              style={{
                '--dur': `${2.5 + (i % 7) * 0.4}s`,
                '--delay': `${(i % 11) * 0.18}s`,
              }}
            />
          ))}
        </div>

        {/* Logo */}
        <div className="flex items-center gap-2.5 relative z-10">
          <div className="w-8 h-8 bg-white flex-shrink-0" />
          <span className="font-display font-semibold text-lg text-white">NotaFiscal</span>
        </div>

        {/* Content */}
        <div className="flex flex-col gap-10 relative z-10">
          <div>
            <h1 className="font-display font-semibold text-[40px] text-white leading-tight tracking-tight">
              Controle total<br />das suas notas<br />fiscais.
            </h1>
            <p className="mt-4 text-white/75 text-sm">NF-e · NFC-e · NFS-e · Comprovantes PIX</p>
          </div>
          <div className="flex flex-col gap-3">
            {FEATURES.map((f, i) => (
              <div key={f} className="flex items-center gap-3 group">
                <div
                  className="w-4 h-4 flex-shrink-0 transition-colors duration-300"
                  style={{ background: 'rgba(255,255,255,0.25)' }}
                />
                <span className="text-white text-sm">{f}</span>
              </div>
            ))}
          </div>
        </div>

        <p className="text-white/40 text-xs relative z-10">© {new Date().getFullYear()} NotaFiscal</p>
      </div>

      {/* Form panel */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-sm">
          <h2 className="font-display font-semibold text-[40px] tracking-tight">Entrar</h2>
          <p className="text-muted text-sm mt-2 mb-8">Acesse sua conta para continuar</p>

          <form onSubmit={handleSubmit} className="flex flex-col gap-5">
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium uppercase tracking-wide text-muted">E-mail</label>
              <input
                type="email" value={email} onChange={(e) => setEmail(e.target.value)} required
                placeholder="seu@empresa.com"
                className="h-11 border border-border bg-surface px-3.5 text-sm outline-none focus:border-[#0D0D0D] transition-colors"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium uppercase tracking-wide text-muted">Senha</label>
              <input
                type="password" value={password} onChange={(e) => setPassword(e.target.value)} required
                placeholder="••••••••"
                className="h-11 border border-border bg-surface px-3.5 text-sm outline-none focus:border-[#0D0D0D] transition-colors"
              />
            </div>

            {error && (
              <p className="text-accent text-sm flex items-center gap-2">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent" />
                {error}
              </p>
            )}

            <button
              type="submit" disabled={loading}
              className="h-12 bg-accent text-white font-display font-medium text-sm hover:bg-red-700 transition-colors disabled:opacity-60 mt-2 relative overflow-hidden group"
            >
              <span className="relative z-10">{loading ? 'Entrando...' : 'Entrar'}</span>
              <span className="absolute inset-0 bg-white/10 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-500 skew-x-12" />
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
