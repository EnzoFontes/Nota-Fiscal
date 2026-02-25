import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import UploadZone from '../components/UploadZone'
import api from '../api/client'

export default function Upload() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [manualLoading, setManualLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFile = async (file) => {
    setError(null)
    setLoading(true)
    const form = new FormData()
    form.append('file', file)
    try {
      const { data } = await api.post('/notas/upload', form)
      navigate(`/review/${data.id}`)
    } catch (err) {
      const detail = err.response?.data?.detail
      if (detail?.tips) {
        setError(detail)
      } else {
        setError({ message: typeof detail === 'string' ? detail : 'Erro ao processar documento' })
      }
    } finally {
      setLoading(false)
    }
  }

  const handleManualPix = async () => {
    setError(null)
    setManualLoading(true)
    try {
      const { data } = await api.post('/notas/novo-pix')
      navigate(`/review/${data.id}`)
    } catch (err) {
      setError({ message: 'Erro ao criar documento. Tente novamente.' })
    } finally {
      setManualLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 p-10 pl-12">
        <h1 className="font-display font-semibold text-[40px] tracking-tight mb-2">Upload</h1>
        <p className="text-muted text-sm mb-10">
          Envie um XML de NF-e, PDF, foto de nota fiscal ou comprovante PIX
        </p>

        <UploadZone onFile={handleFile} loading={loading} />

        {/* Manual PIX entry fallback */}
        <div className="mt-5 flex items-center gap-4">
          <div className="flex-1 border-t border-border" />
          <span className="text-xs text-muted">ou</span>
          <div className="flex-1 border-t border-border" />
        </div>
        <button
          onClick={handleManualPix}
          disabled={manualLoading || loading}
          className="mt-5 w-full h-11 border border-border text-sm font-medium hover:border-[#0D0D0D] transition-colors disabled:opacity-60"
        >
          {manualLoading ? 'Criando...' : 'Inserir comprovante PIX manualmente'}
        </button>

        {error && (
          <div className="mt-6 border-2 border-accent p-6">
            <p className="font-display font-semibold text-accent mb-1">{error.message}</p>
            {error.confidence != null && (
              <p className="text-sm text-muted mb-3">
                Confiança OCR: {error.confidence.toFixed(0)}%
              </p>
            )}
            {error.tips && (
              <ul className="flex flex-col gap-2 mt-3">
                {error.tips.map((tip) => (
                  <li key={tip} className="text-sm flex gap-2">
                    <span className="text-accent font-medium">—</span>
                    {tip}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
