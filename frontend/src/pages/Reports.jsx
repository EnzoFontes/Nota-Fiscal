import React, { useEffect, useState } from 'react'
import Sidebar from '../components/Sidebar'
import DocumentTable from '../components/DocumentTable'
import api from '../api/client'

const TYPE_PILLS = [
  { value: '', label: 'Todos' },
  { value: 'nfe',  label: 'NF-e' },
  { value: 'nfce', label: 'NFC-e' },
  { value: 'nfse', label: 'NFS-e' },
  { value: 'pix',  label: 'PIX' },
]

const fmtBRL = (v) =>
  new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL' }).format(v)

function SkeletonRow() {
  return (
    <div className="px-5 py-4 flex items-center gap-6 border-b border-border">
      <div className="skeleton h-5 w-12 rounded" />
      <div className="skeleton h-3 w-28 rounded flex-1" />
      <div className="skeleton h-3 w-36 rounded" />
      <div className="skeleton h-3 w-20 rounded" />
      <div className="skeleton h-3 w-16 rounded" />
    </div>
  )
}

export default function Reports() {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(true)
  const [tipo, setTipo] = useState('')
  const [dataInicio, setDataInicio] = useState('')
  const [dataFim, setDataFim] = useState('')

  const fetchDocs = (t = tipo, di = dataInicio, df = dataFim) => {
    setLoading(true)
    const params = new URLSearchParams({ limit: '200' })
    if (t) params.set('tipo', t)
    if (di) params.set('data_inicio', di)
    if (df) params.set('data_fim', df)
    api.get(`/notas/?${params}`)
      .then((r) => setDocs(r.data))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchDocs() }, []) // eslint-disable-line

  const handleDelete = async (id) => {
    await api.delete(`/notas/${id}`)
    setDocs((prev) => prev.filter((d) => d.id !== id))
  }

  const handleExport = (format) => {
    const params = new URLSearchParams()
    if (tipo) params.set('tipo', tipo)
    if (dataInicio) params.set('data_inicio', dataInicio)
    if (dataFim) params.set('data_fim', dataFim)
    const query = params.toString() ? `?${params}` : ''

    if (format === 'xml') {
      api.get(`/notas/export/xml${query}`, { responseType: 'blob' }).then((r) => {
        const url = URL.createObjectURL(r.data)
        Object.assign(document.createElement('a'), { href: url, download: 'notas_fiscais.xml' }).click()
        URL.revokeObjectURL(url)
      })
    } else {
      api.get(`/notas/export/csv${query}`, { responseType: 'blob' }).then((r) => {
        const url = URL.createObjectURL(r.data)
        Object.assign(document.createElement('a'), { href: url, download: 'notas_fiscais.csv' }).click()
        URL.revokeObjectURL(url)
      })
    }
  }

  const handleTipo = (v) => {
    setTipo(v)
    fetchDocs(v, dataInicio, dataFim)
  }

  const clearFilters = () => {
    setTipo('')
    setDataInicio('')
    setDataFim('')
    fetchDocs('', '', '')
  }

  const hasFilters = tipo || dataInicio || dataFim
  const total = docs.reduce((s, d) => s + (d.valor_total ?? 0), 0)

  return (
    <div className="flex min-h-screen bg-[#fafafa]">
      <Sidebar />
      <main className="flex-1 overflow-auto">

        {/* Header */}
        <div className="bg-white border-b border-border px-12 py-7 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-1 h-10 bg-accent flex-shrink-0" />
            <div>
              <h1 className="font-display font-semibold text-[32px] tracking-tight leading-none">Relatórios</h1>
              <p className="text-muted text-xs mt-1.5">Documentos confirmados</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleExport('csv')}
              className="h-10 px-4 border border-border text-sm font-medium hover:border-[#0D0D0D] transition-colors flex items-center gap-2"
            >
              <svg width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M7 1v8M4 6l3 3 3-3M2 10v2a1 1 0 001 1h8a1 1 0 001-1v-2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              CSV
            </button>
            <button
              onClick={() => handleExport('xml')}
              className="h-10 px-4 border border-border text-sm font-medium hover:border-[#0D0D0D] transition-colors flex items-center gap-2"
            >
              <svg width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M7 1v8M4 6l3 3 3-3M2 10v2a1 1 0 001 1h8a1 1 0 001-1v-2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              XML
            </button>
          </div>
        </div>

        <div className="px-12 py-8">

          {/* Summary bar */}
          {!loading && docs.length > 0 && (
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-white border border-border px-5 py-4 flex items-center justify-between">
                <span className="text-xs text-muted uppercase tracking-wide font-medium">Documentos</span>
                <span className="font-display font-semibold text-xl">{docs.length}</span>
              </div>
              <div className="bg-white border border-border px-5 py-4 flex items-center justify-between">
                <span className="text-xs text-muted uppercase tracking-wide font-medium">Valor total</span>
                <span className="font-display font-semibold text-xl text-accent">{fmtBRL(total)}</span>
              </div>
              <div className="bg-white border border-border px-5 py-4 flex items-center justify-between">
                <span className="text-xs text-muted uppercase tracking-wide font-medium">Média por doc</span>
                <span className="font-display font-semibold text-xl">{fmtBRL(total / docs.length)}</span>
              </div>
            </div>
          )}

          {/* Filter bar */}
          <div className="bg-white border border-border p-5 mb-6">
            <div className="flex items-center gap-6 flex-wrap">
              {/* Type pills */}
              <div className="flex items-center gap-1.5">
                {TYPE_PILLS.map(({ value, label }) => (
                  <button
                    key={value}
                    onClick={() => handleTipo(value)}
                    className={`px-3 py-1.5 text-xs font-medium transition-all ${
                      tipo === value
                        ? 'bg-accent text-white'
                        : 'border border-border text-muted hover:border-[#0D0D0D] hover:text-[#0D0D0D]'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {/* Divider */}
              <div className="w-px h-6 bg-border flex-shrink-0" />

              {/* Date range */}
              <div className="flex items-center gap-2">
                <label className="text-[11px] text-muted font-medium uppercase tracking-wide">De</label>
                <input
                  type="date" value={dataInicio}
                  onChange={(e) => setDataInicio(e.target.value)}
                  className="h-8 border border-border px-2.5 text-xs outline-none focus:border-[#0D0D0D] transition-colors bg-white"
                />
                <label className="text-[11px] text-muted font-medium uppercase tracking-wide">Até</label>
                <input
                  type="date" value={dataFim}
                  onChange={(e) => setDataFim(e.target.value)}
                  className="h-8 border border-border px-2.5 text-xs outline-none focus:border-[#0D0D0D] transition-colors bg-white"
                />
              </div>

              {/* Apply / Clear */}
              <div className="flex items-center gap-2 ml-auto">
                {hasFilters && (
                  <button
                    onClick={clearFilters}
                    className="text-xs text-muted hover:text-accent transition-colors flex items-center gap-1"
                  >
                    <span>×</span> Limpar filtros
                  </button>
                )}
                <button
                  onClick={() => fetchDocs()}
                  className="h-8 px-4 bg-accent text-white text-xs font-medium hover:bg-red-700 transition-colors"
                >
                  Aplicar
                </button>
              </div>
            </div>

            {/* Active filter chips */}
            {hasFilters && (
              <div className="flex items-center gap-2 mt-3 pt-3 border-t border-border">
                <span className="text-[11px] text-muted">Filtros ativos:</span>
                {tipo && (
                  <span className="inline-flex items-center gap-1 text-[11px] bg-accent/10 text-accent border border-accent/20 px-2 py-0.5">
                    Tipo: {tipo.toUpperCase()}
                    <button onClick={() => handleTipo('')} className="hover:opacity-60 transition-opacity ml-1">×</button>
                  </span>
                )}
                {dataInicio && (
                  <span className="inline-flex items-center gap-1 text-[11px] bg-surface border border-border text-muted px-2 py-0.5">
                    De: {new Date(dataInicio + 'T00:00:00').toLocaleDateString('pt-BR')}
                  </span>
                )}
                {dataFim && (
                  <span className="inline-flex items-center gap-1 text-[11px] bg-surface border border-border text-muted px-2 py-0.5">
                    Até: {new Date(dataFim + 'T00:00:00').toLocaleDateString('pt-BR')}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Table */}
          <div className="bg-white border border-border">
            {loading ? (
              <div>
                {[...Array(6)].map((_, i) => <SkeletonRow key={i} />)}
              </div>
            ) : docs.length === 0 ? (
              <div className="py-20 flex flex-col items-center gap-4">
                <div className="w-12 h-12 border-2 border-dashed border-border flex items-center justify-center">
                  <span className="text-muted text-xl font-light">∅</span>
                </div>
                <div className="text-center">
                  <p className="font-display font-medium text-[#0D0D0D]">Nenhum documento encontrado</p>
                  <p className="text-xs text-muted mt-1">
                    {hasFilters ? 'Tente ajustar os filtros acima' : 'Faça upload de uma nota ou comprovante'}
                  </p>
                </div>
              </div>
            ) : (
              <DocumentTable documents={docs} onDelete={handleDelete} />
            )}
          </div>

        </div>
      </main>
    </div>
  )
}
