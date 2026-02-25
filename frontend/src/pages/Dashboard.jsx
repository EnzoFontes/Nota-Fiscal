import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import DocumentTable from '../components/DocumentTable'
import api from '../api/client'

const fmtBRL = (v) =>
  new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL' }).format(v ?? 0)

const TYPE_META = [
  { key: 'nfe',  label: 'NF-e',  color: '#3b82f6', bg: '#eff6ff' },
  { key: 'nfce', label: 'NFC-e', color: '#a855f7', bg: '#faf5ff' },
  { key: 'pix',  label: 'PIX',   color: '#f97316', bg: '#fff7ed' },
  { key: 'nfse', label: 'NFS-e', color: '#22c55e', bg: '#f0fdf4' },
]

function SkeletonCard() {
  return (
    <div className="border border-border p-7 flex flex-col gap-3">
      <div className="skeleton h-2.5 w-20 rounded" />
      <div className="skeleton h-8 w-32 rounded" />
    </div>
  )
}

function SkeletonRows() {
  return (
    <div className="border border-border divide-y divide-border">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="px-5 py-4 flex items-center gap-6" style={{ animationDelay: `${i * 60}ms` }}>
          <div className="skeleton h-5 w-12 rounded" />
          <div className="skeleton h-3 w-24 rounded flex-1" />
          <div className="skeleton h-3 w-32 rounded" />
          <div className="skeleton h-3 w-20 rounded" />
          <div className="skeleton h-3 w-16 rounded" />
        </div>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [stats, setStats] = useState(null)
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.get('/dashboard/stats'), api.get('/notas/?limit=10')])
      .then(([s, d]) => { setStats(s.data); setDocs(d.data) })
      .finally(() => setLoading(false))
  }, [])

  const handleDelete = async (id) => {
    await api.delete(`/notas/${id}`)
    setDocs((prev) => prev.filter((d) => d.id !== id))
    setStats((prev) => prev ? { ...prev, total_documentos: (prev.total_documentos ?? 1) - 1 } : prev)
  }

  const monthLabel = new Date().toLocaleDateString('pt-BR', { month: 'long', year: 'numeric' })

  // Type breakdown from recent docs
  const typeCounts = docs.reduce((acc, d) => {
    acc[d.tipo] = (acc[d.tipo] ?? 0) + 1
    return acc
  }, {})
  const totalDocs = docs.length || 1

  return (
    <div className="flex min-h-screen bg-[#fafafa]">
      <Sidebar />
      <main className="flex-1 overflow-auto">

        {/* Top header bar */}
        <div className="bg-white border-b border-border px-12 py-7 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-1 h-10 bg-accent flex-shrink-0" />
            <div>
              <h1 className="font-display font-semibold text-[32px] tracking-tight leading-none">Dashboard</h1>
              <p className="text-muted text-xs mt-1.5 capitalize">{monthLabel}</p>
            </div>
          </div>
          <button
            onClick={() => navigate('/upload')}
            className="h-10 px-6 bg-accent text-white text-sm font-display font-medium hover:bg-red-700 transition-colors flex items-center gap-2 group"
          >
            <span className="text-white/70 group-hover:text-white transition-colors">+</span>
            Novo Upload
          </button>
        </div>

        <div className="px-12 py-8">

          {/* Stats grid */}
          <div className="grid grid-cols-4 gap-5 mb-8">
            {loading ? (
              [...Array(4)].map((_, i) => <SkeletonCard key={i} />)
            ) : (
              <>
                {/* Primary stat - accent */}
                <div className="stats-card col-span-1 border border-accent/20 bg-accent p-7 flex flex-col gap-2 cursor-default">
                  <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Total do Mês</p>
                  <p className="font-display text-[28px] font-semibold tracking-tight leading-none text-white">
                    {fmtBRL(stats?.total_mes)}
                  </p>
                  <p className="text-xs text-white/50 mt-1">{stats?.total_documentos ?? 0} documentos</p>
                </div>

                <div className="stats-card border border-border bg-white p-7 flex flex-col gap-2 cursor-default">
                  <p className="text-xs font-medium text-muted uppercase tracking-wide">Comprovantes PIX</p>
                  <p className="font-display text-[28px] font-semibold tracking-tight leading-none text-[#0D0D0D]">
                    {fmtBRL(stats?.total_pix)}
                  </p>
                  <div className="mt-1 h-1 bg-surface rounded-full overflow-hidden">
                    <div
                      className="h-full bg-orange-400 rounded-full transition-all duration-700"
                      style={{ width: stats?.total_mes ? `${Math.min(100, (stats.total_pix / stats.total_mes) * 100)}%` : '0%' }}
                    />
                  </div>
                </div>

                <div className="stats-card border border-border bg-white p-7 flex flex-col gap-2 cursor-default">
                  <p className="text-xs font-medium text-muted uppercase tracking-wide">Notas Fiscais</p>
                  <p className="font-display text-[28px] font-semibold tracking-tight leading-none text-[#0D0D0D]">
                    {fmtBRL(stats?.total_nf)}
                  </p>
                  <div className="mt-1 h-1 bg-surface rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-400 rounded-full transition-all duration-700"
                      style={{ width: stats?.total_mes ? `${Math.min(100, (stats.total_nf / stats.total_mes) * 100)}%` : '0%' }}
                    />
                  </div>
                </div>

                <div className="stats-card border border-border bg-white p-7 flex flex-col gap-2 cursor-default">
                  <p className="text-xs font-medium text-muted uppercase tracking-wide">Aguardando Revisão</p>
                  <p className={`font-display text-[28px] font-semibold tracking-tight leading-none ${
                    (stats?.aguardando_revisao ?? 0) > 0 ? 'text-yellow-500' : 'text-[#0D0D0D]'
                  }`}>
                    {stats?.aguardando_revisao ?? 0}
                  </p>
                  <p className="text-xs text-muted mt-1">documentos</p>
                </div>
              </>
            )}
          </div>

          {/* Document type breakdown */}
          {!loading && docs.length > 0 && (
            <div className="bg-white border border-border p-6 mb-8">
              <p className="text-xs font-medium text-muted uppercase tracking-wide mb-4">
                Distribuição por tipo — últimos {docs.length} documentos
              </p>
              <div className="flex gap-1 h-2 rounded-sm overflow-hidden mb-4">
                {TYPE_META.map(({ key, color }) => {
                  const pct = ((typeCounts[key] ?? 0) / totalDocs) * 100
                  return pct > 0 ? (
                    <div
                      key={key}
                      className="h-full transition-all duration-700"
                      style={{ width: `${pct}%`, background: color }}
                    />
                  ) : null
                })}
              </div>
              <div className="flex gap-5 flex-wrap">
                {TYPE_META.map(({ key, label, color }) => {
                  const count = typeCounts[key] ?? 0
                  if (!count) return null
                  return (
                    <div key={key} className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                      <span className="text-xs text-muted">{label}</span>
                      <span className="text-xs font-semibold text-[#0D0D0D]">{count}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Recent docs section */}
          <div className="bg-white border border-border">
            <div className="flex items-center justify-between px-6 py-4 border-b border-border">
              <div className="flex items-center gap-3">
                <h2 className="font-display font-semibold text-base">Documentos Recentes</h2>
                {!loading && docs.length > 0 && (
                  <span className="text-[11px] bg-surface border border-border px-2 py-0.5 text-muted font-medium">
                    {docs.length}
                  </span>
                )}
              </div>
              <button
                onClick={() => navigate('/reports')}
                className="text-xs text-muted hover:text-accent transition-colors flex items-center gap-1 group"
              >
                Ver todos
                <span className="transition-transform group-hover:translate-x-0.5">→</span>
              </button>
            </div>

            {loading ? (
              <SkeletonRows />
            ) : (
              <DocumentTable documents={docs} onDelete={handleDelete} />
            )}
          </div>

        </div>
      </main>
    </div>
  )
}
