import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'

const TYPE_LABELS = { nfe: 'NF-e', nfce: 'NFC-e', nfse: 'NFS-e', pix: 'PIX' }

const TYPE_COLORS = {
  nfe:  'border-blue-200 text-blue-700 bg-blue-50',
  nfce: 'border-purple-200 text-purple-700 bg-purple-50',
  nfse: 'border-green-200 text-green-700 bg-green-50',
  pix:  'border-orange-200 text-orange-700 bg-orange-50',
}

const STATUS_CLASS = {
  validado:     'text-green-600',
  nao_validado: 'text-yellow-500',
  falha:        'text-accent',
  nao_aplicavel:'text-muted',
}

const STATUS_DOT = {
  validado:     'bg-green-500',
  nao_validado: 'bg-yellow-400',
  falha:        'bg-red-500',
  nao_aplicavel:'bg-gray-300',
}

const fmtBRL = (v) =>
  v != null
    ? new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL' }).format(v)
    : '—'

const fmtDate = (v) => (v ? new Date(v).toLocaleDateString('pt-BR') : '—')

function DeleteButton({ docId, onDelete }) {
  const [confirming, setConfirming] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleClick = (e) => {
    e.stopPropagation()
    setConfirming(true)
  }

  const handleConfirm = async (e) => {
    e.stopPropagation()
    setLoading(true)
    try {
      await onDelete(docId)
    } finally {
      setLoading(false)
      setConfirming(false)
    }
  }

  const handleCancel = (e) => {
    e.stopPropagation()
    setConfirming(false)
  }

  if (confirming) {
    return (
      <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
        <button
          onClick={handleConfirm}
          disabled={loading}
          className="text-[11px] font-medium text-white bg-accent px-2 py-1 hover:bg-red-700 transition-colors disabled:opacity-60"
        >
          {loading ? '...' : 'Confirmar'}
        </button>
        <button
          onClick={handleCancel}
          className="text-[11px] text-muted hover:text-[#0D0D0D] transition-colors"
        >
          Cancelar
        </button>
      </div>
    )
  }

  return (
    <button
      onClick={handleClick}
      className="text-[11px] text-muted hover:text-accent transition-all opacity-0 group-hover:opacity-100 group-hover:translate-x-0 translate-x-1"
      title="Remover documento"
    >
      Remover
    </button>
  )
}

const COLS = ['Tipo', 'Nº / Descrição', 'Emitente', 'Valor', 'Data', 'Status', '']

export default function DocumentTable({ documents = [], onDelete }) {
  const navigate = useNavigate()

  return (
    <div className="border border-border overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-surface">
            {COLS.map((h) => (
              <th
                key={h}
                className="text-left px-5 py-3.5 text-[11px] font-medium text-muted uppercase tracking-wide"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {documents.length === 0 ? (
            <tr>
              <td colSpan={COLS.length} className="text-center py-16 text-muted text-sm">
                Nenhum documento encontrado
              </td>
            </tr>
          ) : (
            documents.map((doc) => (
              <tr
                key={doc.id}
                onClick={() => navigate(`/review/${doc.id}`)}
                className="group border-b border-border last:border-0 cursor-pointer hover:bg-surface transition-all duration-150"
              >
                <td className="px-5 py-4">
                  <span className={`text-[11px] font-medium px-2 py-0.5 border ${TYPE_COLORS[doc.tipo] ?? 'border-border text-muted'}`}>
                    {TYPE_LABELS[doc.tipo] ?? doc.tipo}
                  </span>
                </td>
                <td className="px-5 py-4 font-medium">
                  {doc.numero ?? doc.pix_txid ?? `#${doc.id}`}
                </td>
                <td className="px-5 py-4 text-muted max-w-[200px] truncate">
                  {doc.emitente_razao_social ?? doc.pix_recebedor_nome ?? '—'}
                </td>
                <td className="px-5 py-4 font-semibold font-display">{fmtBRL(doc.valor_total)}</td>
                <td className="px-5 py-4 text-muted">{fmtDate(doc.data_importacao)}</td>
                <td className={`px-5 py-4 capitalize ${STATUS_CLASS[doc.status_sefaz] ?? ''}`}>
                  <span className="flex items-center gap-1.5">
                    <span className={`inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 ${STATUS_DOT[doc.status_sefaz] ?? 'bg-gray-300'}`} />
                    {doc.status_sefaz?.replace('_', ' ') ?? '—'}
                  </span>
                </td>
                <td className="px-5 py-4 text-right">
                  {onDelete && <DeleteButton docId={doc.id} onDelete={onDelete} />}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}
