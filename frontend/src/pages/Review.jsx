import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import api from '../api/client'

const TYPE_LABELS = { nfe: 'NF-e', nfce: 'NFC-e', nfse: 'NFS-e', pix: 'PIX' }

function Field({ label, value, onChange, warn }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-medium flex items-center gap-2">
        {label}
        {warn && (
          <span className="text-[10px] bg-yellow-100 text-yellow-700 px-1.5 py-0.5 font-normal">
            Baixa confiança
          </span>
        )}
      </label>
      <input
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        className={`h-10 border px-3 text-sm outline-none focus:border-[#0D0D0D] transition-colors ${
          warn ? 'border-yellow-300 bg-yellow-50' : 'border-border bg-surface'
        }`}
      />
    </div>
  )
}

export default function Review() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [nota, setNota] = useState(null)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [discarding, setDiscarding] = useState(false)

  useEffect(() => {
    api.get(`/notas/${id}`).then((r) => setNota(r.data))
  }, [id])

  const upd = (field) => (val) => setNota((n) => ({ ...n, [field]: val }))

  const handleSave = async () => {
    setSaving(true)
    try {
      await api.put(`/notas/${id}`, { ...nota, confirmado: true })
      setSaved(true)
      setTimeout(() => navigate('/'), 1000)
    } catch {
      // Save failed — user remains on the page to retry
    } finally {
      setSaving(false)
    }
  }

  const handleDiscard = async () => {
    // If the document was never confirmed, delete it from the DB entirely
    if (nota && !nota.confirmado) {
      setDiscarding(true)
      try {
        await api.delete(`/notas/${id}`)
      } catch {
        // Discard failed — navigate away anyway; document can be cleaned later
      } finally {
        setDiscarding(false)
      }
    }
    navigate('/')
  }

  if (!nota) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 p-10 pl-12 text-muted text-sm">Carregando...</main>
      </div>
    )
  }

  const lowConf = nota.confidence_score != null && nota.confidence_score < 80
  const isPix = nota.tipo === 'pix'
  const isDraft = !nota.confirmado

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 p-10 pl-12 max-w-4xl overflow-auto">
        <button onClick={handleDiscard} className="text-sm text-muted hover:text-[#0D0D0D] mb-6">
          ← Voltar
        </button>

        <div className="flex items-start justify-between mb-3">
          <h1 className="font-display font-semibold text-[32px] tracking-tight">
            Revisar Documento
          </h1>
          {isDraft && (
            <span className="text-[11px] font-medium border border-yellow-300 text-yellow-700 bg-yellow-50 px-2 py-0.5 mt-2">
              Rascunho — não salvo
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 mb-8">
          <span className="text-[11px] font-medium border border-border px-2 py-0.5 uppercase">
            {TYPE_LABELS[nota.tipo] ?? nota.tipo}
          </span>
          {nota.confidence_score != null && (
            <span className={`text-xs ${lowConf ? 'text-yellow-600' : 'text-success'}`}>
              OCR: {nota.confidence_score.toFixed(0)}%
            </span>
          )}
          <span className="text-xs text-muted capitalize">
            SEFAZ: {nota.status_sefaz?.replace('_', ' ') ?? '—'}
          </span>
        </div>

        <div className="flex flex-col gap-5">
          {isPix ? (
            <>
              <p className="text-[11px] font-medium text-muted uppercase tracking-wide border-b border-border pb-2">
                Dados PIX
              </p>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Valor Total" value={nota.valor_total} onChange={upd('valor_total')} warn={lowConf} />
                <Field label="TXID" value={nota.pix_txid} onChange={upd('pix_txid')} warn={lowConf} />
                <Field label="Pagador" value={nota.pix_pagador_nome} onChange={upd('pix_pagador_nome')} warn={lowConf} />
                <Field label="CPF/CNPJ Pagador" value={nota.pix_pagador_cpf_cnpj} onChange={upd('pix_pagador_cpf_cnpj')} warn={lowConf} />
                <Field label="Recebedor" value={nota.pix_recebedor_nome} onChange={upd('pix_recebedor_nome')} warn={lowConf} />
                <Field label="Chave PIX" value={nota.pix_recebedor_chave} onChange={upd('pix_recebedor_chave')} warn={lowConf} />
              </div>
            </>
          ) : (
            <>
              <p className="text-[11px] font-medium text-muted uppercase tracking-wide border-b border-border pb-2">
                Dados da Nota
              </p>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Número" value={nota.numero} onChange={upd('numero')} warn={lowConf} />
                <Field label="Série" value={nota.serie} onChange={upd('serie')} warn={lowConf} />
                <div className="col-span-2">
                  <Field label="Chave de Acesso (44 dígitos)" value={nota.chave_acesso} onChange={upd('chave_acesso')} warn={lowConf} />
                </div>
                <Field label="Data de Emissão" value={nota.data_emissao?.slice(0, 10)} onChange={upd('data_emissao')} warn={lowConf} />
                <Field label="Emitente CNPJ" value={nota.emitente_cnpj} onChange={upd('emitente_cnpj')} warn={lowConf} />
                <Field label="Emitente" value={nota.emitente_razao_social} onChange={upd('emitente_razao_social')} warn={lowConf} />
                <Field label="Destinatário" value={nota.destinatario_nome} onChange={upd('destinatario_nome')} warn={lowConf} />
                <Field label="Valor Total" value={nota.valor_total} onChange={upd('valor_total')} warn={lowConf} />
                <Field label="Valor Desconto" value={nota.valor_desconto} onChange={upd('valor_desconto')} />
              </div>

              {nota.itens?.length > 0 && (
                <>
                  <p className="text-[11px] font-medium text-muted uppercase tracking-wide border-b border-border pb-2 mt-2">
                    Itens ({nota.itens.length})
                  </p>
                  <div className="border border-border overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border bg-surface">
                          {['Descrição', 'NCM', 'Qtd', 'Un', 'Vlr Unit.', 'Vlr Total'].map((h) => (
                            <th key={h} className="text-left px-4 py-3 text-[11px] text-muted font-medium uppercase tracking-wide">
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {nota.itens.map((item, i) => (
                          <tr key={i} className="border-b border-border last:border-0">
                            <td className="px-4 py-3">{item.descricao}</td>
                            <td className="px-4 py-3 text-muted">{item.ncm}</td>
                            <td className="px-4 py-3">{item.quantidade}</td>
                            <td className="px-4 py-3 text-muted">{item.unidade}</td>
                            <td className="px-4 py-3">{item.valor_unitario?.toFixed(2)}</td>
                            <td className="px-4 py-3 font-semibold font-display">{item.valor_total?.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </>
          )}

          <Field label="Observações" value={nota.observacoes} onChange={upd('observacoes')} />
        </div>

        <div className="mt-8 flex items-center gap-4">
          <button
            onClick={handleSave}
            disabled={saving || saved}
            className="h-12 px-8 bg-accent text-white font-display font-medium text-sm hover:bg-red-700 transition-colors disabled:opacity-60"
          >
            {saved ? 'Salvo!' : saving ? 'Salvando...' : 'Confirmar e Salvar'}
          </button>
          <button
            onClick={handleDiscard}
            disabled={discarding || saving}
            className="text-sm text-muted hover:text-[#0D0D0D] transition-colors disabled:opacity-60"
          >
            {discarding ? 'Descartando...' : isDraft ? 'Descartar rascunho' : 'Cancelar'}
          </button>
        </div>
      </main>
    </div>
  )
}
