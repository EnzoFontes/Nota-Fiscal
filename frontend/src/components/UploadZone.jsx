import React, { useCallback, useState } from 'react'

export default function UploadZone({ onFile, loading }) {
  const [dragging, setDragging] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }, [onFile])

  return (
    <label
      className={`block border-2 border-dashed cursor-pointer transition-colors ${
        dragging ? 'border-accent bg-red-50' : 'border-border hover:border-placeholder'
      } p-16 text-center`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input
        type="file"
        className="hidden"
        accept=".xml,.pdf,.jpg,.jpeg,.png,.webp,.tiff"
        onChange={(e) => { const f = e.target.files[0]; if (f) onFile(f) }}
        disabled={loading}
      />
      {loading ? (
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <div>
            <p className="font-display font-medium text-[#0D0D0D]">Processando documento...</p>
            <p className="text-sm text-muted mt-1">Lendo e validando na SEFAZ</p>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border border-border bg-surface flex items-center justify-center">
            <span className="text-2xl text-muted font-light">+</span>
          </div>
          <div>
            <p className="font-display font-medium text-[#0D0D0D]">
              Arraste um arquivo ou clique para selecionar
            </p>
            <p className="text-sm text-muted mt-1">
              XML · PDF · Imagem JPG / PNG / WEBP · Comprovante PIX
            </p>
          </div>
        </div>
      )}
    </label>
  )
}
