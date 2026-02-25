import React from 'react'

export default function StatsCard({ label, value, sub, accent = false }) {
  return (
    <div className="stats-card border border-border p-7 flex flex-col gap-2 cursor-default">
      <p className="text-xs font-medium text-muted uppercase tracking-wide">{label}</p>
      <p className={`font-display text-[32px] font-semibold tracking-tight leading-none ${
        accent ? 'text-accent' : 'text-[#0D0D0D]'
      }`}>
        {value}
      </p>
      {sub && <p className="text-xs text-muted">{sub}</p>}
    </div>
  )
}
