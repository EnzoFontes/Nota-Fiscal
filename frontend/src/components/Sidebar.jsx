import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const NAV = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/upload', label: 'Upload' },
  { to: '/reports', label: 'Relatórios' },
]

export default function Sidebar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  return (
    <aside className="w-60 h-screen flex-shrink-0 border-r border-border flex flex-col bg-white">
      {/* Logo */}
      <div className="px-8 py-7 border-b border-border flex items-center gap-2.5 group">
        <div className="w-7 h-7 bg-accent flex-shrink-0 transition-transform duration-300 group-hover:rotate-12" />
        <span className="font-display font-semibold text-[18px] text-[#0D0D0D]">NotaFiscal</span>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-4 py-5 flex flex-col gap-0.5">
        {NAV.map(({ to, label, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `py-2.5 text-sm font-medium font-display transition-all duration-150 block relative border-l-2 pl-[calc(1.25rem-2px)] ${
                isActive
                  ? 'text-accent bg-red-50 border-accent'
                  : 'text-muted hover:text-[#0D0D0D] hover:bg-surface border-transparent hover:border-border'
              }`
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>

      {/* User */}
      <div className="px-6 py-5 border-t border-border">
        <p className="text-sm font-medium text-[#0D0D0D] truncate">{user?.email}</p>
        <p className="text-xs text-muted capitalize mt-0.5">{user?.role}</p>
        <button
          onClick={() => { logout(); navigate('/login') }}
          className="mt-3 text-xs text-muted hover:text-accent transition-colors"
        >
          Sair
        </button>
      </div>
    </aside>
  )
}
