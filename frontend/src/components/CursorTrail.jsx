import React, { useEffect, useRef } from 'react'

export default function CursorTrail() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    let points = []
    let animId

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const onMove = (e) => {
      points.push({ x: e.clientX, y: e.clientY, life: 1 })
      // Keep at most 22 points in the trail
      if (points.length > 22) points.shift()
    }
    window.addEventListener('mousemove', onMove)

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      for (let i = 0; i < points.length; i++) {
        const p = points[i]
        const t = (i + 1) / points.length // 0=oldest tail, 1=freshest

        const radius = 4 * t * p.life
        const alpha = t * p.life * 0.55

        ctx.beginPath()
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(220, 38, 38, ${alpha})`
        ctx.fill()

        // Decay: older points disappear faster
        p.life -= 0.035 + (1 - t) * 0.02
      }

      points = points.filter((p) => p.life > 0)
      animId = requestAnimationFrame(render)
    }
    render()

    return () => {
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousemove', onMove)
      cancelAnimationFrame(animId)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'fixed', top: 0, left: 0, pointerEvents: 'none', zIndex: 9999 }}
    />
  )
}
