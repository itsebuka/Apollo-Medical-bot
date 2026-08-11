/**
 * StatusBar — Top-of-screen system status indicator
 *
 * Displays real-time backend connectivity status, vector DB document count,
 * and model uptime. This component polls /health every 30 seconds.
 *
 * States:
 *   'checking'    — initial state, polling in progress
 *   'operational' — backend alive, model loaded
 *   'error'       — backend unreachable or model not loaded
 */
import { useEffect, useState, useCallback } from 'react'

const POLL_INTERVAL_MS = 30_000 // 30 seconds

export default function StatusBar({ onStatusChange }) {
  const [status, setStatus] = useState('checking')
  const [details, setDetails] = useState(null)

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch('/health', { signal: AbortSignal.timeout(5000) })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setStatus('operational')
      setDetails(data)
      onStatusChange?.('operational')
    } catch {
      setStatus('error')
      setDetails(null)
      onStatusChange?.('error')
    }
  }, [onStatusChange])

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [checkHealth])

  // ── Status indicator dot ──────────────────────────────────────────────────
  const dotColor = {
    checking:    'bg-warning animate-pulse',
    operational: 'bg-neon animate-pulse',
    error:       'bg-danger',
  }[status]

  const label = {
    checking:    'Connecting to Apollo Engine...',
    operational: 'Engine Operational',
    error:       'Backend Offline — Start the server',
  }[status]

  return (
    <div
      id="apollo-status-bar"
      className="flex items-center justify-between px-4 py-2 bg-apollo-surface border-b border-apollo-border"
      role="status"
      aria-live="polite"
      aria-label={`Apollo system status: ${label}`}
    >
      {/* Left — status indicator */}
      <div className="flex items-center gap-2.5">
        <span className={`w-2 h-2 rounded-full ${dotColor}`} />
        <span className={`text-xs font-mono font-medium tracking-wider ${
          status === 'operational' ? 'text-neon' :
          status === 'error'       ? 'text-danger' :
                                    'text-warning'
        }`}>
          {label}
        </span>
      </div>

      {/* Right — system details (only when operational) */}
      {status === 'operational' && details && (
        <div className="flex items-center gap-4 text-xs font-mono text-text-muted">
          <span>
            <span className="text-text-secondary">VDB: </span>
            <span className="text-neon">{details.vector_db_document_count}</span>
            <span className="text-text-secondary"> docs</span>
          </span>
          <span>
            <span className="text-text-secondary">Uptime: </span>
            <span className="text-text-secondary">{Math.floor(details.uptime_seconds)}s</span>
          </span>
          <span className="hidden sm:inline">
            <span className="text-text-secondary">Model: </span>
            <span className="text-neon">Llama-3 Q4_K_M</span>
          </span>
        </div>
      )}

      {/* Right — retry button when error */}
      {status === 'error' && (
        <button
          id="status-retry-btn"
          onClick={checkHealth}
          className="text-xs font-mono text-danger border border-danger/30 px-2 py-0.5 rounded
                     hover:bg-danger/10 hover:border-danger transition-all duration-200"
        >
          Retry
        </button>
      )}
    </div>
  )
}
