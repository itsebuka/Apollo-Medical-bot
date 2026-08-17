import { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, MessageSquare, MoreHorizontal, Trash2, Sun, Moon, Database, FolderCheck, Cpu } from 'lucide-react'
import ApolloLogo from './ApolloLogo'

/**
 * Group sessions chronologically: TODAY, YESTERDAY, PREVIOUS 7 DAYS, OLDER
 */
function groupSessionsByDate(sessions = []) {
  const now = new Date()
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const yesterdayStart = todayStart - 86400000
  const sevenDaysStart = todayStart - (6 * 86400000)

  const groups = {
    TODAY: [],
    YESTERDAY: [],
    'PREVIOUS 7 DAYS': [],
    OLDER: []
  }

  sessions.forEach(session => {
    const time = new Date(session.updatedAt || session.createdAt || Date.now()).getTime()
    if (time >= todayStart) {
      groups.TODAY.push(session)
    } else if (time >= yesterdayStart) {
      groups.YESTERDAY.push(session)
    } else if (time >= sevenDaysStart) {
      groups['PREVIOUS 7 DAYS'].push(session)
    } else {
      groups.OLDER.push(session)
    }
  })

  return groups
}

export default function Sidebar({
  sessions = [],
  currentSessionId,
  onSelectSession,
  onClearChat,
  onDeleteSession,
  systemStatus,
  theme,
  onToggleTheme,
  docCount,
  domainCount,
}) {
  const [activeDropdown, setActiveDropdown] = useState(null)

  // Close dropdown if clicked outside
  useEffect(() => {
    const handleClickOutside = () => setActiveDropdown(null)
    if (activeDropdown) {
      document.addEventListener('click', handleClickOutside)
    }
    return () => document.removeEventListener('click', handleClickOutside)
  }, [activeDropdown])

  const groupedSessions = useMemo(() => groupSessionsByDate(sessions), [sessions])

  const displayDocCount = docCount != null ? docCount.toLocaleString() : '64,200'
  const displayDomainCount = domainCount != null ? domainCount : '18'

  return (
    <aside
      id="apollo-sidebar"
      className="w-[280px] h-full flex-shrink-0 flex flex-col bg-slate-100/90 dark:bg-[#09090b]/80 backdrop-blur-xl border-r border-slate-200 dark:border-zinc-800/80 text-slate-800 dark:text-zinc-100 overflow-hidden relative z-20 transition-colors duration-250"
      role="complementary"
    >
      {/* ── Top Section: Brand & New Chat ─────────────────────────────────── */}
      <div className="p-4 pt-5 pb-3 border-b border-slate-200/80 dark:border-zinc-800/60">
        <div className="flex items-center justify-between mb-4 px-1">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-blue-600/10 dark:bg-[#38BDF8]/10 border border-blue-600/30 dark:border-[#38BDF8]/30 flex items-center justify-center text-blue-600 dark:text-[#38BDF8]">
              <ApolloLogo size={16} />
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-semibold tracking-tight text-slate-900 dark:text-white flex items-center gap-1.5">
                Apollo Engine
                <span className="text-[10px] font-mono font-medium px-1.5 py-0.2 rounded bg-blue-600/10 dark:bg-[#38BDF8]/10 text-blue-600 dark:text-[#38BDF8] border border-blue-600/20 dark:border-[#38BDF8]/20">
                  v2.4
                </span>
              </span>
            </div>
          </div>
        </div>

        <motion.button
          onClick={onClearChat}
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.98 }}
          transition={{ type: "spring", stiffness: 400, damping: 20 }}
          className="flex items-center justify-center gap-2.5 w-full bg-blue-600 hover:bg-blue-700 dark:bg-[#38BDF8] dark:hover:bg-[#38BDF8]/90 text-white dark:text-slate-950 font-medium text-sm rounded-xl px-4 py-2.5 shadow-sm hover:shadow-md transition-all group"
        >
          <Plus size={18} className="stroke-[2.5]" />
          <span>New Thread</span>
        </motion.button>
      </div>

      {/* ── Middle Section: Grouped Recent History ─────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-5">
        {Object.entries(groupedSessions).map(([category, items]) => {
          if (items.length === 0) return null
          return (
            <div key={category} className="space-y-1">
              <h2 className="text-[10px] font-mono font-semibold text-slate-400 dark:text-zinc-500 uppercase tracking-widest px-2.5 mb-1.5">
                {category}
              </h2>
              {items.map(session => {
                const isActive = currentSessionId === session.id
                return (
                  <div key={session.id} className="relative flex items-center group">
                    <motion.button
                      onClick={() => onSelectSession(session.id)}
                      whileHover={{ x: 2 }}
                      transition={{ type: "spring", stiffness: 350, damping: 25 }}
                      className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left text-xs transition-all relative ${
                        isActive
                          ? 'bg-slate-200/80 dark:bg-zinc-800/80 border-l-2 border-blue-600 dark:border-[#38BDF8] text-slate-900 dark:text-white font-medium shadow-sm'
                          : 'text-slate-600 dark:text-zinc-400 hover:bg-slate-200/50 dark:hover:bg-zinc-800/40 hover:text-slate-900 dark:hover:text-zinc-200'
                      }`}
                    >
                      <MessageSquare
                        size={14}
                        className={`flex-shrink-0 transition-colors ${
                          isActive ? 'text-blue-600 dark:text-[#38BDF8]' : 'text-slate-400 dark:text-zinc-500 group-hover:text-slate-700 dark:group-hover:text-zinc-300'
                        }`}
                      />
                      <span className="truncate pr-5 font-sans">
                        {session.title || 'Untitled Thread'}
                      </span>
                    </motion.button>

                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setActiveDropdown(activeDropdown === session.id ? null : session.id)
                      }}
                      className={`absolute right-2 p-1 rounded-md text-slate-400 dark:text-zinc-500 hover:text-red-600 dark:hover:text-red-400 hover:bg-slate-300/50 dark:hover:bg-zinc-700/50 transition-all ${
                        activeDropdown === session.id ? 'opacity-100 visible' : 'opacity-0 invisible group-hover:opacity-100 group-hover:visible'
                      }`}
                      title="Thread options"
                    >
                      <MoreHorizontal size={14} />
                    </button>

                    <AnimatePresence>
                      {activeDropdown === session.id && (
                        <motion.div
                          initial={{ opacity: 0, y: -5, scale: 0.95 }}
                          animate={{ opacity: 1, y: 0, scale: 1 }}
                          exit={{ opacity: 0, y: -5, scale: 0.95 }}
                          transition={{ duration: 0.12 }}
                          className="absolute right-0 top-9 w-36 bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-800 rounded-lg shadow-lg z-50 overflow-hidden"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              setActiveDropdown(null)
                              onDeleteSession?.(session.id)
                            }}
                            className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors text-left"
                          >
                            <Trash2 size={13} />
                            Delete Thread
                          </button>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )
              })}
            </div>
          )
        })}

        {sessions.length === 0 && (
          <div className="py-8 text-center px-4 text-xs text-slate-400 dark:text-zinc-500 font-mono">
            No active threads. Start a new medical prompt.
          </div>
        )}
      </div>

      {/* ── Bottom Section: DB Status Footer & Theme Toggle ────────────────── */}
      <div className="p-3 border-t border-slate-200 dark:border-zinc-800/80 bg-slate-100/50 dark:bg-zinc-950/40 space-y-3">
        {/* Database Status Card */}
        <div className="p-3 rounded-xl bg-white/70 dark:bg-zinc-900/60 border border-slate-200/80 dark:border-zinc-800/70 shadow-sm space-y-1.5 font-mono text-[11px]">
          <div className="flex items-center justify-between text-slate-700 dark:text-zinc-300">
            <span className="flex items-center gap-1.5">
              <FolderCheck size={13} className="text-blue-600 dark:text-[#38BDF8]" />
              {displayDomainCount} Domains Loaded
            </span>
          </div>
          <div className="flex items-center justify-between text-slate-700 dark:text-zinc-300">
            <span className="flex items-center gap-1.5">
              <Database size={13} className="text-blue-600 dark:text-[#38BDF8]" />
              {displayDocCount} Embeddings
            </span>
          </div>
          <div className="pt-1 flex items-center justify-between text-[10px] text-slate-500 dark:text-zinc-400 border-t border-slate-100 dark:border-zinc-800/50">
            <span className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${systemStatus === 'operational' ? 'bg-blue-500 dark:bg-[#38BDF8] animate-pulse' : 'bg-red-500'}`} />
              {systemStatus === 'operational' ? 'Offline Engine Ready' : 'System Disconnected'}
            </span>
            <Cpu size={12} className="text-slate-400 dark:text-zinc-500" />
          </div>
        </div>

        {/* Theme Toggle Switch */}
        <div className="flex items-center justify-between px-2 pt-1">
          <span className="text-xs font-mono font-medium text-slate-500 dark:text-zinc-400 flex items-center gap-1.5">
            {theme === 'dark' ? (
              <Moon size={14} className="text-[#38BDF8]" />
            ) : (
              <Sun size={14} className="text-amber-500" />
            )}
            {theme === 'dark' ? 'Dark Mode' : 'Light Mode'}
          </span>

          <button
            onClick={onToggleTheme}
            className={`relative w-11 h-6 rounded-full p-0.5 transition-colors duration-200 focus:outline-none ${
              theme === 'dark' ? 'bg-zinc-800 border border-zinc-700' : 'bg-slate-200 border border-slate-300'
            }`}
            title="Toggle Light / Dark Mode"
            aria-label="Toggle theme"
          >
            <motion.div
              layout
              transition={{ type: "spring", stiffness: 500, damping: 30 }}
              className={`w-4 h-4 rounded-full flex items-center justify-center shadow-sm ${
                theme === 'dark'
                  ? 'translate-x-5 bg-[#38BDF8] text-slate-950'
                  : 'translate-x-0 bg-white text-slate-700'
              }`}
            >
              {theme === 'dark' ? <Moon size={10} /> : <Sun size={10} />}
            </motion.div>
          </button>
        </div>
      </div>
    </aside>
  )
}
