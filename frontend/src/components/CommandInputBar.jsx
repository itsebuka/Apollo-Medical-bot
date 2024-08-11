import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Paperclip, Mic, Send, Square, Pencil, X, FileText, Loader2, Stethoscope, Microscope, ChevronDown, AlertCircle } from 'lucide-react'

// Fallback domain list used if /domains fetch fails (offline safety net)
const FALLBACK_SCOPE_OPTIONS = [
  { id: 'all', label: 'All Folders', icon: '🌐' },
  { id: 'virology', label: 'Virology', icon: '🧬' },
  { id: 'neuroscience', label: 'Neuroscience', icon: '🧠' },
  { id: 'pathophysiology', label: 'Pathophysiology', icon: '🫀' },
  { id: 'pharmacology', label: 'Pharmacology', icon: '💊' },
  { id: 'bacteriology', label: 'Bacteriology', icon: '🧫' },
  { id: 'epidemiology', label: 'Epidemiology', icon: '📊' },
  { id: 'homeopathy', label: 'Homeopathy', icon: '🌿' },
]

// Domain → emoji icon mapping for dynamically fetched domains
const DOMAIN_ICONS = {
  virology: '🧬', neuroscience: '🧠', pathophysiology: '🫀',
  pharmacology: '💊', bacteriology: '🧫', epidemiology: '📊',
  homeopathy: '🌿', general: '📁', default: '🔬',
}

export default function CommandInputBar({
  query,
  setQuery,
  onSubmit,
  isStreaming,
  onStopGeneration,
  inputRef,
  fileInputRef,
  onFileSelect,
  isUploading,
  uploadedDoc,
  onRemoveUploadedDoc,
  isListening,
  onToggleVoice,
  editingMessageId,
  onCancelEdit,
  // Lifted state props for scope and mode — so App.jsx can read them and pass to the backend
  scope,
  onScopeChange,
  mode,
  onModeChange,
}) {
  const [scopeOptions, setScopeOptions] = useState(FALLBACK_SCOPE_OPTIONS)
  const [isScopeOpen, setIsScopeOpen] = useState(false)
  const scopeDropdownRef = useRef(null)

  // Fetch live domain list from /domains endpoint on mount.
  // Falls back silently to FALLBACK_SCOPE_OPTIONS if server is offline or fetch fails.
  useEffect(() => {
    fetch('/domains')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && Array.isArray(data.domains) && data.domains.length > 0) {
          const dynamic = [
            { id: 'all', label: 'All Folders', icon: '🌐' },
            ...data.domains.map(d => ({
              id: d.toLowerCase(),
              label: d.charAt(0).toUpperCase() + d.slice(1),
              icon: DOMAIN_ICONS[d.toLowerCase()] || DOMAIN_ICONS.default,
            }))
          ]
          setScopeOptions(dynamic)
        }
      })
      .catch(() => { /* silent fallback — keep FALLBACK_SCOPE_OPTIONS */ })
  }, [])

  // Close scope dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (scopeDropdownRef.current && !scopeDropdownRef.current.contains(e.target)) {
        setIsScopeOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const selectedScopeObj = scopeOptions.find(s => s.id === scope) || scopeOptions[0]

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSubmit(e)
    }
  }

  const hasText = query.trim().length > 0

  return (
    <div className="w-full max-w-3xl mx-auto px-4 z-20 pb-6 shrink-0 relative">
      {/* Hidden File Input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={onFileSelect}
        accept=".pdf,.txt"
        className="hidden"
      />

      {/* ── Banners: Edit Mode, Uploaded Doc, and Truncation Warning ─────── */}
      <div className="flex flex-wrap items-center gap-2 mb-2 px-2">
        {editingMessageId && (
          <div className="flex items-center justify-between w-full bg-amber-500/10 border border-amber-500/30 text-amber-600 dark:text-amber-400 px-3 py-1.5 rounded-xl text-xs font-mono">
            <div className="flex items-center gap-2">
              <Pencil size={13} />
              <span>Editing message in thread</span>
            </div>
            <button
              onClick={onCancelEdit}
              className="hover:text-slate-900 dark:hover:text-white transition-colors"
            >
              <X size={14} />
            </button>
          </div>
        )}

        {uploadedDoc && (
          <div className="flex items-center gap-2 bg-emerald-500/10 dark:bg-[#00E599]/10 border border-emerald-500/30 dark:border-[#00E599]/30 text-emerald-600 dark:text-[#00E599] px-3 py-1.5 rounded-xl text-xs font-mono">
            <FileText size={14} />
            <span className="truncate max-w-[240px] font-medium">{uploadedDoc.filename}</span>
            {uploadedDoc.truncated && (
              <span title="Document was too large and was trimmed to 3000 characters" className="flex items-center gap-1 ml-1 text-amber-500 dark:text-amber-400">
                <AlertCircle size={12} />
                <span className="text-[10px]">truncated</span>
              </span>
            )}
            <button
              onClick={onRemoveUploadedDoc}
              className="hover:text-slate-900 dark:hover:text-white transition-colors ml-1"
              title="Remove attached document"
            >
              <X size={13} />
            </button>
          </div>
        )}
      </div>

      {/* ── High-Tech Command Container ────────────────────────────────────── */}
      <div className="bg-white/90 dark:bg-[#121519]/90 backdrop-blur-xl border border-slate-200 dark:border-zinc-800 rounded-3xl p-3 shadow-lg dark:shadow-2xl transition-all duration-200 focus-within:border-emerald-500/50 dark:focus-within:border-[#00E599]/50 focus-within:ring-2 focus-within:ring-emerald-500/10 dark:focus-within:ring-[#00E599]/10">
        
        {/* ── Sub-Control Bar: Scope Selector & Mode Toggle ──────────────── */}
        <div className="flex items-center justify-between gap-2 pb-2 mb-2 border-b border-slate-100 dark:border-zinc-800/80 px-1 text-xs font-mono">
          
          {/* Scope Selector Dropdown (Opens UPWARDS so items are fully visible and scrollable) */}
          <div className="relative" ref={scopeDropdownRef}>
            <button
              type="button"
              onClick={() => setIsScopeOpen(!isScopeOpen)}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-100 dark:bg-zinc-800/60 hover:bg-slate-200 dark:hover:bg-zinc-800 border border-slate-200 dark:border-zinc-700/60 text-slate-700 dark:text-zinc-300 transition-colors font-medium text-[11px]"
            >
              <span>{selectedScopeObj.icon}</span>
              <span>{selectedScopeObj.label}</span>
              <ChevronDown size={12} className={`text-slate-400 dark:text-zinc-500 transition-transform duration-200 ${isScopeOpen ? 'rotate-180' : ''}`} />
            </button>

            <AnimatePresence>
              {isScopeOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 6, scale: 0.96 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 6, scale: 0.96 }}
                  transition={{ duration: 0.12 }}
                  className="absolute left-0 bottom-full mb-2 w-48 max-h-56 overflow-y-auto bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-800 rounded-xl shadow-2xl z-50 p-1 space-y-0.5 text-xs font-mono"
                >
                  {scopeOptions.map((opt) => (
                    <button
                      key={opt.id}
                      onClick={() => {
                        onScopeChange(opt.id)
                        setIsScopeOpen(false)
                      }}
                      className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left text-xs transition-colors ${
                        scope === opt.id
                          ? 'bg-emerald-500/10 dark:bg-[#00E599]/10 text-emerald-600 dark:text-[#00E599] font-semibold'
                          : 'text-slate-700 dark:text-zinc-300 hover:bg-slate-100 dark:hover:bg-zinc-800/80'
                      }`}
                    >
                      <span className="text-sm">{opt.icon}</span>
                      <span>{opt.label}</span>
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Mode Toggle: Triage Mode vs Deep Research Mode */}
          <div className="flex items-center p-0.5 rounded-lg bg-slate-100 dark:bg-zinc-800/60 border border-slate-200 dark:border-zinc-700/60 text-[11px]">
            <button
              type="button"
              onClick={() => onModeChange('triage')}
              className={`flex items-center gap-1 px-2 py-1 rounded-md transition-all ${
                mode === 'triage'
                  ? 'bg-white dark:bg-zinc-900 text-emerald-600 dark:text-[#00E599] font-semibold shadow-xs'
                  : 'text-slate-500 dark:text-zinc-400 hover:text-slate-800 dark:hover:text-zinc-200'
              }`}
            >
              <Stethoscope size={12} />
              <span>Triage Mode</span>
            </button>
            <button
              type="button"
              onClick={() => onModeChange('research')}
              className={`flex items-center gap-1 px-2 py-1 rounded-md transition-all ${
                mode === 'research'
                  ? 'bg-white dark:bg-zinc-900 text-emerald-600 dark:text-[#00E599] font-semibold shadow-xs'
                  : 'text-slate-500 dark:text-zinc-400 hover:text-slate-800 dark:hover:text-zinc-200'
              }`}
            >
              <Microscope size={12} />
              <span>Deep Research</span>
            </button>
          </div>

        </div>

        {/* ── Main Input Area ──────────────────────────────────────────────── */}
        <div className="flex items-end gap-2 px-1">
          
          {/* File Attachment Button */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading || isStreaming}
            title="Attach medical document (PDF or TXT)"
            className="p-2 text-slate-400 hover:text-slate-700 dark:text-zinc-500 dark:hover:text-zinc-200 hover:bg-slate-100 dark:hover:bg-zinc-800/80 rounded-xl transition-colors flex-shrink-0 disabled:opacity-40 mb-1"
          >
            {isUploading ? <Loader2 size={18} className="animate-spin text-emerald-500 dark:text-[#00E599]" /> : <Paperclip size={18} />}
          </button>

          {/* Text Area Input */}
          <textarea
            ref={inputRef}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              e.target.style.height = 'auto'
              e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`
            }}
            onKeyDown={handleKeyDown}
            placeholder={
              mode === 'triage'
                ? "Describe patient symptoms or clinical presentation..."
                : "Enter pharmacological, genomic, or pathophysiological query..."
            }
            className="flex-1 bg-transparent border-none text-slate-900 dark:text-zinc-100 placeholder:text-slate-400 dark:placeholder:text-zinc-500 px-2 py-1.5 resize-none max-h-40 min-h-[40px] focus:outline-none text-sm font-sans leading-relaxed"
            rows={1}
          />

          {/* Action Buttons: Voice / Send / Stop */}
          <div className="flex items-center gap-1.5 flex-shrink-0 mb-1">
            {isStreaming ? (
              // Stop Generation Button
              <button
                type="button"
                onClick={onStopGeneration}
                title="Stop generating"
                className="p-2.5 bg-red-500/10 text-red-600 dark:text-red-400 hover:bg-red-500/20 border border-red-500/30 rounded-xl transition-all animate-pulse"
              >
                <Square size={16} className="fill-current" />
              </button>
            ) : !hasText ? (
              // Voice Input Button with Pulse Ring Animation when active
              <button
                type="button"
                onClick={onToggleVoice}
                title={isListening ? "Listening... Click to stop" : "Voice input"}
                className={`relative p-2.5 rounded-xl transition-all ${
                  isListening
                    ? 'bg-red-500 text-white shadow-md shadow-red-500/30'
                    : 'text-slate-400 dark:text-zinc-500 hover:text-emerald-600 dark:hover:text-[#00E599] hover:bg-slate-100 dark:hover:bg-zinc-800'
                }`}
              >
                {isListening && (
                  <span className="absolute inset-0 rounded-xl bg-red-500/40 animate-ping pointer-events-none" />
                )}
                <Mic size={18} />
              </button>
            ) : (
              // Morphing Send Button
              <motion.button
                type="button"
                onClick={onSubmit}
                initial={{ scale: 0.9, opacity: 0.8 }}
                animate={{ scale: 1, opacity: 1 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className={`p-2.5 rounded-xl font-medium transition-all shadow-md flex items-center justify-center ${
                  editingMessageId
                    ? 'bg-amber-500 hover:bg-amber-600 text-white shadow-amber-500/20'
                    : 'bg-emerald-600 hover:bg-emerald-700 dark:bg-[#00E599] dark:hover:bg-[#00E599]/90 text-white dark:text-zinc-950 shadow-emerald-500/20'
                }`}
              >
                {editingMessageId ? <Pencil size={18} /> : <Send size={18} className="stroke-[2.5]" />}
              </motion.button>
            )}
          </div>
        </div>

      </div>

      {/* ── Keyboard Shortcut Hints ────────────────────────────────────────── */}
      <div className="mt-2 text-center flex items-center justify-center gap-2 font-mono text-[10px] text-slate-400 dark:text-zinc-500">
        <span>Press <kbd className="px-1.5 py-0.5 rounded bg-slate-200 dark:bg-zinc-800 text-slate-700 dark:text-zinc-300 border border-slate-300 dark:border-zinc-700 font-medium">↵</kbd> to send</span>
        <span>·</span>
        <span><kbd className="px-1.5 py-0.5 rounded bg-slate-200 dark:bg-zinc-800 text-slate-700 dark:text-zinc-300 border border-slate-300 dark:border-zinc-700 font-medium">Shift+↵</kbd> for new line</span>
      </div>
    </div>
  )
}
