/**
 * MessageBubble — Renders a single chat message
 *
 * Props:
 *   message.role:      'user' | 'assistant'
 *   message.content:   string (supports Markdown for assistant messages)
 *   message.isStreaming: bool — shows blinking cursor at end
 *   message.timestamp: Date object
 */
import { useState, useEffect } from 'react'
import { Copy, Check, AlertTriangle, Pencil, RefreshCw, Volume2, Square } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { motion } from 'framer-motion'
import ApolloLogo from './ApolloLogo'

function formatTime(date) {
  if (!date) return ''
  const d = date instanceof Date ? date : new Date(date)
  return d.toLocaleTimeString('en-NG', { hour: '2-digit', minute: '2-digit' })
}

// ── User Message Bubble ───────────────────────────────────────────────────────
function UserBubble({ messageId, content, timestamp, onEdit }) {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 14, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: "spring", stiffness: 320, damping: 26 }}
      className="flex flex-col items-end gap-1.5 group"
    >
      <div className="flex items-end gap-2.5 max-w-[85%] sm:max-w-[75%]">
        {/* Edit button — visible on hover */}
        {onEdit && (
          <button
            onClick={() => onEdit(messageId, content)}
            title="Edit message"
            className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5 text-slate-400 dark:text-zinc-500 hover:text-blue-600 dark:hover:text-[#38BDF8] hover:bg-slate-200 dark:hover:bg-zinc-800 rounded-full flex-shrink-0 self-center"
          >
            <Pencil size={13} />
          </button>
        )}

        {/* Message bubble — Signature Cobalt Blue background with high-contrast text */}
        <div
          className="bg-blue-600 dark:bg-[#38BDF8] text-white dark:text-slate-950 font-medium rounded-2xl rounded-tr-xs px-5 py-3 shadow-md transition-all duration-200 selection:bg-blue-900 selection:text-white dark:selection:bg-slate-950 dark:selection:text-sky-200"
        >
          <p className="text-sm font-sans leading-relaxed whitespace-pre-wrap break-words">
            {content}
          </p>
        </div>

        {/* User avatar */}
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600/10 dark:bg-[#38BDF8]/10 border border-blue-600/30 dark:border-[#38BDF8]/30 flex items-center justify-center text-blue-600 dark:text-[#38BDF8] font-mono text-xs shadow-xs">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </svg>
        </div>
      </div>

      {/* Timestamp */}
      <span className="text-[10px] text-slate-400 dark:text-zinc-500 font-mono mr-10">
        {formatTime(timestamp)}
      </span>
    </motion.div>
  )
}

// ── Apollo Assistant Bubble ───────────────────────────────────────────────────
function AssistantBubble({ content, isStreaming, timestamp, isLowConfidence, isError, onRetry }) {
  const [copied, setCopied] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  
  useEffect(() => {
    return () => {
      if (isSpeaking && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel()
      }
    }
  }, [isSpeaking])

  const handleSpeak = () => {
    if (!('speechSynthesis' in window)) {
      alert("Text-to-speech is not supported in this browser.")
      return
    }

    if (isSpeaking) {
      window.speechSynthesis.cancel()
      setIsSpeaking(false)
      return
    }

    window.speechSynthesis.cancel()

    const cleanContent = content
      .replace(/\[Source:.*?\]/g, '')
      .replace(/\[.*?\]\(.*?\)/g, '')
      .replace(/[*#_`~]/g, '')

    const utterance = new SpeechSynthesisUtterance(cleanContent)
    utterance.rate = 1.05
    
    const voices = window.speechSynthesis.getVoices()
    const preferredVoice = voices.find(v => v.name.includes('Zira') || v.name.includes('Female') || v.name.includes('Google')) || voices[0]
    if (preferredVoice) utterance.voice = preferredVoice

    utterance.onend = () => setIsSpeaking(false)
    utterance.onerror = () => setIsSpeaking(false)

    setIsSpeaking(true)
    window.speechSynthesis.speak(utterance)
  }
  
  const handleCopy = () => {
    const textToCopy = `Apollo Triage Summary\nGenerated: ${formatTime(timestamp)}\n\n${content}`
    navigator.clipboard.writeText(textToCopy)
      .then(() => {
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      })
      .catch(() => {
        try {
          const el = document.createElement('textarea')
          el.value = textToCopy
          el.style.position = 'fixed'
          el.style.opacity = '0'
          document.body.appendChild(el)
          el.select()
          document.execCommand('copy')
          document.body.removeChild(el)
          setCopied(true)
          setTimeout(() => setCopied(false), 2000)
        } catch { /* Silent fail */ }
      })
  }

  const processedContent = content.replace(/\[Source(?: File)?: (.*?)\]/g, '[$1](#citation)')

  return (
    <motion.div 
      initial={{ opacity: 0, y: 14, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: "spring", stiffness: 320, damping: 26 }}
      className="flex flex-col items-start gap-1.5"
    >
      <div className="flex items-start gap-3 max-w-[92%] sm:max-w-[85%]">
        {/* Apollo avatar */}
        <div
          className={`flex-shrink-0 w-8 h-8 rounded-full bg-slate-100 dark:bg-zinc-900 border flex
                      items-center justify-center shadow-sm transition-all duration-300 ${
                        isStreaming
                          ? 'border-blue-600 dark:border-[#38BDF8] shadow-blue-500/20'
                          : 'border-slate-300 dark:border-zinc-800'
                      }`}
        >
          <ApolloLogo size={16} animated={isStreaming} />
        </div>

        {/* Message bubble — Light Grey in Light Mode (bg-slate-100/90 border-slate-200/90) & Deep Translucent Charcoal in Dark Mode */}
        <div
          className={`bg-slate-100/90 dark:bg-[#121519]/90 border rounded-2xl rounded-tl-xs
                      px-5 py-4 shadow-sm dark:shadow-md transition-all duration-200 ${
                        isStreaming
                          ? 'border-blue-600/50 dark:border-[#38BDF8]/40'
                          : 'border-slate-200/90 dark:border-zinc-800/80'
                      }`}
        >
          {/* Header label */}
          <div className="flex items-center justify-between gap-2 mb-3 pb-2 border-b border-slate-200/80 dark:border-zinc-800/70">
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono font-semibold tracking-widest text-blue-600 dark:text-[#38BDF8] uppercase">
                Apollo
              </span>
              {isStreaming && (
                <span className="text-[10px] font-mono text-slate-500 dark:text-zinc-500 animate-pulse">
                  · generating
                </span>
              )}
            </div>
          </div>

          {/* Low Confidence Warning */}
          {isLowConfidence && (
            <div className="mb-4 p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-2 text-amber-700 dark:text-amber-400">
              <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
              <p className="text-xs font-medium leading-relaxed font-sans">
                Low Confidence Match: Apollo could not find highly relevant clinical context for this query. Proceed with clinical caution.
              </p>
            </div>
          )}

          {/* Markdown content */}
          <div className="apollo-response font-sans break-words">
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                a: ({node, href, children, ...props}) => {
                  if (href === '#citation') {
                    return (
                      <span className="inline-flex items-center px-2 py-0.5 ml-1 rounded-md text-[10px] font-mono font-medium bg-blue-600/10 dark:bg-[#38BDF8]/10 text-blue-600 dark:text-[#38BDF8] border border-blue-600/30 dark:border-[#38BDF8]/30 cursor-pointer hover:bg-blue-600/20 dark:hover:bg-[#38BDF8]/20 transition-colors" title="Source Document">
                        <svg className="w-3 h-3 mr-1 inline" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>
                        {children}
                      </span>
                    )
                  }
                  return <a href={href} {...props}>{children}</a>
                }
              }}
            >
              {processedContent}
            </ReactMarkdown>
            {/* Blinking cursor shown while streaming */}
            {isStreaming && <span className="typing-cursor" aria-hidden="true" />}
          </div>

          {/* Action Bar */}
          {!isStreaming && (
            <div className="mt-4 pt-3 border-t border-slate-200/80 dark:border-zinc-800/70 flex items-center justify-between gap-2">
              {/* Retry button — only on error messages */}
              {isError && onRetry ? (
                <button
                  onClick={onRetry}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-[11px] font-mono font-medium text-red-600 dark:text-red-400 transition-all"
                  title="Retry this query"
                >
                  <RefreshCw size={13} />
                  RETRY
                </button>
              ) : <div />}
              
              {/* Copy / EHR Export & Speak */}
              {!isError && (
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleSpeak}
                    className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-[11px] font-mono font-medium transition-all ${
                      isSpeaking
                        ? 'bg-blue-600/20 border-blue-600/50 text-blue-600 dark:text-[#38BDF8]'
                        : 'bg-white/80 dark:bg-zinc-800/60 hover:bg-blue-600/10 border-slate-200/90 dark:border-zinc-700/60 text-slate-600 dark:text-zinc-400 hover:text-blue-600 dark:hover:text-[#38BDF8]'
                    }`}
                    title={isSpeaking ? "Stop speaking" : "Read answer aloud"}
                  >
                    {isSpeaking ? <Square size={13} className="fill-current" /> : <Volume2 size={13} />}
                    {isSpeaking ? 'STOP' : 'READ ALOUD'}
                  </button>

                  <button
                    onClick={handleCopy}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/80 dark:bg-zinc-800/60 hover:bg-blue-600/10 border border-slate-200/90 dark:border-zinc-700/60 text-[11px] font-mono font-medium text-slate-600 dark:text-zinc-400 hover:text-blue-600 dark:hover:text-[#38BDF8] transition-all"
                    title="Copy for Electronic Health Record"
                  >
                    {copied ? <Check size={13} /> : <Copy size={13} />}
                    {copied ? 'COPIED TO CLIPBOARD' : 'EXPORT EHR SUMMARY'}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Timestamp */}
      {!isStreaming && (
        <span className="text-[10px] text-slate-400 dark:text-zinc-500 font-mono ml-11">
          {formatTime(timestamp)}
        </span>
      )}
    </motion.div>
  )
}

// ── Public Component ──────────────────────────────────────────────────────────
export default function MessageBubble({ message, onEdit, onRetry }) {
  if (message.role === 'user') {
    return (
      <UserBubble
        messageId={message.id}
        content={message.content}
        timestamp={message.timestamp}
        onEdit={onEdit}
      />
    )
  }
  return (
    <AssistantBubble
      content={message.content}
      isStreaming={message.isStreaming}
      timestamp={message.timestamp}
      isLowConfidence={message.isLowConfidence}
      isError={message.isError}
      onRetry={onRetry ? () => onRetry(message.id, message.retryQuery) : null}
    />
  )
}
