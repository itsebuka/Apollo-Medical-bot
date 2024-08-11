import { useState, useRef, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Sidebar from './components/Sidebar'
import MessageBubble from './components/MessageBubble'
import HeroStarterCards from './components/HeroStarterCards'
import CommandInputBar from './components/CommandInputBar'
import { AlertTriangle } from 'lucide-react'
import ApolloLogo from './components/ApolloLogo'
import { playClickSound, playChimeSound } from './utils/audio'
import localforage from 'localforage'

// ─────────────────────────────────────────────────────────────────────────────
// APOLLO THINKING INDICATOR
// Shown during the TTFT wait (between submit and first token).
// Rotates through medical-themed deliberation phrases like Claude's thinking display.
// ─────────────────────────────────────────────────────────────────────────────
const THINKING_PHRASES = [
  'Searching knowledge base...',
  'Retrieving clinical context...',
  'Cross-referencing source documents...',
  'Analysing retrieved passages...',
  'Evaluating diagnostic context...',
  'Consulting reference materials...',
  'Synthesising clinical evidence...',
  'Reviewing pharmacological data...',
  'Assessing epidemiological context...',
  'Deliberating on differential findings...',
  'Mapping pathophysiological pathways...',
  'Formulating clinical response...',
]

function ApolloThinkingIndicator() {
  const [phraseIdx, setPhraseIdx] = useState(0)
  const [dotCount, setDotCount] = useState(1)

  useEffect(() => {
    const phraseTimer = setInterval(() => {
      setPhraseIdx(i => (i + 1) % THINKING_PHRASES.length)
    }, 2800)
    const dotTimer = setInterval(() => {
      setDotCount(d => (d % 3) + 1)
    }, 500)
    return () => {
      clearInterval(phraseTimer)
      clearInterval(dotTimer)
    }
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 260, damping: 22 }}
      className="flex items-start gap-3 max-w-[90%]"
    >
      {/* Apollo avatar with pulse ring */}
      <div className="relative flex-shrink-0">
        <div className="w-8 h-8 rounded-full bg-white dark:bg-zinc-900 border border-emerald-500/60 dark:border-[#00E599]/60 flex items-center justify-center shadow-sm">
          <ApolloLogo size={16} animated={true} />
        </div>
        {/* Outer pulse ring */}
        <motion.div
          animate={{ scale: [1, 1.55, 1], opacity: [0.5, 0, 0.5] }}
          transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
          className="absolute inset-0 rounded-full border border-emerald-500/40 dark:border-[#00E599]/40"
        />
      </div>

      {/* Thinking bubble */}
      <div className="bg-slate-100/90 dark:bg-[#121519]/90 border border-emerald-500/40 dark:border-[#00E599]/30 rounded-2xl rounded-tl-xs px-5 py-3 shadow-sm min-w-[220px]">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-[10px] font-mono font-semibold tracking-widest text-emerald-600 dark:text-[#00E599] uppercase">Apollo</span>
          <span className="text-[10px] font-mono text-slate-400 dark:text-zinc-500 animate-pulse">· thinking</span>
        </div>
        <AnimatePresence mode="wait">
          <motion.p
            key={phraseIdx}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.35 }}
            className="text-xs text-slate-600 dark:text-zinc-400 font-mono"
          >
            {THINKING_PHRASES[phraseIdx].replace('...', '.'.repeat(dotCount))}
          </motion.p>
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ROOT APP — High-End Medical Workstation
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [systemStatus, setSystemStatus] = useState('operational')
  const [docCount, setDocCount] = useState(null)
  const [domainCount, setDomainCount] = useState(null)
  const [sessions, setSessions] = useState([])
  const [currentSessionId, setCurrentSessionId] = useState(null)
  const [isReady, setIsReady] = useState(false)

  // ── Poll Backend Health & Knowledge Stats ──────────────────────────────────
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch('/health')
        if (res.ok) {
          const data = await res.json()
          setSystemStatus('operational')
          if (data.vector_db_document_count) {
            setDocCount(data.vector_db_document_count)
          }
        } else {
          setSystemStatus('error')
        }
      } catch {
        setSystemStatus('error')
      }
    }

    async function checkDomains() {
      try {
        const res = await fetch('/domains')
        if (res.ok) {
          const data = await res.json()
          if (data.count != null) {
            setDomainCount(data.count)
          }
        }
      } catch { /* keep fallback */ }
    }

    checkHealth()
    checkDomains()
    const timer = setInterval(() => {
      checkHealth()
      checkDomains()
    }, 30000)
    return () => clearInterval(timer)
  }, [])

  // ── Theme Engine State (Dark / Light Mode) ─────────────────────────────────
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('apollo-theme')
    if (saved) return saved
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
    localStorage.setItem('apollo-theme', theme)
  }, [theme])

  const toggleTheme = useCallback(() => {
    playClickSound()
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'))
  }, [])
  
  // ── Load History from IndexedDB ───────────────────────────────────────────
  useEffect(() => {
    async function loadSessions() {
      try {
        let saved = await localforage.getItem('apollo-sessions')
        if (!saved) {
          const legacy = localStorage.getItem('apollo-sessions')
          if (legacy) {
             saved = JSON.parse(legacy)
          }
        }

        if (saved && Array.isArray(saved) && saved.length > 0) {
          const parsed = saved.map(s => ({
            ...s,
            updatedAt: new Date(s.updatedAt),
            messages: s.messages.map(m => ({ ...m, timestamp: new Date(m.timestamp) }))
          }))
          setSessions(parsed)
          setCurrentSessionId(parsed[0].id)
        }
      } catch (e) {
        console.error('Failed to load sessions:', e)
      } finally {
        setIsReady(true)
      }
    }
    loadSessions()
  }, [])

  const currentSessionIdRef = useRef(currentSessionId)
  useEffect(() => { currentSessionIdRef.current = currentSessionId }, [currentSessionId])

  // Derive messages from current session
  const messages = sessions.find(s => s.id === currentSessionId)?.messages || []

  // Custom setMessages proxy
  const setMessages = useCallback((updater) => {
    setSessions(prevSessions => {
      const activeId = currentSessionIdRef.current
      if (!activeId) return prevSessions
      
      return prevSessions.map(s => {
        if (s.id === activeId) {
          const nextMessages = typeof updater === 'function' ? updater(s.messages) : updater
          return { ...s, messages: nextMessages, updatedAt: new Date() }
        }
        return s
      })
    })
  }, [])

  const [isStreaming, setIsStreaming] = useState(false)
  const [showTyping, setShowTyping] = useState(false)
  
  // Streaming Refs
  const bottomRef = useRef(null)
  const chatContainerRef = useRef(null)
  const isAutoScrollEnabled = useRef(true)
  const streamedContentRef = useRef('')
  const displayedContentRef = useRef('')
  const assistantIdRef = useRef(null)
  const rafRef = useRef(null)

  // Input State
  const [query, setQuery] = useState('')
  const inputRef = useRef(null)
  const [editingMessageId, setEditingMessageId] = useState(null)
  const abortControllerRef = useRef(null)

  // Scope (domain filter) and Mode (HyDE toggle) — lifted from CommandInputBar
  // scope: 'all' | <domain string>  — maps to ChromaDB 'where' filter
  // mode:  'triage' | 'research'    — 'research' enables HyDE expansion
  const [scope, setScope] = useState('all')
  const [mode, setMode] = useState('triage')

  // Document Upload & Voice Input State
  const [uploadedDoc, setUploadedDoc] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const fileInputRef = useRef(null)
  const recognitionRef = useRef(null)

  // ── Auto-scroll ───────────────────────────────────────────────────────────
  const handleScroll = useCallback(() => {
    if (!chatContainerRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current
    isAutoScrollEnabled.current = scrollHeight - scrollTop - clientHeight < 100
  }, [])

  // ── Save History to IndexedDB ───────────────────────────────────────────
  useEffect(() => {
    if (isReady) {
      localforage.setItem('apollo-sessions', sessions).catch(console.error)
    }
  }, [sessions, isReady])

  useEffect(() => {
    if (isAutoScrollEnabled.current && chatContainerRef.current) {
      if (isStreaming) {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
      } else {
        chatContainerRef.current.scrollTo({
          top: chatContainerRef.current.scrollHeight,
          behavior: 'smooth'
        })
      }
    }
  }, [messages, showTyping, isStreaming])

  // ── Cleanup RAF ───────────────────────────────────────────────────────────
  useEffect(() => () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }, [])

  // ── RAF flush loop ────────────────────────────────────────────────────────
  const startFlush = useCallback(() => {
    const tick = () => {
      const id = assistantIdRef.current
      const targetContent = streamedContentRef.current
      let currentContent = displayedContentRef.current

      if (currentContent !== targetContent) {
        const diff = targetContent.length - currentContent.length
        const charsToAdd = Math.max(1, Math.floor(diff / 4))
        
        currentContent = targetContent.slice(0, currentContent.length + charsToAdd)
        displayedContentRef.current = currentContent
        
        if (id) {
          setMessages(prev =>
            prev.map(m => m.id === id ? { ...m, content: currentContent, isStreaming: true } : m)
          )
        }
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }, [setMessages])

  const stopFlush = useCallback(() => {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
  }, [])

  // ── Clear Chat ────────────────────────────────────────────────────────────
  const handleClearChat = useCallback(() => {
    playClickSound()
    stopFlush()
    setCurrentSessionId(null)
    setIsStreaming(false)
    setShowTyping(false)
    streamedContentRef.current = ''
    assistantIdRef.current = null
    isAutoScrollEnabled.current = true
  }, [stopFlush])

  // ── Delete Session ────────────────────────────────────────────────────────
  const handleDeleteSession = useCallback((sessionId) => {
    playClickSound()
    setSessions(prev => prev.filter(s => s.id !== sessionId))
    if (currentSessionIdRef.current === sessionId) {
      setCurrentSessionId(null)
      setIsStreaming(false)
      setShowTyping(false)
      streamedContentRef.current = ''
      assistantIdRef.current = null
      isAutoScrollEnabled.current = true
      stopFlush()
    }
  }, [stopFlush])

  // ── Stop Generation ────────────────────────────────────────────────────────
  const handleStopGeneration = useCallback(() => {
    playClickSound()
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    stopFlush()
    const aId = assistantIdRef.current
    if (aId) {
      const final = streamedContentRef.current
      setMessages(prev => prev.map(m => m.id === aId ? { ...m, content: final || '*(Generation stopped)*', isStreaming: false } : m))
    }
    setIsStreaming(false)
    setShowTyping(false)
    assistantIdRef.current = null
  }, [stopFlush, setMessages])

  // ── Edit Message ───────────────────────────────────────────────────────────
  const handleEditMessage = useCallback((messageId, content) => {
    playClickSound()
    const targetContent = content !== undefined ? content : (typeof messageId === 'string' ? messageId : '')
    const targetId = content !== undefined ? messageId : null

    setEditingMessageId(targetId)
    setQuery(targetContent || '')
    setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.focus()
        inputRef.current.style.height = 'auto'
        inputRef.current.style.height = `${inputRef.current.scrollHeight}px`
      }
    }, 50)
  }, [])

  const handleCancelEdit = useCallback(() => {
    setEditingMessageId(null)
    setQuery('')
    if (inputRef.current) {
      inputRef.current.style.height = 'auto'
    }
  }, [])

  // ── Document Upload ───────────────────────────────────────────────────────
  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setIsUploading(true)
    playClickSound()
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await fetch('/upload_context', { method: 'POST', body: formData })
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || 'Failed to upload document')
      }
      const data = await res.json()
      setUploadedDoc(data)
    } catch (err) {
      console.error('File upload error:', err)
      alert(`Failed to process document: ${err.message}`)
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  // ── Voice Input (Web Speech API) ──────────────────────────────────────────
  const handleToggleVoice = () => {
    playClickSound()
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      alert('Voice input is not supported in this browser. Please use Google Chrome or Microsoft Edge.')
      return
    }
    if (isListening) {
      recognitionRef.current?.stop()
      setIsListening(false)
      return
    }

    try {
      const recognition = new SpeechRecognition()
      recognition.continuous = false
      recognition.interimResults = false
      recognition.lang = 'en-US'

      recognition.onstart = () => setIsListening(true)
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript
        setQuery(prev => (prev ? `${prev} ${transcript}` : transcript))
        setIsListening(false)
      }
      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error)
        setIsListening(false)
      }
      recognition.onend = () => setIsListening(false)

      recognitionRef.current = recognition
      recognition.start()
    } catch (err) {
      console.error('Speech recognition start error:', err)
      setIsListening(false)
    }
  }

  // ── Submit Handler ────────────────────────────────────────────────────────
  const handleSubmit = async (e, customQuery) => {
    e?.preventDefault()
    const textToSend = customQuery || query
    if (!textToSend.trim() || isStreaming) return
    playClickSound()

    let activeId = currentSessionIdRef.current
    if (editingMessageId) {
      setMessages(prev => {
        const idx = prev.findIndex(m => m.id === editingMessageId)
        return idx !== -1 ? prev.slice(0, idx) : prev
      })
      setEditingMessageId(null)
    }
    
    const newTitle = textToSend.trim().split(' ').slice(0, 5).join(' ') + (textToSend.trim().split(' ').length > 5 ? '...' : '')
    if (!activeId) {
      activeId = `sess-${Date.now()}`
      const newSession = {
        id: activeId,
        title: newTitle,
        messages: [],
        updatedAt: new Date()
      }
      setSessions(prev => [newSession, ...prev])
      setCurrentSessionId(activeId)
      currentSessionIdRef.current = activeId
    } else {
      setSessions(prev => prev.map(s => s.id === activeId ? { ...s, title: newTitle, updatedAt: new Date() } : s))
    }

    setQuery('')
    if (inputRef.current) {
      inputRef.current.style.height = 'auto'
    }

    const userMsg = { id: `u-${Date.now()}`, role: 'user', content: textToSend, timestamp: new Date() }
    const aId = `a-${Date.now()}`
    assistantIdRef.current = aId
    streamedContentRef.current = ''
    displayedContentRef.current = ''

    setMessages(prev => [...prev, userMsg])
    setShowTyping(true)
    setIsStreaming(true)
    isAutoScrollEnabled.current = true

    try {
      const currentHistory = sessions.find(s => s.id === activeId)?.messages || []
      const messagesPayload = [...currentHistory, userMsg].map(m => ({ role: m.role, content: m.content }))

      abortControllerRef.current = new AbortController()
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messagesPayload,
          uploaded_context: uploadedDoc?.text || null,
          // domain_filter: null means search entire corpus (All Folders).
          // Any other value restricts ChromaDB retrieval to that domain.
          domain_filter: scope !== 'all' ? scope : null,
          // use_hyde: Deep Research mode enables HyDE expansion for better
          // retrieval on complex queries. Triage Mode skips it for speed.
          use_hyde: mode === 'research',
        }),
        signal: abortControllerRef.current.signal,
      })

      if (!res.ok) throw new Error(`Server error: HTTP ${res.status}`)

      setShowTyping(false)
      const topSimilarityStr = res.headers.get('X-RAG-Top-Similarity')
      const topSimilarity = topSimilarityStr ? parseFloat(topSimilarityStr) : 1.0
      const isLowConfidence = topSimilarity < 0.50 && topSimilarity >= 0.35
      
      setMessages(prev => [...prev, { id: aId, role: 'assistant', content: '', isStreaming: true, timestamp: new Date(), isLowConfidence }])
      startFlush()

      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buf = ''
      let streamEnded = false

      const finaliseStreaming = () => {
        if (streamEnded) return
        streamEnded = true
        stopFlush()
        const final = streamedContentRef.current
        setMessages(prev => prev.map(m => m.id === aId ? { ...m, content: final, isStreaming: false } : m))
        setIsStreaming(false)
        assistantIdRef.current = null
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          finaliseStreaming()
          playChimeSound()
          break
        }
        buf += decoder.decode(value, { stream: true })
        const events = buf.split('\n\n')
        buf = events.pop() || '' 

        for (const event of events) {
          if (!event.startsWith('data: ')) continue
          let payload
          try {
            payload = JSON.parse(event.slice(6).trim())
          } catch {
            continue
          }
          if (payload.type === 'token') {
            streamedContentRef.current += payload.content
          } else if (payload.type === 'end') {
            finaliseStreaming()
          } else if (payload.type === 'error') {
            throw new Error(payload.content)
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return
      stopFlush()
      setShowTyping(false)
      setIsStreaming(false)
      const errMsg = {
        id: `err-${Date.now()}`, role: 'assistant', isStreaming: false, timestamp: new Date(),
        content: `**⚠️ Connection Error**\n\n${err.message}`,
        isError: true,
        retryQuery: textToSend,
      }
      setMessages(prev => [...prev.filter(m => m.id !== aId), errMsg])
      assistantIdRef.current = null
      setSystemStatus('error')
    }
  }

  // ── Retry Failed Message ─────────────────────────────────────────────────────
  const handleRetry = useCallback((errorMessageId, retryQuery) => {
    playClickSound()
    setMessages(prev => prev.filter(m => m.id !== errorMessageId))
    setQuery(retryQuery)
    setSystemStatus('operational')
    setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.focus()
        inputRef.current.style.height = 'auto'
        inputRef.current.style.height = `${inputRef.current.scrollHeight}px`
      }
    }, 50)
  }, [setMessages])

  // Handle Preset Starter Card Click
  const handleSelectStarterPrompt = (promptText) => {
    setQuery(promptText)
    handleSubmit(null, promptText)
  }

  return (
    <div className="flex h-screen w-full bg-slate-50 dark:bg-[#09090b] overflow-hidden font-sans text-slate-900 dark:text-zinc-100 transition-colors duration-250">
      
      {/* ── Left Sidebar ─────────────────────────────────────────────────── */}
      <Sidebar 
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={(id) => { playClickSound(); setCurrentSessionId(id) }}
        onClearChat={handleClearChat} 
        onDeleteSession={handleDeleteSession}
        systemStatus={systemStatus}
        theme={theme}
        onToggleTheme={toggleTheme}
        docCount={docCount}
        domainCount={domainCount}
      />

      {/* ── Main Workstation Canvas ───────────────────────────────────────── */}
      <main className="flex-1 flex flex-col relative overflow-hidden bg-slate-50 dark:bg-[#09090b] z-10">
        
        {/* Background Grid Pattern */}
        <div 
          className="absolute inset-0 bg-grid-pattern bg-grid pointer-events-none z-0 opacity-40 dark:opacity-30"
        />

        {/* Processing Glow (Active during streaming) */}
        {isStreaming && (
          <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-emerald-500/10 dark:bg-[#00E599]/10 rounded-full blur-[140px] animate-pulse pointer-events-none z-0" />
        )}

        {/* ── Header Offline Error Alert ──────────────────────────────────── */}
        {systemStatus === 'error' && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400 px-4 py-2 rounded-full flex items-center gap-2 backdrop-blur-md shadow-md text-xs font-mono font-medium">
            <AlertTriangle size={15} />
            <span>Apollo Backend Offline — Reconnecting...</span>
          </div>
        )}

        {/* ── Chat Canvas OR Quick Action Hero Screen ──────────────────────── */}
        <div 
          ref={chatContainerRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto px-4 sm:px-12 lg:px-24 xl:px-44 pt-10 pb-6 z-10 flex flex-col"
        >
          {messages.length === 0 ? (
            // Hero Screen with 2x2 Quick Action Starter Cards
            <HeroStarterCards onSelectPrompt={handleSelectStarterPrompt} />
          ) : (
            // Conversation Message List
            <div className="flex flex-col space-y-6 pb-16">
              {messages.map((msg) => (
                <div key={msg.id} className="animate-fade-in-up">
                  <MessageBubble
                    message={msg}
                    onEdit={msg.role === 'user' && !isStreaming ? handleEditMessage : null}
                    onRetry={msg.isError && !isStreaming ? handleRetry : null}
                  />
                </div>
              ))}
              {showTyping && <ApolloThinkingIndicator />}

              <div ref={bottomRef} aria-hidden="true" className="h-4" />
            </div>
          )}
        </div>

        {/* ── Floating Command Input Bar ──────────────────────────────────── */}
        <CommandInputBar
          query={query}
          setQuery={setQuery}
          onSubmit={handleSubmit}
          isStreaming={isStreaming}
          onStopGeneration={handleStopGeneration}
          inputRef={inputRef}
          fileInputRef={fileInputRef}
          onFileSelect={handleFileSelect}
          isUploading={isUploading}
          uploadedDoc={uploadedDoc}
          onRemoveUploadedDoc={() => setUploadedDoc(null)}
          isListening={isListening}
          onToggleVoice={handleToggleVoice}
          editingMessageId={editingMessageId}
          onCancelEdit={handleCancelEdit}
          scope={scope}
          onScopeChange={setScope}
          mode={mode}
          onModeChange={setMode}
        />

      </main>
    </div>
  )
}
