/**
 * ChatInterface — The main chat panel
 *
 * This is the core stateful component. It manages:
 *   - The messages array (user + assistant message objects)
 *   - The streaming state machine (idle → streaming → done)
 *   - Reading the Server-Sent Events stream from /chat
 *   - Auto-scrolling to the latest message
 *
 * SSE STREAMING IMPLEMENTATION:
 *   We use fetch() with ReadableStream instead of EventSource because
 *   EventSource only supports GET requests. Our /chat endpoint is POST.
 *
 *   The stream reader decodes raw bytes → UTF-8 string → splits on '\n\n'
 *   (SSE message delimiter) → parses each "data: {...}" JSON payload →
 *   appends the token to the current assistant message in state.
 *
 *   React batches rapid state updates, so we use a ref (streamedContentRef)
 *   to accumulate the full streamed string without triggering re-renders on
 *   every single token. We then flush to state every ~50ms via a RAF loop.
 *
 * WHY THIS MATTERS FOR UX:
 *   Updating React state on every token would trigger hundreds of re-renders
 *   per second, causing visible jank. The ref-buffered + RAF-flushed approach
 *   gives us smooth, 60fps text animation.
 */
import { useState, useRef, useEffect, useCallback } from 'react'
import MessageBubble from './MessageBubble'
import TypingIndicator from './TypingIndicator'
import InputBar from './InputBar'

import WelcomeScreen from './WelcomeScreen'

const API_BASE = '' // Empty = use Vite proxy (relative URL)

export default function ChatInterface({ isDisabled }) {
  const [messages, setMessages]     = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [showTyping, setShowTyping]  = useState(false)

  const bottomRef             = useRef(null)   // Scroll anchor at bottom of message list
  const streamedContentRef    = useRef('')      // Accumulates streamed tokens between RAF flushes
  const currentAssistantIdRef = useRef(null)   // ID of the assistant message being streamed
  const rafRef                = useRef(null)   // RequestAnimationFrame handle

  // ── Auto-scroll to bottom when new content arrives ───────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, showTyping])

  // ── Cleanup RAF on unmount ────────────────────────────────────────────────
  useEffect(() => {
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [])

  // ── RAF flush loop: syncs the ref buffer → React state ───────────────────
  // Called once per animation frame while streaming, NOT on every token.
  // This is the key technique that prevents jank during rapid token output.
  const startFlushLoop = useCallback(() => {
    const flush = () => {
      const content = streamedContentRef.current
      const id = currentAssistantIdRef.current
      if (id && content !== undefined) {
        setMessages(prev => prev.map(m =>
          m.id === id ? { ...m, content, isStreaming: true } : m
        ))
      }
      rafRef.current = requestAnimationFrame(flush)
    }
    rafRef.current = requestAnimationFrame(flush)
  }, [])

  const stopFlushLoop = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [])

  // ── Core: Submit query and handle SSE stream ──────────────────────────────
  const handleSubmit = useCallback(async (query) => {
    if (isStreaming) return

    // 1. Add user message to state immediately (optimistic update)
    const userMsg = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: query,
      timestamp: new Date(),
    }

    // 2. Prepare a placeholder assistant message (will be filled by stream)
    const assistantId = `assistant-${Date.now()}`
    currentAssistantIdRef.current = assistantId
    streamedContentRef.current = ''

    setMessages(prev => [...prev, userMsg])
    setShowTyping(true)
    setIsStreaming(true)

    try {
      // 3. Fire the POST request to /chat
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, n_results: 3 }),
      })

      if (!response.ok) {
        throw new Error(`API error: HTTP ${response.status}`)
      }

      // 4. Hide typing indicator, add empty assistant bubble, start RAF flush
      setShowTyping(false)
      const assistantMsg = {
        id: assistantId,
        role: 'assistant',
        content: '',
        isStreaming: true,
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, assistantMsg])
      startFlushLoop()

      // 5. Read the stream byte by byte
      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = '' // Partial SSE message buffer

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        // Decode bytes to string and append to buffer
        buffer += decoder.decode(value, { stream: true })

        // SSE messages are delimited by '\n\n'
        const events = buffer.split('\n\n')
        // Keep the last (potentially incomplete) chunk in the buffer
        buffer = events.pop() || ''

        for (const event of events) {
          // Each event looks like: "data: {...json...}"
          if (!event.startsWith('data: ')) continue
          const jsonStr = event.slice(6).trim() // Remove "data: " prefix
          if (!jsonStr) continue

          try {
            const payload = JSON.parse(jsonStr)

            if (payload.type === 'token') {
              // Accumulate token into ref — NO state update here (that's the point)
              streamedContentRef.current += payload.content

            } else if (payload.type === 'end') {
              // Stream complete — stop the flush loop and do one final state sync
              stopFlushLoop()
              const finalContent = streamedContentRef.current
              setMessages(prev => prev.map(m =>
                m.id === assistantId
                  ? { ...m, content: finalContent, isStreaming: false }
                  : m
              ))
              setIsStreaming(false)
              currentAssistantIdRef.current = null

            } else if (payload.type === 'error') {
              throw new Error(payload.content)
            }
          } catch (parseError) {
            // Silently skip malformed SSE events (e.g., partial JSON)
            console.warn('[Apollo] SSE parse error:', parseError)
          }
        }
      }

    } catch (err) {
      // Handle network errors or generation failures gracefully
      console.error('[Apollo] Stream error:', err)
      stopFlushLoop()
      setShowTyping(false)
      setIsStreaming(false)

      const errorMsg = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `**⚠️ Connection Error**\n\n${err.message}\n\nPlease ensure the Apollo backend server is running:\n\`\`\`\nuvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1\n\`\`\``,
        isStreaming: false,
        timestamp: new Date(),
      }

      setMessages(prev => {
        // Remove the empty placeholder if it was already added
        const filtered = prev.filter(m => m.id !== currentAssistantIdRef.current)
        return [...filtered, errorMsg]
      })
      currentAssistantIdRef.current = null
    }
  }, [isStreaming, startFlushLoop, stopFlushLoop])

  // ── Clear Chat ────────────────────────────────────────────────────────────
  const handleClearChat = useCallback(() => {
    setMessages([])
    setIsStreaming(false)
    setShowTyping(false)
    streamedContentRef.current = ''
    stopFlushLoop()
  }, [stopFlushLoop])

  return (
    // Export clearChat so App.jsx can wire the sidebar button
    // (returned as second element via a simple wrapper trick)
    <ChatInterfaceInner
      messages={messages}
      isStreaming={isStreaming}
      showTyping={showTyping}
      isDisabled={isDisabled}
      onSubmit={handleSubmit}
      onClearChat={handleClearChat}
      bottomRef={bottomRef}
    />
  )
}

// Expose clearChat via a custom hook-friendly pattern
ChatInterface.displayName = 'ChatInterface'

// ── Inner render component (pure layout) ──────────────────────────────────
function ChatInterfaceInner({
  messages, isStreaming, showTyping, isDisabled,
  onSubmit, onClearChat, bottomRef
}) {
  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      {/* Message list */}
      <div
        id="apollo-message-list"
        className="flex-1 overflow-y-auto px-6 py-6 space-y-5 flex flex-col"
        role="log"
        aria-label="Apollo conversation"
        aria-live="polite"
      >
        {messages.length === 0 ? (
          <WelcomeScreen />
        ) : (
          <>
            {messages.map(msg => (
              <MessageBubble key={msg.id} message={msg} />
            ))}

            {/* Typing indicator shown while waiting for first token */}
            {showTyping && <TypingIndicator />}

            {/* Invisible scroll anchor */}
            <div ref={bottomRef} aria-hidden="true" />
          </>
        )}
      </div>

      {/* Input bar */}
      <InputBar
        onSubmit={onSubmit}
        isStreaming={isStreaming}
        isDisabled={isDisabled}
      />
    </div>
  )
}

// Export clearChat handler for Sidebar wiring
export { ChatInterface as default }
