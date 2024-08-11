/**
 * InputBar — Query input and submission control
 *
 * Features:
 *   - Auto-expanding textarea (grows to 5 lines max)
 *   - Submit on Enter (Shift+Enter for newline)
 *   - Character count with neon warning at limit
 *   - Disabled state during streaming (prevents concurrent requests)
 *   - Neon green send button with glow on hover
 *   - Example prompt suggestions shown when chat is empty
 *
 * Why a textarea instead of <input>?
 *   Medical queries are often long and multi-part. A textarea that auto-grows
 *   respects the user's writing flow without forcing horizontal scrolling.
 */
import { useState, useRef, useEffect } from 'react'

const MAX_CHARS = 1000

const EXAMPLE_PROMPTS = [
  "What swallow food is best for a diabetic patient in Nigeria?",
  "What are the FAST signs of a stroke I should watch out for?",
  "My BP reading is 160/100. What should I do?",
  "Suggest a low-GI Nigerian meal plan for my newly diagnosed mother.",
]

export default function InputBar({ onSubmit, isStreaming, isDisabled }) {
  const [query, setQuery] = useState('')
  const textareaRef = useRef(null)

  // Auto-resize textarea height based on content
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px` // max 5 lines ≈ 120px
  }, [query])

  const handleSubmit = () => {
    const trimmed = query.trim()
    if (!trimmed || isStreaming || isDisabled) return
    onSubmit(trimmed)
    setQuery('')
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e) => {
    // Enter = submit; Shift+Enter = newline
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleExampleClick = (prompt) => {
    setQuery(prompt)
    textareaRef.current?.focus()
  }

  const charCount = query.length
  const isNearLimit = charCount > MAX_CHARS * 0.85
  const canSubmit = query.trim().length > 2 && !isStreaming && !isDisabled

  return (
    <div id="apollo-input-bar" className="border-t border-apollo-border bg-apollo-surface">
      {/* Example prompts (shown when textarea is empty) */}
      {query.length === 0 && !isStreaming && (
        <div className="px-4 pt-3 pb-1 flex flex-wrap gap-2">
          {EXAMPLE_PROMPTS.map((prompt, i) => (
            <button
              key={i}
              id={`example-prompt-${i}`}
              onClick={() => handleExampleClick(prompt)}
              className="text-[11px] font-mono text-text-secondary border border-apollo-border
                         rounded-full px-3 py-1 hover:border-neon/50 hover:text-neon
                         hover:bg-neon/5 transition-all duration-200 text-left"
              disabled={isDisabled}
            >
              {prompt}
            </button>
          ))}
        </div>
      )}

      {/* Input row */}
      <div className="flex items-end gap-3 px-4 py-3">
        {/* Textarea */}
        <div
          className={`flex-1 relative bg-apollo-elevated border rounded-xl
                      transition-all duration-300 overflow-hidden
                      ${isStreaming || isDisabled
                        ? 'border-apollo-muted opacity-60'
                        : 'border-apollo-border hover:border-neon/30 focus-within:border-neon/60'
                      }
                      focus-within:shadow-neon-sm`}
        >
          <textarea
            ref={textareaRef}
            id="apollo-query-input"
            value={query}
            onChange={(e) => setQuery(e.target.value.slice(0, MAX_CHARS))}
            onKeyDown={handleKeyDown}
            placeholder={
              isStreaming
                ? 'Apollo is responding...'
                : isDisabled
                ? 'Start the backend server to begin...'
                : 'Ask Apollo about NCDs, dietary guidance, symptoms...'
            }
            disabled={isStreaming || isDisabled}
            rows={1}
            aria-label="Medical query input"
            className="w-full bg-transparent text-text-primary text-sm
                       placeholder-text-muted font-sans resize-none
                       px-4 py-3 pr-14 leading-relaxed
                       disabled:cursor-not-allowed"
          />

          {/* Character count (inside textarea, bottom-right) */}
          {query.length > 0 && (
            <span
              className={`absolute bottom-2 right-3 text-[10px] font-mono pointer-events-none
                          transition-colors duration-200
                          ${isNearLimit ? 'text-warning' : 'text-text-muted'}`}
            >
              {charCount}/{MAX_CHARS}
            </span>
          )}
        </div>

        {/* Send button */}
        <button
          id="apollo-send-btn"
          onClick={handleSubmit}
          disabled={!canSubmit}
          aria-label="Send query to Apollo"
          className={`flex-shrink-0 w-11 h-11 rounded-xl flex items-center justify-center
                      font-bold transition-all duration-300 focus:outline-none
                      ${canSubmit
                        ? 'bg-neon text-apollo-black hover:shadow-neon-md hover:scale-105 active:scale-95'
                        : 'bg-apollo-muted text-text-muted cursor-not-allowed'
                      }`}
        >
          {isStreaming ? (
            /* Spinner while streaming */
            <svg className="w-4 h-4 animate-spin-slow" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z"/>
            </svg>
          ) : (
            /* Send arrow */
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M22 2L11 13" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M22 2L15 22L11 13L2 9L22 2Z" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )}
        </button>
      </div>

      {/* Footer note */}
      <p className="text-center text-[10px] text-text-muted font-mono pb-2 px-4">
        Apollo provides triage support only.{' '}
        <span className="text-danger/70">Always consult a qualified physician.</span>
      </p>
    </div>
  )
}
