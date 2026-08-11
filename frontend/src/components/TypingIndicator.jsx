/**
 * TypingIndicator — Animated "Apollo is thinking" indicator
 *
 * Displayed while the LLM is generating a response (between
 * sending the request and receiving the first token).
 *
 * Uses three bouncing neon dots — a universally recognized
 * "loading" metaphor in chat interfaces.
 */
export default function TypingIndicator() {
  return (
    <div
      id="apollo-typing-indicator"
      className="flex items-start gap-3 animate-fade-in"
      role="status"
      aria-label="Apollo is generating a response"
    >
      {/* Apollo avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-apollo-surface border border-neon/30
                      flex items-center justify-center shadow-neon-sm">
        <span className="text-neon text-xs font-mono font-bold">A</span>
      </div>

      {/* Bouncing dots container */}
      <div className="bg-apollo-elevated border border-apollo-border rounded-2xl rounded-tl-sm
                      px-4 py-3 flex items-center gap-1.5 shadow-message">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-2 h-2 rounded-full bg-neon animate-dot-bounce"
            style={{ animationDelay: `${i * 0.16}s` }}
          />
        ))}
        <span className="ml-2 text-xs text-text-muted font-mono">
          Retrieving context...
        </span>
      </div>
    </div>
  )
}
