import ApolloLogo from './ApolloLogo'

const NCD_TAGS = ['Hypertension', 'Type 2 Diabetes', 'Dietary Guidance', 'Stroke Prevention', 'Cardiovascular Risk']
const ENGINE_STACK = [
  { label: 'LLM',       value: 'Llama-3 8B Q4_K_M' },
  { label: 'Runtime',   value: 'llama.cpp (CPU)' },
  { label: 'RAG DB',    value: 'ChromaDB (local)' },
  { label: 'Embeddings',value: 'MiniLM-L6-v2' },
  { label: 'Context',   value: '8k tokens' },
]

export default function WelcomeScreen() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center overflow-y-auto py-10 animate-fade-in-up">
      <div className="flex flex-col items-center mb-8">
        <ApolloLogo size={64} animated={false} />
        <h1 className="mt-4 text-neon font-bold text-2xl tracking-tight text-neon-glow">
          Apollo
        </h1>
        <p className="text-text-muted text-xs font-mono tracking-widest uppercase mt-1">
          Medical Triage
        </p>
      </div>

      <div className="w-full max-w-sm bg-apollo-surface border border-apollo-border rounded-xl overflow-hidden shadow-message">
        {/* ── NCD Focus Areas ──────────────────────────────────────────────── */}
        <div className="p-6 border-b border-apollo-border">
          <h2 className="text-[10px] font-mono font-semibold tracking-[0.15em] text-text-muted
                         uppercase mb-3">
            Clinical Domains
          </h2>
          <div className="flex flex-wrap gap-1.5">
            {NCD_TAGS.map((tag) => (
              <span
                key={tag}
                className="text-[11px] font-mono text-neon/80 bg-neon/5 border border-neon/20
                           rounded-full px-2.5 py-0.5"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>

        {/* ── Engine Info ──────────────────────────────────────────────────── */}
        <div className="p-6 space-y-3">
          <h2 className="text-[10px] font-mono font-semibold tracking-[0.15em] text-text-muted
                         uppercase mb-3">
            Engine Stack
          </h2>
          {ENGINE_STACK.map(({ label, value }) => (
            <div key={label} className="flex items-baseline justify-between">
              <span className="text-[11px] font-mono text-text-muted">{label}</span>
              <span className="text-[11px] font-mono text-text-secondary">{value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
