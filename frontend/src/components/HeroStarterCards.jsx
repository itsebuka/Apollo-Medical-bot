import { motion } from 'framer-motion'
import { Dna, Pill, Microscope, Stethoscope, Sparkles } from 'lucide-react'
import ApolloLogo from './ApolloLogo'

const STARTER_CARDS = [
  {
    id: 'diff-dx',
    icon: Dna,
    category: 'Differential Diagnosis',
    title: 'Autoimmune Encephalitis',
    snippet: 'Synthesize signs for Autoimmune Encephalitis...',
    prompt: 'Synthesize signs and symptoms for Autoimmune Encephalitis including diagnostic criteria, antibody markers, and MRI/CSF findings.',
  },
  {
    id: 'drug-interaction',
    icon: Pill,
    category: 'Drug-Drug Interactions',
    title: 'Warfarin & St. John\'s Wort',
    snippet: 'Analyze interactions between Warfarin & St. John\'s Wort...',
    prompt: 'Analyze interactions between Warfarin and St. John\'s Wort including CYP3A4/CYP2C9 induction mechanisms and INR clinical monitoring.',
  },
  {
    id: 'genomic-analysis',
    icon: Microscope,
    category: 'Genomic Analysis',
    title: 'HBV ORF Overlap Constraints',
    snippet: 'Compare HBV ORF overlap mutation constraints...',
    prompt: 'Compare Hepatitis B Virus (HBV) ORF overlap mutation constraints, secondary RNA folding structures, and evolutionary selection pressures.',
  },
  {
    id: 'surgical-landmarks',
    icon: Stethoscope,
    category: 'Surgical Landmarks',
    title: 'Recurrent Laryngeal Nerve',
    snippet: 'Map recurrent laryngeal nerve surgical relations...',
    prompt: 'Map recurrent laryngeal nerve surgical anatomical relations, Berry\'s ligament landmarks, and intraoperative nerve identification protocols.',
  },
]

export default function HeroStarterCards({ onSelectPrompt }) {
  return (
    <div className="w-full max-w-4xl mx-auto flex flex-col items-center justify-center text-center my-auto py-6 px-4 animate-fade-in">
      
      {/* ── Technical Status Badge ────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 dark:bg-[#00E599]/10 border border-emerald-500/30 dark:border-[#00E599]/30 text-emerald-600 dark:text-[#00E599] text-xs font-mono font-medium tracking-wider shadow-sm mb-6"
      >
        <span className="w-2 h-2 rounded-full bg-emerald-500 dark:bg-[#00E599] animate-pulse" />
        <span>APOLLO ENGINE v2.4 &nbsp;|&nbsp; OFFLINE VECTOR DB ACTIVE</span>
      </motion.div>

      {/* ── Logo & Hero Title with Vertical Gradient ──────────────────────── */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.35, delay: 0.05 }}
        className="flex flex-col items-center"
      >
        <div className="mb-4 p-3 rounded-2xl bg-slate-100 dark:bg-zinc-900 border border-slate-200 dark:border-zinc-800 shadow-sm text-emerald-600 dark:text-[#00E599]">
          <ApolloLogo size={48} />
        </div>

        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight bg-gradient-to-b from-slate-900 via-slate-800 to-slate-600 dark:from-white dark:via-zinc-200 dark:to-zinc-400 bg-clip-text text-transparent mb-3">
          Clinical Decision Support
        </h1>

        <p className="text-slate-600 dark:text-zinc-400 text-sm sm:text-base max-w-xl mx-auto mb-10 font-sans leading-relaxed">
          Select a clinical prompt below or type your case inquiry to query Apollo's localized offline vector knowledge base.
        </p>
      </motion.div>

      {/* ── 2x2 Interactive Starter Cards Grid ───────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full text-left">
        {STARTER_CARDS.map((card, idx) => {
          const Icon = card.icon
          return (
            <motion.div
              key={card.id}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.1 + idx * 0.05 }}
              onClick={() => onSelectPrompt(card.prompt)}
              onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onSelectPrompt(card.prompt)}
              role="button"
              tabIndex={0}
              className="group relative p-4 rounded-2xl bg-white dark:bg-zinc-900/80 border border-slate-200 dark:border-zinc-800/80 shadow-sm hover:shadow-md hover:border-emerald-500/40 dark:hover:border-[#00E599]/40 transition-all duration-200 cursor-pointer flex flex-col justify-between overflow-hidden focus:outline-none focus:ring-2 focus:ring-emerald-500/50 dark:focus:ring-[#00E599]/50"
            >
              {/* Corner hover glow effect */}
              <div className="absolute -top-12 -right-12 w-24 h-24 bg-emerald-500/10 dark:bg-[#00E599]/10 rounded-full blur-xl group-hover:scale-150 transition-transform duration-300 pointer-events-none" />

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] font-mono font-medium text-emerald-600 dark:text-[#00E599] uppercase tracking-wider flex items-center gap-1.5">
                    <Icon size={14} className="stroke-[2]" />
                    {card.category}
                  </span>
                  <Sparkles size={13} className="text-slate-300 dark:text-zinc-600 group-hover:text-emerald-500 dark:group-hover:text-[#00E599] transition-colors" />
                </div>

                <h3 className="text-sm font-semibold text-slate-900 dark:text-zinc-100 group-hover:text-emerald-600 dark:group-hover:text-[#00E599] transition-colors mb-1">
                  {card.title}
                </h3>

                <p className="text-xs text-slate-500 dark:text-zinc-400 font-sans leading-relaxed line-clamp-2">
                  "{card.snippet}"
                </p>
              </div>

              <div className="mt-3 pt-2.5 border-t border-slate-100 dark:border-zinc-800/60 flex items-center justify-between text-[11px] font-mono text-slate-400 dark:text-zinc-500 group-hover:text-slate-700 dark:group-hover:text-zinc-300">
                <span>Run Preset Query</span>
                <span className="text-emerald-600 dark:text-[#00E599] font-sans group-hover:translate-x-1 transition-transform">
                  &rarr;
                </span>
              </div>
            </motion.div>
          )
        })}
      </div>

    </div>
  )
}
