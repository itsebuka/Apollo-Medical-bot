/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dynamic Theme Color Mapping
        'apollo-base':     'var(--bg-base)',
        'apollo-surface':  'var(--bg-surface)',
        'apollo-elevated': 'var(--bg-elevated)',
        'apollo-glass':    'var(--bg-glass)',
        'apollo-border':   'var(--border-color)',
        'apollo-subtle':   'var(--border-subtle)',
        'neon':            'var(--accent)',
        'neon-hover':      'var(--accent-hover)',
        'neon-glow':       'var(--accent-alpha-15)',
        'neon-glow-strong':'var(--accent-alpha-30)',
        'text-primary':    'var(--text-primary)',
        'text-secondary':  'var(--text-secondary)',
        'text-muted':      'var(--text-muted)',
        'danger':          '#ef4444',
        'warning':         '#f59e0b',
      },
      fontFamily: {
        sans:  ['Inter', 'Geist', 'system-ui', 'sans-serif'],
        mono:  ['JetBrains Mono', 'Fira Code', 'Roboto Mono', 'monospace'],
      },
      animation: {
        'pulse-neon':   'pulseNeon 2s ease-in-out infinite',
        'fade-in-up':   'fadeInUp 0.4s ease-out forwards',
        'fade-in':      'fadeIn 0.3s ease-out forwards',
        'cursor-blink': 'cursorBlink 0.8s step-end infinite',
        'scan-line':    'scanLine 3s ease-in-out infinite',
        'spin-slow':    'spin 3s linear infinite',
        'dot-bounce':   'dotBounce 1.4s ease-in-out infinite',
        'pulse-ring':   'pulseRing 1.5s cubic-bezier(0.215, 0.61, 0.355, 1) infinite',
      },
      keyframes: {
        pulseNeon: {
          '0%, 100%': { boxShadow: '0 0 5px var(--accent), 0 0 10px var(--accent)' },
          '50%':      { boxShadow: '0 0 20px var(--accent), 0 0 40px var(--accent)' },
        },
        pulseRing: {
          '0%': { transform: 'scale(0.95)', opacity: '0.8' },
          '50%': { transform: 'scale(1.3)', opacity: '0' },
          '100%': { transform: 'scale(0.95)', opacity: '0' },
        },
        fadeInUp: {
          '0%':   { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        cursorBlink: {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0' },
        },
        scanLine: {
          '0%':   { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(400%)' },
        },
        dotBounce: {
          '0%, 80%, 100%': { transform: 'scale(0)', opacity: '0.3' },
          '40%':           { transform: 'scale(1)', opacity: '1' },
        },
      },
      boxShadow: {
        'neon-sm':  '0 0 8px var(--accent-alpha-30)',
        'neon-md':  '0 0 16px var(--accent-alpha-30), 0 0 32px var(--accent-alpha-15)',
        'neon-lg':  '0 0 24px var(--accent-alpha-30), 0 0 48px var(--accent-alpha-15)',
        'surface':  'var(--shadow-surface)',
        'message':  'var(--shadow-main)',
      },
      backgroundImage: {
        'grid-pattern': `linear-gradient(var(--grid-line) 1px, transparent 1px),
                         linear-gradient(90deg, var(--grid-line) 1px, transparent 1px)`,
      },
      backgroundSize: {
        'grid': '40px 40px',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
