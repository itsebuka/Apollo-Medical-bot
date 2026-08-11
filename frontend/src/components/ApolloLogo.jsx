/**
 * ApolloLogo — SVG brand mark for the Apollo Medical System
 *
 * Engineering note: Inline SVG (vs <img>) keeps the logo offline-safe
 * and allows CSS animation/coloring via currentColor.
 */
export default function ApolloLogo({ size = 32, animated = false }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={animated ? 'animate-pulse text-neon' : 'text-neon'}
      aria-label="Apollo Medical System Logo"
    >
      {/* Staff */}
      <path d="M12 2v20" />
      {/* Snake winding */}
      <path d="M8 6.5C8 5 9.5 4 11 4s3 1 3 2.5-1.5 2.5-3 3.5S8 11.5 8 13s1.5 2.5 3 3.5 3 1.5 3 3-1.5 2.5-3 2.5" />
      {/* Snake eye */}
      <circle cx="14" cy="5" r="0.5" fill="currentColor" stroke="none" />
    </svg>
  )
}
