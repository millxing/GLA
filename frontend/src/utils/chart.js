/** "Nice" evenly spaced axis ticks (steps of 1, 2, 2.5, 5 × 10^n) covering [min, max]; `integer` forbids fractional steps. */
export function niceTicks(min, max, targetCount, { integer = false } = {}) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1]
  if (min === max) {
    min -= 1
    max += 1
  }
  const rawStep = (max - min) / Math.max(targetCount, 1)
  const magnitude = 10 ** Math.floor(Math.log10(rawStep))
  const normalized = rawStep / magnitude
  let step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.5 ? 2.5 : normalized <= 5 ? 5 : 10) * magnitude
  // Integer mode: avoid fractional steps like 2.5 or 0.5 (ticks such as 87.5)
  // (the only fractional step ≥ 1 is 2.5, which doubles to 5)
  if (integer && !Number.isInteger(step)) step = step < 1 ? 1 : step * 2
  const start = Math.floor(min / step) * step
  const end = Math.ceil(max / step) * step
  const ticks = []
  for (let t = start; t <= end + step / 2; t += step) {
    ticks.push(Number(t.toFixed(6)))
  }
  return ticks
}

/**
 * Recharts <Tooltip wrapperStyle>: keeps the tooltip box narrower than a phone viewport so,
 * with allowEscapeViewBox false, it can always be placed inside the chart. Content must wrap.
 */
export const TOOLTIP_WRAPPER_STYLE = { maxWidth: 'min(280px, calc(100vw - 48px))', zIndex: 10 }

/** "2026-10-20" → "10/20" */
export function formatShortDate(dateText) {
  const [, month, day] = String(dateText || '').split('-')
  if (!month || !day) return String(dateText || '')
  return `${Number(month)}/${Number(day)}`
}
