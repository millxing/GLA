import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import SegmentedControl from './SegmentedControl'
import './GameTimeline.css'

const GARBAGE_WP_ON = 0.95
const GARBAGE_WP_OFF = 0.90

function toIntOrNull(value) {
  if (value === null || value === undefined || value === '') return null
  const num = Number(value)
  if (!Number.isFinite(num)) return null
  return Math.trunc(num)
}

function toFloatOrNull(value) {
  if (value === null || value === undefined || value === '') return null
  const num = Number(value)
  if (!Number.isFinite(num)) return null
  return num
}

function periodLength(period) {
  return period > 4 ? 300 : 720
}

function elapsedSeconds(period, secondsLeft) {
  if (!Number.isFinite(period) || period <= 0 || !Number.isFinite(secondsLeft)) return null
  let elapsed = 0
  for (let p = 1; p < period; p += 1) elapsed += periodLength(p)
  return elapsed + (periodLength(period) - secondsLeft)
}

function formatPeriod(period) {
  if (!Number.isFinite(period) || period <= 0) return ''
  if (period <= 4) return `Q${period}`
  return `OT${period - 4}`
}

function periodLabelShort(period) {
  if (period <= 4) return `Q${period}`
  return `OT${period - 4}`
}

function buildPeriodTicks(maxPeriod) {
  const ticks = []
  let elapsed = 0
  for (let p = 1; p <= maxPeriod; p += 1) {
    ticks.push({ x: elapsed, label: periodLabelShort(p) })
    elapsed += periodLength(p)
  }
  return ticks
}

function clockToSecondsRemaining(clockText, period) {
  if (!clockText) return null
  const match = String(clockText).trim().match(/^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$/)
  if (!match) return null

  const mins = match[1] ? Number(match[1]) : 0
  const secs = match[2] ? Number(match[2]) : 0
  const total = mins * 60 + secs
  const whole = Math.floor(total)
  const maxSeconds = period > 4 ? 300 : 720

  if (!Number.isFinite(whole)) return null
  if (whole < 0) return 0
  if (whole > maxSeconds) return maxSeconds
  return whole
}

function chartPointsFromEvents(events) {
  const raw = []
  events.forEach((event, idx) => {
    const period = toIntOrNull(event?.period)
    const secondsLeft = clockToSecondsRemaining(event?.clock, period)
    if (!period || secondsLeft === null) return

    const state = event?.game_log_state || {}
    const home = toIntOrNull(state?.pts_home)
    const road = toIntOrNull(state?.pts_road)
    if (!Number.isFinite(home) || !Number.isFinite(road)) return

    const xRaw = elapsedSeconds(period, secondsLeft)
    if (!Number.isFinite(xRaw)) return
    const homeWinProbRaw = toFloatOrNull(event?.home_win_prob)
    const homeWinProb = Number.isFinite(homeWinProbRaw)
      ? Math.min(1, Math.max(0, homeWinProbRaw))
      : null

    raw.push({
      xRaw,
      period,
      secondsLeft,
      home,
      road,
      homeWinProb,
      eventIndex: toIntOrNull(event?.event_index) ?? idx + 1,
    })
  })

  raw.sort((a, b) => a.eventIndex - b.eventIndex)

  let eventOrderDrops = 0
  let clockBacktracks = 0
  let timestampRevisits = 0
  let timestampConflicts = 0
  let prevEventHome = 0
  let prevEventRoad = 0
  let hasPrevEvent = false
  let prevXRaw = null
  let prevKey = null
  const seenKeys = new Set()
  const firstScoreByKey = new Map()

  const points = []
  let prevXAdjusted = -Infinity
  const epsilon = 0.001

  for (const p of raw) {
    if (hasPrevEvent && (p.home < prevEventHome || p.road < prevEventRoad)) {
      eventOrderDrops += 1
    }
    prevEventHome = p.home
    prevEventRoad = p.road
    hasPrevEvent = true

    if (prevXRaw !== null && p.xRaw < prevXRaw) {
      clockBacktracks += 1
    }
    prevXRaw = p.xRaw

    const key = `${p.period}|${p.secondsLeft}`
    if (seenKeys.has(key) && prevKey !== key) {
      timestampRevisits += 1
    }
    seenKeys.add(key)
    prevKey = key

    const firstScore = firstScoreByKey.get(key)
    if (!firstScore) {
      firstScoreByKey.set(key, { home: p.home, road: p.road })
    } else if (firstScore.home !== p.home || firstScore.road !== p.road) {
      timestampConflicts += 1
    }

    let x = p.xRaw
    if (x < prevXAdjusted) {
      x = prevXAdjusted + epsilon
    }
    prevXAdjusted = x

    points.push({
      x,
      period: p.period,
      secondsLeft: p.secondsLeft,
      home: p.home,
      road: p.road,
      diff: p.home - p.road,
      homeWinProb: p.homeWinProb,
      eventIndex: p.eventIndex,
    })
  }

  return {
    points,
    diagnostics: {
      eventOrderDrops,
      clockBacktracks,
      timestampRevisits,
      timestampConflicts,
    },
  }
}

function isClutchPoint(point) {
  if (!point) return null
  const absDiff = Math.abs(point.diff)
  const inClutch = (
    Number.isFinite(point.period) &&
    point.period >= 4 &&
    Number.isFinite(point.secondsLeft) &&
    point.secondsLeft < 300 &&
    absDiff <= 5
  )
  return inClutch
}

function buildTimelineZoneSegments(points, xMax) {
  if (!Array.isArray(points) || points.length === 0 || !Number.isFinite(xMax) || xMax <= 0) {
    return []
  }

  const finalPeriod = points.reduce(
    (acc, point) => (
      Number.isFinite(point?.period) && point.period > acc ? point.period : acc
    ),
    0
  )
  const segments = []
  const epsilon = 1e-6
  let garbageActive = false

  for (let i = 0; i < points.length; i += 1) {
    const curr = points[i]
    const next = points[i + 1]
    const start = Math.max(0, curr.x)
    const end = Math.min(xMax, Number.isFinite(next?.x) ? next.x : xMax)
    if (end - start <= epsilon) continue

    const absDiff = Math.abs(curr.diff)
    const canEvaluateWp = Number.isFinite(curr.homeWinProb)

    if (garbageActive) {
      if (
        canEvaluateWp &&
        curr.homeWinProb > (1 - GARBAGE_WP_OFF) &&
        curr.homeWinProb < GARBAGE_WP_OFF
      ) {
        garbageActive = false
      }
    } else {
      const inFinalGameMinute = (
        Number.isFinite(finalPeriod) &&
        finalPeriod > 0 &&
        curr.period === finalPeriod &&
        Number.isFinite(curr.secondsLeft) &&
        curr.secondsLeft < 60
      )
      const shouldEnterGarbage = (
        Number.isFinite(curr.period) &&
        curr.period >= 3 &&
        absDiff > 5 &&
        canEvaluateWp &&
        !inFinalGameMinute &&
        (curr.homeWinProb >= GARBAGE_WP_ON || curr.homeWinProb <= (1 - GARBAGE_WP_ON))
      )
      if (shouldEnterGarbage) {
        garbageActive = true
      }
    }

    const zone = isClutchPoint(curr) ? 'clutch' : (garbageActive ? 'garbage' : null)
    if (!zone) continue

    const prev = segments[segments.length - 1]
    if (prev && prev.zone === zone && Math.abs(prev.end - start) <= epsilon) {
      prev.end = end
    } else {
      segments.push({ zone, start, end })
    }
  }

  return segments
}

function computeTimelineXMax(points, events) {
  const periods = events
    .map((event) => toIntOrNull(event?.period))
    .filter((value) => Number.isFinite(value) && value > 0)
  const maxPeriod = periods.length ? Math.max(...periods) : 1
  let fullDuration = 0
  for (let p = 1; p <= maxPeriod; p += 1) fullDuration += periodLength(p)
  const lastX = points.length ? points[points.length - 1].x : 0
  return Math.max(lastX, fullDuration, 1)
}

function resizeCanvasToDisplaySize(canvas) {
  const dpr = window.devicePixelRatio || 1
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr))
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr))
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
  return { dpr, width, height }
}

function drawLine(ctx, points, color, project, dpr) {
  if (points.length === 0) return
  ctx.strokeStyle = color
  ctx.lineWidth = 2 * dpr
  ctx.lineJoin = 'round'
  ctx.beginPath()
  points.forEach((p, idx) => {
    const [x, y] = project(p)
    if (idx === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.stroke()
}

// Draw one series in the home color above a reference value and the road color below it
function drawSplitLine(ctx, points, project, yRef, area, colors, dpr) {
  const { left, top, width, height } = area
  ctx.save()
  ctx.beginPath()
  ctx.rect(left, top, width, Math.max(0, yRef - top))
  ctx.clip()
  drawLine(ctx, points, colors.home, project, dpr)
  ctx.restore()
  ctx.save()
  ctx.beginPath()
  ctx.rect(left, yRef, width, Math.max(0, top + height - yRef))
  ctx.clip()
  drawLine(ctx, points, colors.road, project, dpr)
  ctx.restore()
}

function drawDot(ctx, x, y, color, dpr, surface) {
  ctx.beginPath()
  ctx.arc(x, y, 4 * dpr, 0, Math.PI * 2)
  ctx.fillStyle = color
  ctx.fill()
  ctx.lineWidth = 2 * dpr
  ctx.strokeStyle = surface
  ctx.stroke()
}

// Canvas can't resolve var(); read the design tokens once from the computed style.
function readChartTokens(element) {
  const style = getComputedStyle(element)
  const token = (name, fallback) => style.getPropertyValue(name).trim() || fallback
  return {
    home: token('--color-home', '#1d4ed8'),
    road: token('--color-road', '#c2410c'),
    grid: token('--chart-grid', '#e8eaee'),
    axis: token('--chart-axis', '#667085'),
    reference: token('--chart-reference', '#14161a'),
    text: token('--color-text-muted', '#667085'),
    crosshair: token('--color-text-secondary', '#4b5260'),
    surface: token('--color-background', '#ffffff'),
    garbage: token('--timeline-garbage', 'rgba(138, 145, 158, 0.18)'),
    clutch: token('--timeline-clutch', 'rgba(234, 179, 8, 0.2)'),
    fontSans: token('--font-sans', 'sans-serif'),
  }
}

function selectedClass(selectedEventIndex, eventIndex) {
  return Number.isFinite(selectedEventIndex) && selectedEventIndex === eventIndex ? 'selected' : ''
}

function formatPercentile(value) {
  const numeric = toFloatOrNull(value)
  if (!Number.isFinite(numeric)) return 'N/A'
  const clipped = Math.max(0, Math.min(100, numeric))
  return `${clipped.toFixed(1)}%`
}

function formatTimeRemaining(secondsLeft) {
  if (!Number.isFinite(secondsLeft)) return ''
  const totalSeconds = Math.max(0, Math.floor(secondsLeft))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

const CHART_MODE_OPTIONS = [
  { value: 'both', label: 'Scores' },
  { value: 'diff', label: 'Differential' },
  { value: 'wp', label: 'Win Probability' },
]

const PLOT_PADDING = { left: 44, right: 12, top: 10, bottom: 26 }

export default function GameTimeline({ timeline }) {
  const [chartMode, setChartMode] = useState('both')
  const [selectedEventIndex, setSelectedEventIndex] = useState(null)
  // Crosshair position as an index into displayedPoints; source decides whether to announce it
  const [hover, setHover] = useState(null)
  const canvasRef = useRef(null)
  const tableShellRef = useRef(null)
  const chartMetaRef = useRef(null)
  const tokensRef = useRef(null)
  const tooltipRef = useRef(null)

  const events = Array.isArray(timeline?.events) ? timeline.events : []
  const homeTeam = timeline?.home_team || 'Home'
  const roadTeam = timeline?.road_team || 'Road'
  const excitementPercentile = timeline?.excitement_percentile
  const comebackPercentile = timeline?.comeback_percentile
  const { points } = useMemo(() => chartPointsFromEvents(events), [events])
  const zoneLegendFlags = useMemo(() => {
    if (!points.length) return { hasGarbage: false, hasClutch: false }
    const xMax = computeTimelineXMax(points, events)
    const segments = buildTimelineZoneSegments(points, xMax)
    return {
      hasGarbage: segments.some((segment) => segment.zone === 'garbage'),
      hasClutch: segments.some((segment) => segment.zone === 'clutch'),
    }
  }, [events, points])
  const wpPoints = useMemo(
    () => points.filter((point) => Number.isFinite(point.homeWinProb)),
    [points]
  )
  const displayedPoints = chartMode === 'wp' ? wpPoints : points
  const descriptionByEventIndex = useMemo(() => {
    const map = new Map()
    events.forEach((event) => {
      const eventIndex = toIntOrNull(event?.event_index)
      if (Number.isFinite(eventIndex) && event?.description) map.set(eventIndex, event.description)
    })
    return map
  }, [events])

  const tableRows = useMemo(() => (
    events.map((event, idx) => {
      const period = toIntOrNull(event?.period)
      const secondsLeft = clockToSecondsRemaining(event?.clock, period)
      const state = event?.game_log_state || {}
      const home = toIntOrNull(state?.pts_home)
      const road = toIntOrNull(state?.pts_road)
      const diff = Number.isFinite(home) && Number.isFinite(road) ? home - road : null
      const possession = event?.possession_team_tricode ||
        (event?.possession_after_side === 'home' ? homeTeam
          : event?.possession_after_side === 'road' ? roadTeam : '')
      const homeWinProb = toFloatOrNull(event?.home_win_prob)
      return {
        key: `${event?.event_index ?? idx}-${idx}`,
        eventIndex: toIntOrNull(event?.event_index),
        description: event?.description || '',
        periodLabel: formatPeriod(period),
        secondsLeft,
        home,
        road,
        diff,
        homeWinProb,
        possession,
      }
    })
  ), [events, homeTeam, roadTeam])

  useEffect(() => {
    setSelectedEventIndex(null)
    setChartMode('both')
    setHover(null)
  }, [timeline?.game_id])

  useEffect(() => {
    setHover(null)
  }, [chartMode])

  const hoverPoint = hover && displayedPoints[hover.index] ? displayedPoints[hover.index] : null

  const drawChart = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    if (!tokensRef.current) tokensRef.current = readChartTokens(canvas)
    const colors = tokensRef.current

    const { width, height, dpr } = resizeCanvasToDisplaySize(canvas)
    ctx.clearRect(0, 0, width, height)

    if (!displayedPoints.length) {
      chartMetaRef.current = null
      return
    }

    const left = PLOT_PADDING.left * dpr
    const right = PLOT_PADDING.right * dpr
    const top = PLOT_PADDING.top * dpr
    const bottom = PLOT_PADDING.bottom * dpr
    const plotW = Math.max(1, width - left - right)
    const plotH = Math.max(1, height - top - bottom)
    const area = { left, top, width: plotW, height: plotH }

    const xMax = computeTimelineXMax(points, events)
    const periods = events
      .map((event) => toIntOrNull(event?.period))
      .filter((value) => Number.isFinite(value) && value > 0)
    const maxPeriod = periods.length ? Math.max(...periods) : 1
    const xTicks = buildPeriodTicks(maxPeriod)

    let yMin
    let yMax
    let yTicks = []
    if (chartMode === 'diff') {
      const diffs = points.map((point) => point.diff)
      const diffMin = Math.min(...diffs)
      const diffMax = Math.max(...diffs)
      yMin = Math.floor(Math.min(0, diffMin) / 5) * 5
      yMax = Math.ceil(Math.max(0, diffMax) / 5) * 5
      if (yMin === yMax) {
        if (yMax <= 0) yMin -= 5
        else yMax += 5
      }
      for (let y = yMin; y <= yMax; y += 5) yTicks.push(y)
    } else if (chartMode === 'wp') {
      yMin = 0
      yMax = 100
      for (let y = 0; y <= 100; y += 10) yTicks.push(y)
    } else {
      const values = points.flatMap((point) => [point.home, point.road])
      yMin = 0
      const maxScore = Math.max(20, ...values)
      yMax = Math.ceil(maxScore / 20) * 20
      for (let y = 0; y <= yMax; y += 20) yTicks.push(y)
    }

    const ySpan = Math.max(1, yMax - yMin)
    const px = (x) => left + (x / xMax) * plotW
    const py = (y) => top + ((yMax - y) / ySpan) * plotH

    const zoneSegments = buildTimelineZoneSegments(points, xMax)
    for (const segment of zoneSegments) {
      const x0 = px(segment.start)
      const x1 = px(segment.end)
      const widthPx = x1 - x0
      if (widthPx <= 0) continue
      ctx.fillStyle = segment.zone === 'garbage' ? colors.garbage : colors.clutch
      ctx.fillRect(x0, top, widthPx, plotH)
    }

    chartMetaRef.current = { left, plotW, xMax, dpr }

    ctx.strokeStyle = colors.grid
    ctx.lineWidth = 1 * dpr
    for (const tick of xTicks) {
      const x = Math.round(px(tick.x)) + 0.5
      ctx.beginPath()
      ctx.moveTo(x, top)
      ctx.lineTo(x, top + plotH)
      ctx.stroke()
    }
    for (const tick of yTicks) {
      const y = Math.round(py(tick)) + 0.5
      ctx.beginPath()
      ctx.moveTo(left, y)
      ctx.lineTo(left + plotW, y)
      ctx.stroke()
    }

    // IBM Plex Sans figures are tabular by default, so tick numbers stay aligned
    ctx.fillStyle = colors.text
    ctx.font = `500 ${12 * dpr}px ${colors.fontSans}`
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (const tick of yTicks) {
      const label = chartMode === 'wp' ? `${tick}%` : String(Math.round(tick))
      ctx.fillText(label, left - (6 * dpr), py(tick))
    }

    ctx.strokeStyle = colors.axis
    ctx.lineWidth = 1 * dpr
    ctx.beginPath()
    ctx.moveTo(left, top + plotH)
    ctx.lineTo(left + plotW, top + plotH)
    ctx.stroke()

    ctx.textAlign = 'left'
    ctx.textBaseline = 'alphabetic'
    for (const tick of xTicks) {
      ctx.fillText(tick.label, px(tick.x) + 4 * dpr, top + plotH + (17 * dpr))
    }

    let project
    if (chartMode === 'diff') {
      const y0 = py(0)
      ctx.strokeStyle = colors.reference
      ctx.lineWidth = 1 * dpr
      ctx.beginPath()
      ctx.moveTo(left, y0)
      ctx.lineTo(left + plotW, y0)
      ctx.stroke()
      project = (point) => [px(point.x), py(point.diff)]
      drawSplitLine(ctx, displayedPoints, project, y0, area, colors, dpr)
    } else if (chartMode === 'wp') {
      const y50 = py(50)
      ctx.strokeStyle = colors.reference
      ctx.lineWidth = 1 * dpr
      ctx.beginPath()
      ctx.moveTo(left, y50)
      ctx.lineTo(left + plotW, y50)
      ctx.stroke()
      project = (point) => [px(point.x), py(point.homeWinProb * 100)]
      drawSplitLine(ctx, displayedPoints, project, y50, area, colors, dpr)
    } else {
      drawLine(ctx, points, colors.road, (point) => [px(point.x), py(point.road)], dpr)
      drawLine(ctx, points, colors.home, (point) => [px(point.x), py(point.home)], dpr)
    }

    if (Number.isFinite(selectedEventIndex)) {
      let markerX = null
      let bestDistance = Number.POSITIVE_INFINITY
      for (const point of displayedPoints) {
        const distance = Math.abs(point.eventIndex - selectedEventIndex)
        if (distance < bestDistance) {
          bestDistance = distance
          markerX = point.x
          if (distance === 0) break
        }
      }
      if (markerX !== null) {
        const x = px(markerX)
        ctx.strokeStyle = colors.reference
        ctx.lineWidth = 2 * dpr
        ctx.beginPath()
        ctx.moveTo(x, top)
        ctx.lineTo(x, top + plotH)
        ctx.stroke()
      }
    }

    if (hoverPoint) {
      const x = px(hoverPoint.x)
      ctx.save()
      ctx.strokeStyle = colors.crosshair
      ctx.lineWidth = 1 * dpr
      ctx.setLineDash([4 * dpr, 3 * dpr])
      ctx.beginPath()
      ctx.moveTo(x, top)
      ctx.lineTo(x, top + plotH)
      ctx.stroke()
      ctx.restore()
      if (chartMode === 'both') {
        drawDot(ctx, x, py(hoverPoint.road), colors.road, dpr, colors.surface)
        drawDot(ctx, x, py(hoverPoint.home), colors.home, dpr, colors.surface)
      } else {
        const [, y] = project(hoverPoint)
        const value = chartMode === 'diff' ? hoverPoint.diff : hoverPoint.homeWinProb * 100
        const reference = chartMode === 'diff' ? 0 : 50
        drawDot(ctx, x, y, value >= reference ? colors.home : colors.road, dpr, colors.surface)
      }
    }
  }, [chartMode, displayedPoints, events, hoverPoint, points, selectedEventIndex])

  useEffect(() => {
    drawChart()
  }, [drawChart])

  useEffect(() => {
    const onResize = () => {
      drawChart()
    }

    onResize()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [drawChart, timeline?.game_id])

  // Canvas text doesn't trigger web-font loading; request Plex Sans 500 explicitly, then redraw.
  useEffect(() => {
    let cancelled = false
    const fonts = document.fonts
    if (!fonts) return undefined
    Promise.all([fonts.load('500 12px "IBM Plex Sans"'), fonts.ready])
      .catch(() => {})
      .then(() => {
        if (!cancelled) drawChart()
      })
    return () => { cancelled = true }
  }, [drawChart])

  useEffect(() => {
    const tableShell = tableShellRef.current
    if (!tableShell || !Number.isFinite(selectedEventIndex)) return
    const row = tableShell.querySelector(`tr[data-event-index="${selectedEventIndex}"]`)
    if (!row) return

    const rowTop = row.offsetTop
    const rowBottom = rowTop + row.offsetHeight
    const viewTop = tableShell.scrollTop
    const viewBottom = viewTop + tableShell.clientHeight
    const padding = 10

    if (rowTop < viewTop + padding || rowBottom > viewBottom - padding) {
      const targetTop = Math.max(0, Math.floor(rowTop - tableShell.clientHeight * 0.35))
      tableShell.scrollTop = targetTop
    }
  }, [selectedEventIndex])

  const nearestPointIndex = (clientX) => {
    const canvas = canvasRef.current
    const chartMeta = chartMetaRef.current
    if (!canvas || !chartMeta || !displayedPoints.length) return null

    const rect = canvas.getBoundingClientRect()
    if (!rect.width || !rect.height) return null
    const xCanvas = (clientX - rect.left) * chartMeta.dpr
    const xClamped = Math.min(
      chartMeta.left + chartMeta.plotW,
      Math.max(chartMeta.left, xCanvas)
    )
    const xValue = ((xClamped - chartMeta.left) / chartMeta.plotW) * chartMeta.xMax

    let bestIndex = null
    let bestDistance = Number.POSITIVE_INFINITY
    displayedPoints.forEach((point, index) => {
      const distance = Math.abs(point.x - xValue)
      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = index
      }
    })
    return bestIndex
  }

  const onChartClick = (event) => {
    const index = nearestPointIndex(event.clientX)
    const point = index === null ? null : displayedPoints[index]
    if (!point || !Number.isFinite(point.eventIndex)) return
    setSelectedEventIndex(point.eventIndex)
  }

  const onChartPointer = (event) => {
    const index = nearestPointIndex(event.clientX)
    if (index === null) return
    setHover((prev) => (prev && prev.index === index && prev.source === 'pointer' ? prev : { index, source: 'pointer' }))
  }

  const onChartPointerLeave = (event) => {
    // On touch the tooltip stays after the finger lifts so it can be read
    if (event.pointerType === 'mouse') setHover(null)
  }

  const onChartKeyDown = (event) => {
    if (!displayedPoints.length) return
    const last = displayedPoints.length - 1
    const current = hover ? hover.index : null
    const step = event.shiftKey ? 10 : 1
    let next = null
    if (event.key === 'ArrowRight') next = current === null ? 0 : Math.min(last, current + step)
    else if (event.key === 'ArrowLeft') next = current === null ? last : Math.max(0, current - step)
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = last
    else if ((event.key === 'Enter' || event.key === ' ') && current !== null) {
      event.preventDefault()
      const point = displayedPoints[current]
      if (point && Number.isFinite(point.eventIndex)) setSelectedEventIndex(point.eventIndex)
      return
    } else if (event.key === 'Escape') {
      setHover(null)
      return
    } else {
      return
    }
    event.preventDefault()
    setHover({ index: next, source: 'keyboard' })
  }

  const onChartBlur = () => {
    setHover((prev) => (prev && prev.source === 'keyboard' ? null : prev))
  }

  const renderTooltipContent = (point) => {
    const diff = point.diff
    const leader = diff > 0 ? homeTeam : diff < 0 ? roadTeam : null
    const description = descriptionByEventIndex.get(point.eventIndex)
    return (
      <>
        <div className="gt-tooltip-time">
          {formatPeriod(point.period)} · {formatTimeRemaining(point.secondsLeft)}
        </div>
        <div className="gt-tooltip-score">
          <span className="gt-text-road">{roadTeam} {point.road}</span>
          <span className="gt-tooltip-sep"> – </span>
          <span className="gt-text-home">{homeTeam} {point.home}</span>
        </div>
        <div className="gt-tooltip-meta">
          {leader ? `${leader} +${Math.abs(diff)}` : 'Tied'}
          {Number.isFinite(point.homeWinProb) && (
            <> · {homeTeam} win prob {(point.homeWinProb * 100).toFixed(1)}%</>
          )}
        </div>
        {description && <div className="gt-tooltip-play">{description}</div>}
      </>
    )
  }

  // Tooltip anchor in CSS px, computed from the same geometry the canvas uses
  const timelineXMax = useMemo(() => computeTimelineXMax(points, events), [events, points])
  const canvasWidth = canvasRef.current?.clientWidth || 0
  const hoverX = hoverPoint && canvasWidth
    ? PLOT_PADDING.left + (hoverPoint.x / timelineXMax) * (canvasWidth - PLOT_PADDING.left - PLOT_PADDING.right)
    : 0

  // Place the tooltip beside the crosshair (left of it past mid-width), clamped inside the plot
  useLayoutEffect(() => {
    const tooltip = tooltipRef.current
    if (!tooltip || !hoverPoint) return
    const gap = 12
    const width = tooltip.offsetWidth
    const flip = hoverX > canvasWidth / 2
    const preferred = flip ? hoverX - gap - width : hoverX + gap
    const left = Math.max(0, Math.min(canvasWidth - width, preferred))
    tooltip.style.left = `${Math.round(left)}px`
  }, [canvasWidth, hoverPoint, hoverX])

  // The table body only changes with the rows or the selected play, not on every crosshair move
  const tableBody = useMemo(() => (
    <tbody>
      {tableRows.map((row, idx) => (
        <tr
          key={row.key}
          data-event-index={Number.isFinite(row.eventIndex) ? row.eventIndex : undefined}
          className={selectedClass(selectedEventIndex, row.eventIndex)}
          onClick={() => {
            if (Number.isFinite(row.eventIndex)) setSelectedEventIndex(row.eventIndex)
          }}
        >
          <td className="gt-row-index">{Number.isFinite(row.eventIndex) ? row.eventIndex : idx + 1}</td>
          <td className="gt-play">
            {/* Narrow screens: clock and score ride above the play text (the columns are hidden) */}
            <span className="gt-play-meta">
              {row.periodLabel} {formatTimeRemaining(row.secondsLeft)}
              {Number.isFinite(row.road) && Number.isFinite(row.home) && (
                <>
                  {' · '}
                  <span className="gt-text-road">{roadTeam} {row.road}</span>
                  {'–'}
                  <span className="gt-text-home">{row.home} {homeTeam}</span>
                </>
              )}
            </span>
            {row.description}
          </td>
          <td>{row.periodLabel}</td>
          <td>{formatTimeRemaining(row.secondsLeft)}</td>
          <td className="num">{Number.isFinite(row.road) ? row.road : ''}</td>
          <td className="num">{Number.isFinite(row.home) ? row.home : ''}</td>
          <td className="num">{Number.isFinite(row.diff) ? (row.diff > 0 ? `+${row.diff}` : row.diff) : ''}</td>
          <td className="num">{Number.isFinite(row.homeWinProb) ? `${(row.homeWinProb * 100).toFixed(1)}%` : ''}</td>
          <td>{row.possession}</td>
        </tr>
      ))}
    </tbody>
  ), [tableRows, selectedEventIndex, homeTeam, roadTeam])

  return (
    <div className="game-timeline-view">
      <section className="card gt-chart-card" aria-label="Game timeline">
        <div className="gt-toolbar">
          <SegmentedControl
            size="sm"
            label="Chart series"
            options={CHART_MODE_OPTIONS}
            value={chartMode}
            onChange={setChartMode}
          />
          <div className="gt-badges" aria-label="Season timeline percentiles">
            <div className="gt-badge">
              <span className="gt-badge-label">Excitement percentile</span>
              <span className="gt-badge-value">{formatPercentile(excitementPercentile)}</span>
            </div>
            <div className="gt-badge">
              <span className="gt-badge-label">Comeback percentile</span>
              <span className="gt-badge-value">{formatPercentile(comebackPercentile)}</span>
            </div>
          </div>
        </div>

        <div className="gt-legend">
          {displayedPoints.length === 0 && 'No chartable events.'}
          {displayedPoints.length > 0 && (
            <>
              {chartMode === 'both' && (
                <>
                  <span className="gt-legend-item"><span className="gt-swatch gt-swatch--road" />{roadTeam}</span>
                  <span className="gt-legend-item"><span className="gt-swatch gt-swatch--home" />{homeTeam}</span>
                </>
              )}
              {chartMode === 'diff' && (
                <>
                  <span className="gt-legend-item"><span className="gt-swatch gt-swatch--home" />{homeTeam} leads (above 0)</span>
                  <span className="gt-legend-item"><span className="gt-swatch gt-swatch--road" />{roadTeam} leads</span>
                </>
              )}
              {chartMode === 'wp' && (
                <>
                  <span className="gt-legend-item"><span className="gt-swatch gt-swatch--home" />{homeTeam} win probability, favored above 50%</span>
                  <span className="gt-legend-item"><span className="gt-swatch gt-swatch--road" />{roadTeam} favored</span>
                </>
              )}
              {zoneLegendFlags.hasGarbage && (
                <span className="gt-legend-item"><span className="gt-swatch gt-swatch--zone gt-swatch--garbage" />Garbage time</span>
              )}
              {zoneLegendFlags.hasClutch && (
                <span className="gt-legend-item"><span className="gt-swatch gt-swatch--zone gt-swatch--clutch" />Clutch time</span>
              )}
            </>
          )}
        </div>

        <div className="gt-plot">
          <canvas
            ref={canvasRef}
            tabIndex={0}
            role="img"
            aria-label={`Game timeline chart, ${roadTeam} at ${homeTeam}. Use left and right arrow keys to step through plays, Enter to select the play in the table.`}
            onClick={onChartClick}
            onPointerMove={onChartPointer}
            onPointerDown={onChartPointer}
            onPointerLeave={onChartPointerLeave}
            onKeyDown={onChartKeyDown}
            onBlur={onChartBlur}
          />
          {hoverPoint && (
            <div className="gt-tooltip" ref={tooltipRef} aria-hidden="true">
              {renderTooltipContent(hoverPoint)}
            </div>
          )}
          <div className="sr-only" aria-live="polite">
            {hoverPoint && hover.source === 'keyboard' && (
              <>
                {formatPeriod(hoverPoint.period)} {formatTimeRemaining(hoverPoint.secondsLeft)}: {roadTeam} {hoverPoint.road}, {homeTeam} {hoverPoint.home}.
                {Number.isFinite(hoverPoint.homeWinProb) && ` ${homeTeam} win probability ${(hoverPoint.homeWinProb * 100).toFixed(1)}%.`}
                {descriptionByEventIndex.get(hoverPoint.eventIndex) ? ` ${descriptionByEventIndex.get(hoverPoint.eventIndex)}` : ''}
              </>
            )}
          </div>
        </div>
      </section>

      <section className="card gt-table-card" aria-label="Play-by-play">
        <div className="gt-table-shell" ref={tableShellRef} tabIndex={0} role="region" aria-label="Play-by-play table">
          <table className="gt-table">
            <thead>
              <tr>
                <th scope="col" className="gt-row-index">#</th>
                <th scope="col">Play</th>
                <th scope="col">Quarter</th>
                <th scope="col">Time Remaining</th>
                <th scope="col" className="num">{roadTeam}</th>
                <th scope="col" className="num">{homeTeam}</th>
                <th scope="col" className="num" title={`${homeTeam} score minus ${roadTeam} score`}>
                  {homeTeam} margin
                </th>
                <th scope="col" className="num">{homeTeam} win prob</th>
                <th scope="col">Possession</th>
              </tr>
            </thead>
            {tableBody}
          </table>
        </div>
      </section>
    </div>
  )
}
