import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './GameTimeline.css'

function toIntOrNull(value) {
  if (value === null || value === undefined || value === '') return null
  const num = Number(value)
  if (!Number.isFinite(num)) return null
  return Math.trunc(num)
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

    raw.push({
      xRaw,
      period,
      secondsLeft,
      home,
      road,
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
      home: p.home,
      road: p.road,
      diff: p.home - p.road,
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
  ctx.beginPath()
  points.forEach((p, idx) => {
    const [x, y] = project(p)
    if (idx === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.stroke()
}

function selectedClass(selectedEventIndex, eventIndex) {
  return Number.isFinite(selectedEventIndex) && selectedEventIndex === eventIndex ? 'selected' : ''
}

export default function GameTimeline({ timeline }) {
  const [chartMode, setChartMode] = useState('both')
  const [selectedEventIndex, setSelectedEventIndex] = useState(null)
  const canvasRef = useRef(null)
  const tableShellRef = useRef(null)
  const chartMetaRef = useRef(null)

  const events = Array.isArray(timeline?.events) ? timeline.events : []
  const homeTeam = timeline?.home_team || 'Home'
  const roadTeam = timeline?.road_team || 'Road'
  const { points } = useMemo(() => chartPointsFromEvents(events), [events])

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
      return {
        key: `${event?.event_index ?? idx}-${idx}`,
        eventIndex: toIntOrNull(event?.event_index),
        description: event?.description || '',
        periodLabel: formatPeriod(period),
        secondsLeft,
        home,
        road,
        diff,
        possession,
      }
    })
  ), [events, homeTeam, roadTeam])

  useEffect(() => {
    setSelectedEventIndex(null)
    setChartMode('both')
  }, [timeline?.game_id])

  const drawChart = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { width, height, dpr } = resizeCanvasToDisplaySize(canvas)
    ctx.clearRect(0, 0, width, height)

    if (!points.length) {
      chartMetaRef.current = null
      return
    }

    const left = 56 * dpr
    const right = 18 * dpr
    const top = 18 * dpr
    const bottom = 52 * dpr
    const plotW = Math.max(1, width - left - right)
    const plotH = Math.max(1, height - top - bottom)

    const periods = events
      .map((event) => toIntOrNull(event?.period))
      .filter((value) => Number.isFinite(value) && value > 0)
    const maxPeriod = periods.length ? Math.max(...periods) : 1
    let fullDuration = 0
    for (let p = 1; p <= maxPeriod; p += 1) fullDuration += periodLength(p)
    const xMax = Math.max(points[points.length - 1].x, fullDuration, 1)
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

    chartMetaRef.current = {
      left,
      plotW,
      xMax,
      dpr,
    }

    ctx.strokeStyle = '#ece7dd'
    ctx.lineWidth = 1 * dpr
    for (const tick of xTicks) {
      const x = px(tick.x)
      ctx.beginPath()
      ctx.moveTo(x, top)
      ctx.lineTo(x, top + plotH)
      ctx.stroke()
    }

    ctx.strokeStyle = '#e5e0d8'
    ctx.lineWidth = 1 * dpr
    for (const tick of yTicks) {
      const y = py(tick)
      ctx.beginPath()
      ctx.moveTo(left, y)
      ctx.lineTo(left + plotW, y)
      ctx.stroke()
    }

    ctx.fillStyle = '#666'
    ctx.font = `625 ${12 * dpr}px "JetBrains Mono", monospace`
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (const tick of yTicks) {
      const y = py(tick)
      // Right-align y-axis labels with a small gap from the axis.
      ctx.fillText(String(Math.round(tick)), left - (6 * dpr), y)
    }

    ctx.strokeStyle = '#bdb8af'
    ctx.lineWidth = 1 * dpr
    ctx.beginPath()
    ctx.moveTo(left, top)
    ctx.lineTo(left, top + plotH)
    ctx.lineTo(left + plotW, top + plotH)
    ctx.stroke()

    ctx.strokeStyle = '#cfc9bf'
    ctx.fillStyle = '#666'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'alphabetic'
    for (const tick of xTicks) {
      const x = px(tick.x)
      ctx.beginPath()
      ctx.moveTo(x, top + plotH)
      ctx.lineTo(x, top + plotH + 6 * dpr)
      ctx.stroke()
      ctx.fillText(tick.label, x - (12 * dpr), top + plotH + (18 * dpr))
    }

    if (chartMode === 'diff') {
      if (0 >= yMin && 0 <= yMax) {
        const y0 = py(0)
        ctx.strokeStyle = '#111'
        ctx.lineWidth = 3 * dpr
        ctx.beginPath()
        ctx.moveTo(left, y0)
        ctx.lineTo(left + plotW, y0)
        ctx.stroke()
      }
      drawLine(ctx, points, '#0f766e', (point) => [px(point.x), py(point.diff)], dpr)
    } else {
      drawLine(ctx, points, '#1d4ed8', (point) => [px(point.x), py(point.home)], dpr)
      drawLine(ctx, points, '#dc2626', (point) => [px(point.x), py(point.road)], dpr)
    }

    if (Number.isFinite(selectedEventIndex)) {
      let markerX = null
      let bestDistance = Number.POSITIVE_INFINITY
      for (const point of points) {
        const distance = Math.abs(point.eventIndex - selectedEventIndex)
        if (distance < bestDistance) {
          bestDistance = distance
          markerX = point.x
          if (distance === 0) break
        }
      }
      if (markerX !== null) {
        const x = px(markerX)
        ctx.strokeStyle = '#111'
        ctx.lineWidth = 2 * dpr
        ctx.beginPath()
        ctx.moveTo(x, top)
        ctx.lineTo(x, top + plotH)
        ctx.stroke()
      }
    }
  }, [chartMode, events, points, selectedEventIndex])

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

  const onChartClick = (event) => {
    const canvas = canvasRef.current
    const chartMeta = chartMetaRef.current
    if (!canvas || !chartMeta || !points.length) return

    const rect = canvas.getBoundingClientRect()
    if (!rect.width || !rect.height) return
    const xCanvas = (event.clientX - rect.left) * chartMeta.dpr
    const xClamped = Math.min(
      chartMeta.left + chartMeta.plotW,
      Math.max(chartMeta.left, xCanvas)
    )
    const xValue = ((xClamped - chartMeta.left) / chartMeta.plotW) * chartMeta.xMax

    let bestPoint = null
    let bestDistance = Number.POSITIVE_INFINITY
    for (const point of points) {
      const distance = Math.abs(point.x - xValue)
      if (distance < bestDistance) {
        bestDistance = distance
        bestPoint = point
      }
    }
    if (!bestPoint || !Number.isFinite(bestPoint.eventIndex)) return
    setSelectedEventIndex(bestPoint.eventIndex)
  }

  return (
    <div className="game-timeline-view">
      <div className="game-timeline-chart card">
        <div className="timeline-header">
          <h2 className="card-title">Game Timeline</h2>
          <div className="timeline-mode-toggle-wrap">
            <div className="timeline-mode-toggle" role="group" aria-label="Chart series toggle">
              <button
                type="button"
                className={chartMode === 'both' ? 'active' : ''}
                onClick={() => setChartMode('both')}
              >
                Scores
              </button>
              <button
                type="button"
                className={chartMode === 'diff' ? 'active' : ''}
                onClick={() => setChartMode('diff')}
              >
                Differential
              </button>
            </div>
          </div>
        </div>

        <div className="timeline-chart-shell">
          <div className="timeline-chart-legend">
            {points.length === 0 && 'No chartable events.'}
            {points.length > 0 && chartMode === 'diff' && (
              <>
                <span className="timeline-legend-home">{homeTeam}</span>
                <span className="timeline-legend-separator"> - </span>
                <span className="timeline-legend-road">{roadTeam}</span>
              </>
            )}
            {points.length > 0 && chartMode === 'both' && (
              <>
                <span className="legend-item timeline-legend-home">
                  <span className="legend-swatch"></span>
                  {homeTeam}
                </span>
                <span className="legend-item timeline-legend-road">
                  <span className="legend-swatch"></span>
                  {roadTeam}
                </span>
              </>
            )}
          </div>
          <canvas ref={canvasRef} onClick={onChartClick} />
        </div>
      </div>

      <div className="game-timeline-table card">
        <div className="timeline-table-shell" ref={tableShellRef}>
          <table className="timeline-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Play</th>
                <th>Quarter</th>
                <th>Seconds Left</th>
                <th className="num">{homeTeam}</th>
                <th className="num">{roadTeam}</th>
                <th className="num">Differential</th>
                <th>Possession</th>
              </tr>
            </thead>
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
                  <td className="row-index">{Number.isFinite(row.eventIndex) ? row.eventIndex : idx + 1}</td>
                  <td className="play-desc">{row.description}</td>
                  <td>{row.periodLabel}</td>
                  <td>{Number.isFinite(row.secondsLeft) ? row.secondsLeft : ''}</td>
                  <td className="num">{Number.isFinite(row.home) ? row.home : ''}</td>
                  <td className="num">{Number.isFinite(row.road) ? row.road : ''}</td>
                  <td className="num">{Number.isFinite(row.diff) ? row.diff : ''}</td>
                  <td>{row.possession}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
