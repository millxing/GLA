import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import './Home.css'

const FEATURES = [
  {
    id: 'game-analysis',
    kicker: 'Game Analysis',
    title: 'See what actually won the game',
    highlights: [
      'Factor contributions to rating differential',
      'Show statistics with or without garbage-time included',
      'Game timelines identifying clutch-time and garbage-time',
      'Excitement and comeback rankings for every game',
      'AI-generated post-game analysis',
    ],
    to: '/four-factor',
    cta: 'Open Game Analysis',
    previews: ['ga-00', 'ga-01', 'ga-02', 'ga-03', 'ga-04', 'ga-05'],
    mobilePreview: { id: 'ga-m', alt: 'Factor contributions chart from Game Analysis' },
    duration: 35,
  },
  {
    id: 'league-summary',
    kicker: 'League Summary',
    title: 'Rank every team in any stat, over any period of time',
    highlights: [
      'Sortable rankings across all major stat categories over any time period',
      'Show stats with or without garbage-time included',
      'Show stats during clutch-time',
      'Efficiency Landscape scatter plot for any selected period',
      'Top contributing factors to team performance over the period',
    ],
    to: '/league-summary',
    cta: 'Open League Summary',
    previews: ['ls-01', 'ls-02', 'ls-03'],
    mobilePreview: { id: 'ls-m', alt: 'Efficiency Landscape scatter plot from League Summary' },
    duration: 25,
  },
  {
    id: 'statistical-trends',
    kicker: 'Statistical Trends',
    title: 'Track how performance changes over time',
    highlights: [
      'Time-series views for key stats',
      'Context for hot and cold stretches',
      'Quick pivots between teams and metrics',
      'Click any game to jump straight to Game Analysis',
    ],
    to: '/trends',
    cta: 'Open Statistical Trends',
    previews: ['st-01', 'st-02'],
    mobilePreview: { id: 'st-m', alt: 'Net rating trend chart from Statistical Trends' },
    duration: 20,
  },
  {
    id: 'contribution-analysis',
    kicker: 'Contribution Analysis',
    title: 'Decompose team performance using an eight-factor model',
    highlights: [
      'Eight-factor team profile at a glance',
      'Positive and negative contributions separated cleanly',
      'Top contributing factors with game-by-game mini charts',
    ],
    to: '/contribution-analysis',
    cta: 'Open Contribution Analysis',
    previews: ['ca-01', 'ca-02'],
    mobilePreview: { id: 'ca-m', alt: 'Net rating decomposition chart from Contribution Analysis' },
    duration: 20,
  },
]

const HEADER_H = 56
const SNAP_ZONE = 0.3 // snap only when a section edge is within 30% of the viewport, in the scroll direction

function Home() {
  const [visibleSections, setVisibleSections] = useState(() => new Set())

  // Wordmark roll plays once on load; hovering (or tapping) it replays once it has finished.
  const [rollKey, setRollKey] = useState(0)
  const rollingRef = useRef(true)
  const replayRoll = () => {
    if (rollingRef.current) return
    rollingRef.current = true
    setRollKey((key) => key + 1)
  }

  const scrollToFirstFeature = (event) => {
    const target = document.getElementById(FEATURES[0].id)
    if (!target) return
    event.preventDefault()
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    window.scrollTo({
      top: target.getBoundingClientRect().top + window.scrollY - HEADER_H,
      behavior: reduceMotion ? 'auto' : 'smooth',
    })
  }

  // Reveal each section once it scrolls into view
  useEffect(() => {
    const panels = document.querySelectorAll('.home [data-section-id]')

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return
          const sectionId = entry.target.getAttribute('data-section-id')
          setVisibleSections((previous) => {
            if (previous.has(sectionId)) return previous
            const next = new Set(previous)
            next.add(sectionId)
            return next
          })
        })
      },
      { threshold: 0.15 }
    )

    panels.forEach((panel) => observer.observe(panel))
    return () => observer.disconnect()
  }, [])

  // Soft snap on wide screens only: after scrolling stops, settle on the next section edge
  // in the direction the user was already scrolling. Never pulls back against the user.
  useEffect(() => {
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
    const wide = window.matchMedia('(min-width: 1081px)')

    let timeout
    let isSnapping = false
    let lastY = window.scrollY
    let direction = 0

    const handleScroll = () => {
      const y = window.scrollY
      if (y !== lastY) direction = y > lastY ? 1 : -1
      lastY = y
      if (isSnapping || reduceMotion.matches || !wide.matches) return

      clearTimeout(timeout)
      timeout = setTimeout(() => {
        const scrollTop = window.scrollY
        const threshold = window.innerHeight * SNAP_ZONE
        const edges = [0].concat(
          Array.from(document.querySelectorAll('.home-feature')).map(
            (panel) => panel.getBoundingClientRect().top + scrollTop - HEADER_H
          )
        )

        const target =
          direction > 0
            ? edges.find((edge) => edge > scrollTop + 3 && edge - scrollTop < threshold)
            : edges.reverse().find((edge) => edge < scrollTop - 3 && scrollTop - edge < threshold)

        if (target === undefined) return
        isSnapping = true
        window.scrollTo({ top: target, behavior: 'smooth' })
        setTimeout(() => {
          isSnapping = false
          lastY = window.scrollY
        }, 700)
      }, 140)
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', handleScroll)
      clearTimeout(timeout)
    }
  }, [])

  return (
    <div className="home">
      <section className="home-intro">
        <div className="home-intro-inner">
          <p
            key={rollKey}
            className="home-wordmark"
            onPointerEnter={replayRoll}
            onAnimationEnd={(event) => {
              if (event.animationName === 'home-ball-roll') rollingRef.current = false
            }}
          >
            <span className="home-wordmark-text">Extra Pass Analytics</span>
            <svg className="home-wordmark-ball" viewBox="0 0 48 48" aria-hidden="true" focusable="false">
              <circle cx="24" cy="24" r="22" fill="#ff6b2b" />
              <g stroke="#b8400e" strokeWidth="1.5" opacity="0.55" fill="none">
                <path d="M24 2v44" />
                <path d="M2 24h44" />
                <path d="M8 6c4 9 4 27 0 36" />
                <path d="M40 6c-4 9-4 27 0 36" />
              </g>
            </svg>
          </p>
          <h1 className="home-slogan">
            <span className="home-slogan-line">
              <span className="home-slogan-word">Understand</span>{' '}
              <span className="home-slogan-word">Why</span>
            </span>{' '}
            <span className="home-slogan-line">
              <span className="home-slogan-word">Teams</span>{' '}
              <span className="home-slogan-word">Win</span>
            </span>
          </h1>
          <p className="home-intro-text">
            Factor decomposition for games and multi-game periods, league rankings, and trend analysis
            for all NBA seasons since 2000.
          </p>
        </div>
        <a href={`#${FEATURES[0].id}`} className="home-scroll-cue" onClick={scrollToFirstFeature}>
          <span>Explore the tools</span>
          <svg width="18" height="10" viewBox="0 0 18 10" fill="none" aria-hidden="true" focusable="false">
            <path d="M1 1l8 7.5L17 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </a>
      </section>

      {FEATURES.map((feature, index) => (
        <section
          key={feature.id}
          id={feature.id}
          data-section-id={feature.id}
          aria-labelledby={`${feature.id}-title`}
          className={[
            'home-feature',
            index % 2 === 1 ? 'home-feature--reverse' : '',
            visibleSections.has(feature.id) ? 'is-visible' : '',
          ]
            .filter(Boolean)
            .join(' ')}
        >
          <div className="home-feature-inner">
            <div className="home-feature-copy">
              <p className="home-feature-kicker">{feature.kicker}</p>
              <h2 className="home-feature-title" id={`${feature.id}-title`}>
                {feature.title}
              </h2>
              <ul className="home-feature-list">
                {feature.highlights.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
              <Link to={feature.to} className="btn home-cta">
                {feature.cta}
                <span aria-hidden="true">→</span>
              </Link>
            </div>

            <img
              className="home-mobile-shot"
              src={`/previews/${feature.mobilePreview.id}.png`}
              alt={feature.mobilePreview.alt}
              loading="lazy"
            />

            <div className="home-filmstrip" aria-hidden="true">
              <div className="home-filmstrip-track" style={{ '--filmstrip-duration': `${feature.duration}s` }}>
                {[0, 1].map((copy) => (
                  <div className="home-filmstrip-set" key={copy}>
                    {feature.previews.map((id) => (
                      <img
                        key={`${copy}-${id}`}
                        src={`/previews/${id}.png`}
                        alt=""
                        className="home-filmstrip-shot"
                        loading="lazy"
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      ))}
    </div>
  )
}

export default Home
