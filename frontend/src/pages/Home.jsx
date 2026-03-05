import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import './Home.css'

function Home() {
  const [visibleSections, setVisibleSections] = useState(new Set(['hero']))

  // Intersection observer — reveal animations
  useEffect(() => {
    const panels = document.querySelectorAll('[data-section-id]')

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
      { threshold: [0.2, 0.4] }
    )

    panels.forEach((panel) => observer.observe(panel))
    return () => observer.disconnect()
  }, [])

  // Soft scroll snap — nudge to nearest section after scrolling stops
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

    let timeout
    let isSnapping = false
    const HEADER_H = 58
    const SNAP_ZONE = 0.35 // snap if within 35% of viewport height of a boundary

    const handleScroll = () => {
      if (isSnapping) return
      clearTimeout(timeout)
      timeout = setTimeout(() => {
        const panels = document.querySelectorAll('.home-panel')
        const scrollTop = window.scrollY
        const vh = window.innerHeight
        const threshold = vh * SNAP_ZONE

        for (const panel of panels) {
          const snapTo = panel.offsetTop - HEADER_H
          const distance = Math.abs(scrollTop - snapTo)

          if (distance > 3 && distance < threshold) {
            isSnapping = true
            window.scrollTo({ top: snapTo, behavior: 'smooth' })
            setTimeout(() => { isSnapping = false }, 800)
            break
          }
        }
      }, 120)
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', handleScroll)
      clearTimeout(timeout)
    }
  }, [])

  return (
    <div className="home">
      {/* ── HERO ── */}
      <section
        id="hero"
        data-section-id="hero"
        className={visibleSections.has('hero') ? 'home-panel hero-panel is-visible' : 'home-panel hero-panel'}
      >
        <div className="hero-grid" aria-hidden="true" />
        <div className="hero-content">
          <div className="rolling-banner" aria-label="Extra Pass Analytics">
            <span className="rolling-text">Extra Pass Analytics</span>
            <div className="rolling-ball-track">
              <svg className="rolling-ball" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="24" cy="24" r="22" fill="#ff6b2b" />
                <path d="M24 2v44" stroke="#b8400e" strokeWidth="1.5" opacity="0.4" />
                <path d="M2 24h44" stroke="#b8400e" strokeWidth="1.5" opacity="0.4" />
                <path d="M8 6c4 9 4 27 0 36" stroke="#b8400e" strokeWidth="1.5" opacity="0.4" fill="none" />
                <path d="M40 6c-4 9-4 27 0 36" stroke="#b8400e" strokeWidth="1.5" opacity="0.4" fill="none" />
              </svg>
            </div>
          </div>
          <h1 className="hero-title">
            <span className="hero-line">
              <span className="hero-word hero-word--1">Understand</span>{' '}
              <span className="hero-word hero-word--2">Why</span>
            </span>
            <span className="hero-line">
              <span className="hero-word hero-word--3">Teams</span>{' '}
              <span className="hero-word hero-word--4">Win</span>
            </span>
          </h1>
          <p className="hero-subtitle">
            Factor decomposition for games and multi-game periods, league rankings, and trend analysis for all NBA seasons since 2000.
          </p>
        </div>
        <a href="#game-analysis" className="scroll-cue" aria-label="Scroll to explore">
          <span className="scroll-text">Scroll</span>
          <svg className="scroll-chevron" width="24" height="13" viewBox="0 0 18 10" fill="none">
            <path d="M1 1l8 7.5L17 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="scroll-line" aria-hidden="true" />
        </a>
      </section>

      {/* ── GAME ANALYSIS ── */}
      <section
        id="game-analysis"
        data-section-id="game-analysis"
        className={
          visibleSections.has('game-analysis')
            ? 'home-panel module-panel game-panel is-visible'
            : 'home-panel module-panel game-panel'
        }
      >
        <div className="section-inner">
          <div className="panel-copy">

            <p className="panel-kicker">Game Analysis</p>
            <h2 className="panel-title">See what actually won the game</h2>
            <p className="panel-description">
              Decompose any game into the eight core factors that drive outcomes.
              See which factors helped which team, track the scoring differential and
              win probability play-by-play, exclude garbage-time and get AI-generated analysis of the key factors.
            </p>
            <ul className="panel-highlights">
              <li>Factor contributions to rating differential</li>
              <li>Show statistics with or without garbage-time included</li>
              <li>Game timelines identifying clutch-time and garbage-time</li>
              <li>Excitement and comeback rankings for every game</li>
              <li>AI-generated post-game analysis</li>
            </ul>
            <Link to="/four-factor" className="panel-cta">
              Open Game Analysis
              <span className="cta-arrow">&rarr;</span>
            </Link>
          </div>

          <div className="panel-preview panel-preview--filmstrip" aria-hidden="true">
            <div className="filmstrip-track">
              {[0, 1].map((copy) => (
                <div className="filmstrip-set" key={copy}>
                  {['ga-00', 'ga-01', 'ga-02', 'ga-03', 'ga-04', 'ga-05'].map((id) => (
                    <img
                      key={`${copy}-${id}`}
                      src={`/previews/${id}.png`}
                      alt=""
                      className="filmstrip-shot"
                      loading="lazy"
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
        <a href="#league-summary" className="scroll-cue section-scroll-cue" aria-label="Scroll to next section">
          <svg className="scroll-chevron" width="24" height="13" viewBox="0 0 18 10" fill="none">
            <path d="M1 1l8 7.5L17 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </a>
      </section>

      {/* ── LEAGUE SUMMARY ── */}
      <section
        id="league-summary"
        data-section-id="league-summary"
        className={
          visibleSections.has('league-summary')
            ? 'home-panel module-panel league-panel is-visible'
            : 'home-panel module-panel league-panel'
        }
      >
        <div className="section-inner section-inner--reverse">
          <div className="panel-copy">

            <p className="panel-kicker">League Summary</p>
            <h2 className="panel-title">Rank every team in any stat, over any period of time</h2>
            <p className="panel-description">
              Sort across offensive and defensive metrics and net rating drivers
              over any time horizon to see where every team really stands.
            </p>
            <ul className="panel-highlights">
              <li>Sortable rankings across all major stat categories over any time period</li>
              <li>Show stats with or without garbage-time included</li>
              <li>Show stats during clutch-time</li>
              <li>Efficiency Landscape scatter plot for any selected period</li>
              <li>Top contributing factors to team performance over the period</li>
            </ul>
            <Link to="/league-summary" className="panel-cta">
              Open League Summary
              <span className="cta-arrow">&rarr;</span>
            </Link>
          </div>

          <div className="panel-preview panel-preview--filmstrip ls-filmstrip" aria-hidden="true">
            <div className="filmstrip-track">
              {[0, 1].map((copy) => (
                <div className="filmstrip-set" key={copy}>
                  {['ls-01', 'ls-02', 'ls-03'].map((id) => (
                    <img
                      key={`${copy}-${id}`}
                      src={`/previews/${id}.png`}
                      alt=""
                      className="filmstrip-shot"
                      loading="lazy"
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
        <a href="#statistical-trends" className="scroll-cue section-scroll-cue" aria-label="Scroll to next section">
          <svg className="scroll-chevron" width="24" height="13" viewBox="0 0 18 10" fill="none">
            <path d="M1 1l8 7.5L17 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </a>
      </section>

      {/* ── STATISTICAL TRENDS ── */}
      <section
        id="statistical-trends"
        data-section-id="statistical-trends"
        className={
          visibleSections.has('statistical-trends')
            ? 'home-panel module-panel trends-panel is-visible'
            : 'home-panel module-panel trends-panel'
        }
      >
        <div className="section-inner">
          <div className="panel-copy">

            <p className="panel-kicker">Statistical Trends</p>
            <h2 className="panel-title">Track how performance changes over time</h2>
            <p className="panel-description">
              Visualize team metrics as a time series, identify streaks, and connect sudden changes in results
              to shifts in underlying four-factor performance.
            </p>
            <ul className="panel-highlights">
              <li>Time-series views for key stats</li>
              <li>Context for hot and cold stretches</li>
              <li>Quick pivots between teams and metrics</li>
              <li>Click any game to jump straight to Game Analysis</li>
            </ul>
            <Link to="/trends" className="panel-cta">
              Open Statistical Trends
              <span className="cta-arrow">&rarr;</span>
            </Link>
          </div>

          <div className="panel-preview panel-preview--filmstrip st-filmstrip" aria-hidden="true">
            <div className="filmstrip-track">
              {[0, 1].map((copy) => (
                <div className="filmstrip-set" key={copy}>
                  {['st-01', 'st-02'].map((id) => (
                    <img
                      key={`${copy}-${id}`}
                      src={`/previews/${id}.png`}
                      alt=""
                      className="filmstrip-shot"
                      loading="lazy"
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
        <a href="#contribution-analysis" className="scroll-cue section-scroll-cue" aria-label="Scroll to next section">
          <svg className="scroll-chevron" width="24" height="13" viewBox="0 0 18 10" fill="none">
            <path d="M1 1l8 7.5L17 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </a>
      </section>

      {/* ── CONTRIBUTION ANALYSIS ── */}
      <section
        id="contribution-analysis"
        data-section-id="contribution-analysis"
        className={
          visibleSections.has('contribution-analysis')
            ? 'home-panel module-panel contribution-panel is-visible'
            : 'home-panel module-panel contribution-panel'
        }
      >
        <div className="section-inner section-inner--reverse">
          <div className="panel-copy">

            <p className="panel-kicker">Contribution Analysis</p>
            <h2 className="panel-title">Decompose team performance using an eight-factor model</h2>
            <p className="panel-description">
              Quantify how each team-side and opponent-side factor contributes to net rating over any time period.
              Spot the exact strengths driving performance.
            </p>
            <ul className="panel-highlights">
              <li>Eight-factor team profile at a glance</li>
              <li>Positive and negative contributions separated cleanly</li>
              <li>Top contributing factors with game-by-game mini charts</li>
            </ul>
            <Link to="/contribution-analysis" className="panel-cta">
              Open Contribution Analysis
              <span className="cta-arrow">&rarr;</span>
            </Link>
          </div>

          <div className="panel-preview panel-preview--filmstrip ca-filmstrip" aria-hidden="true">
            <div className="filmstrip-track">
              {[0, 1].map((copy) => (
                <div className="filmstrip-set" key={copy}>
                  {['ca-01', 'ca-02'].map((id) => (
                    <img
                      key={`${copy}-${id}`}
                      src={`/previews/${id}.png`}
                      alt=""
                      className="filmstrip-shot"
                      loading="lazy"
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Home
