import { useEffect, useState } from 'react'
import { Outlet, NavLink, useLocation, useNavigationType } from 'react-router-dom'
import './Layout.css'

const NAV_ITEMS = [
  { to: '/four-factor', label: 'Game Analysis' },
  { to: '/league-summary', label: 'League Summary' },
  { to: '/trends', label: 'Statistical Trends' },
  { to: '/contribution-analysis', label: 'Contribution Analysis' },
  { to: '/blog', label: 'Blog' },
]

const SITE_NAME = 'Extra Pass Analytics'

function Layout() {
  const location = useLocation()
  const [menuOpen, setMenuOpen] = useState(false)
  const isHome = location.pathname === '/'

  const navigationType = useNavigationType()

  // Close the mobile menu whenever the route changes
  useEffect(() => {
    setMenuOpen(false)
  }, [location.pathname])

  // New pages open at the top; Back/Forward (POP) keeps the browser's own scroll restoration.
  // Keyed on pathname only, so filter/query-string changes never jump the page.
  useEffect(() => {
    if (navigationType !== 'POP') window.scrollTo(0, 0)
  }, [location.pathname]) // eslint-disable-line react-hooks/exhaustive-deps

  // Per-route document title (blog posts set their own title in BlogPost)
  useEffect(() => {
    if (location.pathname.startsWith('/blog/')) return
    const item = NAV_ITEMS.find((nav) => nav.to === location.pathname)
    document.title = item ? `${item.label} · ${SITE_NAME}` : SITE_NAME
  }, [location.pathname])

  useEffect(() => {
    if (!menuOpen) return undefined
    const onKey = (event) => {
      if (event.key === 'Escape') setMenuOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [menuOpen])

  const handleLogoClick = (event) => {
    if (isHome) {
      event.preventDefault()
      window.scrollTo({ top: 0, behavior: 'smooth' })
    }
  }

  return (
    <div className={isHome ? 'layout layout--home' : 'layout'}>
      <a className="site-skip-link" href="#main-content">Skip to content</a>
      <header className="site-header">
        <div className="site-header-inner">
          <NavLink to="/" className="site-logo" onClick={handleLogoClick}>
            <img src="/basketball.svg" alt="" className="site-logo-mark" width="22" height="22" />
            <span>Extra Pass Analytics</span>
          </NavLink>
          <button
            type="button"
            className="site-nav-toggle"
            aria-expanded={menuOpen}
            aria-controls="primary-nav"
            aria-label={menuOpen ? 'Close menu' : 'Open menu'}
            onClick={() => setMenuOpen((open) => !open)}
          >
            <span className={`site-nav-toggle-icon ${menuOpen ? 'is-open' : ''}`} aria-hidden="true" />
          </button>
          <nav id="primary-nav" className={`site-nav ${menuOpen ? 'is-open' : ''}`} aria-label="Primary">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => (isActive ? 'site-nav-link active' : 'site-nav-link')}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="site-main" id="main-content" tabIndex={-1}>
        <Outlet />
      </main>
      <footer className="site-footer">
        <div className="site-footer-inner">
          Data via{' '}
          <a href="https://github.com/swar/nba_api" target="_blank" rel="noreferrer">
            nba_api
          </a>
          {' · '}Built by{' '}
          <a href="https://entangledparticles.xyz" target="_blank" rel="noreferrer">
            Entangled Particles
          </a>
        </div>
      </footer>
    </div>
  )
}

export default Layout
