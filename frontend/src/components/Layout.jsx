import { Outlet, NavLink, useLocation } from 'react-router-dom'
import './Layout.css'

function Layout() {
  const location = useLocation()

  const handleLogoClick = (e) => {
    if (location.pathname === '/') {
      e.preventDefault()
      window.scrollTo({ top: 0, behavior: 'smooth' })
    }
  }

  return (
    <div className="layout">
      <header className="header">
        <div className="header-content">
          <NavLink to="/" className="logo" onClick={handleLogoClick}>
            Extra Pass Analytics
          </NavLink>
          <nav className="nav">
            <NavLink to="/four-factor" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Game Analysis
            </NavLink>
            <NavLink to="/league-summary" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              League Summary
            </NavLink>
            <NavLink to="/trends" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Statistical Trends
            </NavLink>
            <NavLink to="/contribution-analysis" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Contribution Analysis
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="main">
        <Outlet />
      </main>
      <footer className="footer">
        <div className="footer-content">
          Data sourced with{' '}
          <a href="https://github.com/swar/nba_api" target="_blank" rel="noreferrer">
            NBA_API
          </a>
          . Built with React and FastAPI by{' '}
          <a href="https://entangledparticles.xyz" target="_blank" rel="noreferrer">
            Entangled Paricles
          </a>
        </div>
      </footer>
    </div>
  )
}

export default Layout
