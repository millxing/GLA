import { Outlet, NavLink } from 'react-router-dom'
import './Layout.css'

function Layout() {
  return (
    <div className="layout">
      <header className="header">
        <div className="header-content">
          <NavLink to="/" className="logo">
            NBA Game Log Analytics
          </NavLink>
          <nav className="nav">
            <NavLink to="/four-factor" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Factor Analysis
            </NavLink>
            <NavLink to="/league-summary" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              League Summary
            </NavLink>
            <NavLink to="/trends" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Statistical Trends
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="main">
        <Outlet />
      </main>
      <footer className="footer">
        <div className="footer-content">
          Data sourced from NBA game logs. Built with React and FastAPI.
        </div>
      </footer>
    </div>
  )
}

export default Layout
