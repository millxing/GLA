import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import FourFactor from './pages/FourFactor'
import LeagueSummary from './pages/LeagueSummary'
import Trends from './pages/Trends'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="four-factor" element={<FourFactor />} />
          <Route path="league-summary" element={<LeagueSummary />} />
          <Route path="trends" element={<Trends />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
