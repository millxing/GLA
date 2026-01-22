import { useEffect, useState } from "react";
import { getSeasons } from "../api";
import { Link } from 'react-router-dom'
import './Home.css'

function Home() {

  const [seasons, setSeasons] = useState([]);
  const [loadingSeasons, setLoadingSeasons] = useState(true);
  const [seasonsError, setSeasonsError] = useState(null);
  const [selectedSeason, setSelectedSeason] = useState("");

  useEffect(() => {
    getSeasons()
      .then((data) => {
        setSeasons(data.seasons);
        // auto-select the most recent season
        if (data.seasons.length > 0) {
          setSelectedSeason(data.seasons[0]);
        }

        setLoadingSeasons(false);
      })
      .catch((err) => {
        setSeasonsError(err.message);
        setLoadingSeasons(false);
      });
  }, []);

  return (
    <div className="home container">
      <div className="hero">
        <h1 className="hero-title">NBA Game Log Analytics</h1>
        <p className="hero-subtitle">
          Advanced basketball statistics analysis using Dean Oliver's Four Factors framework
        </p>
      </div>

      <div className="features grid grid-3">
        <Link to="/four-factor" className="feature-card card">
          <div className="feature-icon">📊</div>
          <h2 className="feature-title">Four-Factor Game Decomposition</h2>
          <p className="feature-description">
            Analyze individual games using Dean Oliver's Four Factors framework to determine
            what factors decided the outcome. Compare home and road team performance across
            shooting efficiency, ball handling, rebounding, and free throw rate.
          </p>
          <span className="feature-link">Analyze Games</span>
        </Link>

        <Link to="/league-summary" className="feature-card card">
          <div className="feature-icon">📋</div>
          <h2 className="feature-title">League Summary Table</h2>
          <p className="feature-description">
            View league-wide team rankings across 23 statistical categories with date filtering.
            Sort by any metric to see how teams compare in offensive rating, defensive rating,
            shooting percentages, and more.
          </p>
          <span className="feature-link">View Rankings</span>
        </Link>

        <Link to="/trends" className="feature-card card">
          <div className="feature-icon">📈</div>
          <h2 className="feature-title">Statistical Trends</h2>
          <p className="feature-description">
            Plot time-series of any team's statistics over a season with moving averages.
            Track performance trends, identify hot and cold streaks, and analyze how teams
            have evolved throughout the season.
          </p>
          <span className="feature-link">Explore Trends</span>
        </Link>
      </div>

      <div className="about card">
        <h2 className="card-title">About the Four Factors</h2>
        <p>
          Dean Oliver's Four Factors framework identifies the key determinants of basketball success:
        </p>
        <ul className="factors-list">
          <li>
            <strong>Effective Field Goal Percentage (eFG%)</strong> - Shooting efficiency adjusted for three-pointers
          </li>
          <li>
            <strong>Turnover Rate (TOV%)</strong> - Ball handling and ball security
          </li>
          <li>
            <strong>Offensive Rebounding Percentage (OREB%)</strong> - Second chance opportunities
          </li>
          <li>
            <strong>Free Throw Rate (FT Rate)</strong> - Getting to the line and converting
          </li>
        </ul>
        <p>
          Research shows that shooting (eFG%) is the most important factor, followed by turnovers,
          rebounding, and free throws. This application uses regression models trained on NBA game data
          to quantify exactly how much each factor contributed to a game's outcome.
        </p>
      </div>
    </div>
  )
}

export default Home
