import React, { useState, useEffect } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { checkHealth } from '../utils/api'

const NAV = [
    { to: '/', icon: '🏠', label: 'Home' },
    { to: '/detect', icon: '🔍', label: 'Detect' },
    { to: '/how-it-works', icon: '📖', label: 'How It Works' },
    { to: '/metrics', icon: '📊', label: 'Results & Metrics' },
    { to: '/about', icon: 'ℹ️', label: 'About' },
]

export default function Layout({ children }) {
    const [online, setOnline] = useState(null)

    useEffect(() => {
        let mounted = true
        const check = () => checkHealth().then(ok => mounted && setOnline(ok)).catch(() => mounted && setOnline(false))
        check()
        const id = setInterval(check, 15000)
        return () => { mounted = false; clearInterval(id) }
    }, [])

    return (
        <div className="layout">
            <aside className="sidebar">
                <div className="sidebar-logo">
                    <div style={{ fontSize: '2rem', marginBottom: '0.4rem' }}>🛡️</div>
                    <h1>Deepfake<br />Detection System</h1>
                    <p>EfficientNet-B0 + Grad-CAM XAI</p>
                </div>

                <nav className="sidebar-nav">
                    {NAV.map(({ to, icon, label }) => (
                        <NavLink
                            key={to}
                            to={to}
                            end={to === '/'}
                            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
                        >
                            <span>{icon}</span>
                            <span>{label}</span>
                        </NavLink>
                    ))}
                </nav>

                <div className="sidebar-footer">
                    {online === null ? (
                        <div className="status-badge" style={{ color: 'var(--muted)', borderColor: 'var(--border)', background: 'transparent' }}>
                            <span className="status-dot" style={{ background: 'var(--muted)' }} />
                            Checking...
                        </div>
                    ) : (
                        <div className={`status-badge${online ? '' : ' offline'}`}>
                            <span className={`status-dot ${online ? 'status-online' : 'status-offline'}`} />
                            {online ? 'Backend Connected' : 'Backend Offline'}
                        </div>
                    )}
                    <div style={{ fontSize: '0.7rem', color: 'var(--muted)', marginTop: '0.5rem' }}>
                        <div style={{ fontFamily: 'JetBrains Mono, monospace' }}>v1.0 · EfficientNet-B0</div>
                        <div>~5.3M params · MPS/CUDA</div>
                    </div>
                </div>
            </aside>

            <main className="main-content">{children}</main>
        </div>
    )
}
