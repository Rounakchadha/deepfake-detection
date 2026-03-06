import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import Detect from './pages/Detect'
import HowItWorks from './pages/HowItWorks'
import Metrics from './pages/Metrics'
import About from './pages/About'

export default function App() {
    return (
        <Layout>
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/detect" element={<Detect />} />
                <Route path="/how-it-works" element={<HowItWorks />} />
                <Route path="/metrics" element={<Metrics />} />
                <Route path="/about" element={<About />} />
            </Routes>
        </Layout>
    )
}
