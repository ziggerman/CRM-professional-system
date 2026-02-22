import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import Dashboard from './pages/Dashboard.tsx'
import Leads from './pages/Leads.tsx'
import Sales from './pages/Sales.tsx'
import { Sidebar } from './components/Sidebar.tsx'

const Placeholder = ({ title }: { title: string }) => (
  <div className="flex flex-col items-center justify-center p-24 h-full text-zinc-400">
    <h1 className="text-3xl font-bold mb-4">{title}</h1>
    <p>This module is under construction.</p>
  </div>
);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Sidebar>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/leads" element={<Leads />} />
          <Route path="/sales" element={<Sales />} />
          <Route path="/broadcast" element={<Placeholder title="Broadcast" />} />
          <Route path="/settings" element={<Placeholder title="Settings" />} />
        </Routes>
      </Sidebar>
    </BrowserRouter>
  </StrictMode>,
)
