import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import SystemStatusPage from "@/pages/system-status";
import OverviewPage from "@/pages/overview";
import TradingTerminalPage from "@/pages/trading";
import RiskWarRoomPage from "@/pages/risk";
import './App.css'

function App() {
  return (
    <Router>
      <DashboardLayout>
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/system-status" element={<SystemStatusPage />} />
          <Route path="/trading" element={<TradingTerminalPage />} />
          <Route path="/risk" element={<RiskWarRoomPage />} />
          <Route path="/logs" element={<div className="p-4">System Logs (Coming Soon)</div>} />
        </Routes>
      </DashboardLayout>
    </Router>
  );
}

export default App;
