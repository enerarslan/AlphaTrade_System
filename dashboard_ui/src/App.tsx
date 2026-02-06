import { Suspense, lazy, useEffect } from "react";
import { BrowserRouter as Router, Navigate, Route, Routes } from "react-router-dom";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { hasPermissionForRole, useStore } from "@/lib/store";
import "./App.css";

const OverviewPage = lazy(() => import("@/pages/overview"));
const TradingTerminalPage = lazy(() => import("@/pages/trading"));
const RiskWarRoomPage = lazy(() => import("@/pages/risk"));
const OperationsPage = lazy(() => import("@/pages/system-status"));
const ModelsPage = lazy(() => import("@/pages/models"));
const AlertsPage = lazy(() => import("@/pages/alerts"));
const SettingsPage = lazy(() => import("@/pages/settings"));
const LoginPage = lazy(() => import("@/pages/login"));

function GuardedRoute({ permission, children }: { permission: string; children: import("react").ReactElement }) {
  const { role } = useStore();
  if (!hasPermissionForRole(role, permission)) {
    return <Navigate to="/" replace />;
  }
  return children;
}

function RouteFallback() {
  return (
    <div className="flex min-h-[40vh] items-center justify-center">
      <div className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm text-slate-600 shadow-sm">Loading module...</div>
    </div>
  );
}

function ProtectedApp() {
  const { token, initialize } = useStore();

  useEffect(() => {
    void initialize();
  }, [initialize]);

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  return (
    <DashboardLayout>
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route
            path="/trading"
            element={
              <GuardedRoute permission="control.trading.status">
                <TradingTerminalPage />
              </GuardedRoute>
            }
          />
          <Route
            path="/risk"
            element={
              <GuardedRoute permission="risk.advanced.read">
                <RiskWarRoomPage />
              </GuardedRoute>
            }
          />
          <Route
            path="/models"
            element={
              <GuardedRoute permission="models.governance.read">
                <ModelsPage />
              </GuardedRoute>
            }
          />
          <Route
            path="/operations"
            element={
              <GuardedRoute permission="operations.sre.read">
                <OperationsPage />
              </GuardedRoute>
            }
          />
          <Route path="/alerts" element={<AlertsPage />} />
          <Route
            path="/settings"
            element={
              <GuardedRoute permission="control.risk.kill_switch.reset">
                <SettingsPage />
              </GuardedRoute>
            }
          />
          <Route path="/system-status" element={<Navigate to="/operations" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </DashboardLayout>
  );
}

function App() {
  return (
    <Router>
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/*" element={<ProtectedApp />} />
        </Routes>
      </Suspense>
    </Router>
  );
}

export default App;
