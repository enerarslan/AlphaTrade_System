import { create } from "zustand";
import { api, buildApiUrl, clearAccessToken, getAccessToken, setAccessToken } from "./api";

type HealthCheckResult = {
  component: string;
  status: "HEALTHY" | "DEGRADED" | "UNHEALTHY" | "UNKNOWN" | "CRITICAL";
  message: string;
  latency_ms: number;
  details: Record<string, unknown>;
  timestamp: string;
};

type SystemHealth = {
  status: string;
  timestamp: string;
  checks: HealthCheckResult[];
};

type Portfolio = {
  timestamp: string;
  equity: number;
  cash: number;
  buying_power: number;
  positions_count: number;
  long_exposure: number;
  short_exposure: number;
  net_exposure: number;
  gross_exposure: number;
  daily_pnl: number;
  total_pnl: number;
};

type Position = {
  symbol: string;
  quantity: number;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  cost_basis: number;
};

type Order = {
  order_id: string;
  symbol: string;
  side: string;
  order_type: string;
  quantity: number;
  filled_qty: number;
  limit_price: number | null;
  stop_price: number | null;
  status: string;
  created_at: string;
  updated_at: string;
};

type RiskMetrics = {
  portfolio_var_95: number;
  portfolio_var_99: number | null;
  current_drawdown: number;
  max_drawdown_30d: number;
  largest_position_pct: number;
  sector_exposures: Record<string, number>;
  beta_exposure: number | null;
  correlation_risk: number | null;
};

type TCAResponse = {
  timestamp: string;
  slippage_bps: number;
  market_impact_bps: number;
  execution_speed_ms: number;
  fill_probability: number;
  venue_breakdown: Record<string, number>;
  cost_savings_vs_vwap: number;
};

type ExecutionQualityResponse = {
  timestamp: string;
  arrival_price_delta_bps: number;
  rejection_rate: number;
  fill_rate_buy: number;
  fill_rate_sell: number;
  venue_slippage_buckets: Record<string, number>;
  latency_distribution_ms: Record<string, number>;
};

type VaRResponse = {
  timestamp: string;
  var_95: number;
  var_99: number;
  cvar_95: number;
  stress_scenarios: Record<string, number>;
  distribution_curve: { pnl: number; probability: number }[];
};

type RiskConcentrationResponse = {
  timestamp: string;
  largest_symbol_pct: number;
  top3_symbols_pct: number;
  hhi_symbol: number;
  hhi_sector: number;
  symbol_weights: Record<string, number>;
  sector_weights: Record<string, number>;
};

type RiskCorrelationResponse = {
  timestamp: string;
  average_pairwise_correlation: number;
  cluster_risk_score: number;
  beta_weighted_exposure: number;
  matrix: Array<{ symbol_a: string; symbol_b: string; correlation: number }>;
};

type RiskStressResponse = {
  timestamp: string;
  scenarios: Record<string, number>;
  worst_case_loss: number;
  resilience_score: number;
};

type RiskAttributionResponse = {
  timestamp: string;
  pre_trade_checks: Array<Record<string, unknown>>;
  post_trade_findings: Array<Record<string, unknown>>;
  breaches_count: number;
};

type ExplainabilityResponse = {
  timestamp: string;
  model_name: string;
  global_importance: Record<string, number>;
  recent_shift: Record<string, number>;
};

type ModelRegistryEntry = {
  model_name: string;
  version_id: string;
  model_version: string;
  model_type: string;
  registered_at: string;
  metrics: Record<string, unknown>;
  tags: string[];
  is_active: boolean;
  path: string;
};

type ModelRegistryResponse = {
  timestamp: string;
  model_count: number;
  versions_count: number;
  active_model: string | null;
  entries: ModelRegistryEntry[];
};

type ModelDriftResponse = {
  timestamp: string;
  model_name: string;
  drift_score: number;
  drift_status: string;
  staleness_reason: string | null;
  recommendation: string;
  feature_shift: Record<string, number>;
};

type ModelValidationGateResponse = {
  timestamp: string;
  model_name: string;
  passed: boolean;
  gates: Array<Record<string, unknown>>;
  failed_gates: string[];
  decision: string;
};

type ChampionChallengerResponse = {
  timestamp: string;
  champion: string | null;
  challenger: string | null;
  comparison: Record<string, number>;
  recommendation: string;
};

type AlertItem = {
  alert_id: string;
  alert_type: string;
  severity: string;
  title: string;
  message: string;
  status: string;
  timestamp: string;
  context: Record<string, unknown>;
};

type LogEntry = {
  timestamp: string;
  level: string;
  category: string;
  message: string;
  extra?: Record<string, unknown>;
};

type AuditRecord = {
  timestamp: string;
  action: string;
  user: string;
  status: string;
  details: Record<string, unknown>;
  prev_hash: string;
  record_hash: string;
};

type MfaStatus = {
  username: string;
  role: string;
  mfa_enabled: boolean;
  mfa_required_for_privileged_actions: boolean;
};

type SloStatus = {
  timestamp: string;
  availability: number;
  error_budget_remaining_pct: number;
  p95_action_latency_ms: number;
  p99_action_latency_ms: number;
  burn_rate_1h: number;
  burn_rate_6h: number;
  status: string;
};

type IncidentRecord = {
  incident_id: string;
  severity: string;
  title: string;
  status: string;
  created_at: string;
  runbook_link: string | null;
  suggested_action: string | null;
};

type IncidentTimelineEvent = {
  timestamp: string;
  source: string;
  event_type: string;
  severity: string;
  message: string;
  context: Record<string, unknown>;
};

type RunbookRecord = {
  alert_type: string;
  runbook_path: string;
  suggested_action: string | null;
};

type AdminUserRecord = {
  username: string;
  role: string;
  has_mfa: boolean;
  role_source: string;
};

type SecurityStatus = {
  jwt_key_count: number;
  active_key_fingerprint: string;
  require_auth: boolean;
  api_key_enabled: boolean;
  mfa_enabled_users: number;
  role_overrides_count: number;
  rate_limit_enabled: boolean;
  rate_limit_limit: number;
  rate_limit_window_seconds: number;
};

type SsoStatus = {
  enabled: boolean;
  configured: boolean;
  issuer: string | null;
  authorization_endpoint: string | null;
  token_endpoint: string | null;
  userinfo_endpoint: string | null;
  redirect_uri: string | null;
  username_claim: string;
  role_claim: string;
  scopes: string[];
};

type SiemStatus = {
  enabled: boolean;
  endpoint: string | null;
  queue_depth: number;
  total_enqueued: number;
  total_delivered: number;
  total_failed: number;
  last_flush_at: string | null;
  last_success_at: string | null;
  last_error: string | null;
};

type MfaEnrollInitResponse = {
  username: string;
  secret: string;
  provisioning_uri: string;
  issuer: string;
};

type TradingStatus = {
  running: boolean;
  pid: number | null;
  started_at: string | null;
};

type CommandJob = {
  job_id: string;
  command: string;
  args: string[];
  status: string;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  exit_code: number | null;
  output: string;
};

type LoginResponse = {
  access_token: string;
  token_type: string;
  expires_in: number;
};

type UserInfo = {
  username: string;
  role?: string;
  mfa_enabled?: boolean;
};

export type DashboardRole = "viewer" | "operator" | "risk" | "admin";

const rolePermissions: Record<DashboardRole, Set<string>> = {
  viewer: new Set(["read.basic"]),
  operator: new Set([
    "read.basic",
    "alerts.manage",
    "control.trading.status",
    "control.trading.start",
    "control.trading.stop",
    "control.trading.restart",
    "control.jobs.create",
    "control.jobs.read",
    "control.jobs.cancel",
    "models.governance.read",
    "operations.sre.read",
    "operations.runbooks.execute",
  ]),
  risk: new Set([
    "read.basic",
    "alerts.manage",
    "risk.advanced.read",
    "control.audit.read",
    "control.trading.status",
    "control.risk.kill_switch.activate",
    "control.risk.kill_switch.reset",
    "control.jobs.read",
    "models.governance.read",
    "operations.sre.read",
  ]),
  admin: new Set(["*"]),
};

export function resolveRole(user: UserInfo | null): DashboardRole {
  const role = (user?.role ?? "viewer").toLowerCase();
  if (role === "admin" || role === "operator" || role === "risk" || role === "viewer") {
    return role;
  }
  return "viewer";
}

export function hasPermissionForRole(role: DashboardRole, permission: string) {
  const grants = rolePermissions[role] ?? rolePermissions.viewer;
  if (grants.has("*") || grants.has(permission)) {
    return true;
  }
  if (permission.startsWith("read.") && grants.has("read.basic")) {
    return true;
  }
  return false;
}

type StartTradingPayload = {
  mode: "live" | "paper" | "dry-run";
  symbols: string[];
  strategy: string;
  capital: number;
  mfa_code?: string;
};

type WsState = {
  portfolio: boolean;
  orders: boolean;
  signals: boolean;
  alerts: boolean;
};

type DashboardState = {
  token: string | null;
  user: UserInfo | null;
  role: DashboardRole;
  authError: string | null;
  isInitializing: boolean;
  isLoading: boolean;
  lastRefreshAt: string | null;

  health: SystemHealth | null;
  portfolio: Portfolio | null;
  positions: Position[];
  orders: Order[];
  riskMetrics: RiskMetrics | null;
  tca: TCAResponse | null;
  executionQuality: ExecutionQualityResponse | null;
  varData: VaRResponse | null;
  riskConcentration: RiskConcentrationResponse | null;
  riskCorrelation: RiskCorrelationResponse | null;
  riskStress: RiskStressResponse | null;
  riskAttribution: RiskAttributionResponse | null;
  explainability: ExplainabilityResponse | null;
  modelRegistry: ModelRegistryResponse | null;
  modelDrift: ModelDriftResponse | null;
  modelValidation: ModelValidationGateResponse | null;
  championChallenger: ChampionChallengerResponse | null;
  mfaStatus: MfaStatus | null;
  mfaEnrollment: MfaEnrollInitResponse | null;
  adminUsers: AdminUserRecord[];
  securityStatus: SecurityStatus | null;
  ssoStatus: SsoStatus | null;
  siemStatus: SiemStatus | null;
  alerts: AlertItem[];
  logs: LogEntry[];
  auditTrail: AuditRecord[];
  sloStatus: SloStatus | null;
  incidents: IncidentRecord[];
  incidentTimeline: IncidentTimelineEvent[];
  runbooks: RunbookRecord[];
  tradingStatus: TradingStatus | null;
  jobs: CommandJob[];
  ws: WsState;
  error: string | null;

  hasPermission: (permission: string) => boolean;
  initialize: () => Promise<void>;
  login: (username: string, password: string) => Promise<boolean>;
  fetchSsoStatus: () => Promise<void>;
  startSsoLogin: () => void;
  completeSsoLoginFromHash: (hash: string) => Promise<boolean>;
  logout: () => void;
  fetchSnapshot: () => Promise<void>;
  fetchHealth: () => Promise<void>;
  fetchPortfolio: () => Promise<void>;
  fetchPositions: () => Promise<void>;
  fetchOrders: () => Promise<void>;
  fetchRiskMetrics: () => Promise<void>;
  fetchTCA: () => Promise<void>;
  fetchExecutionQuality: () => Promise<void>;
  fetchVar: () => Promise<void>;
  fetchRiskConcentration: () => Promise<void>;
  fetchRiskCorrelation: () => Promise<void>;
  fetchRiskStress: () => Promise<void>;
  fetchRiskAttribution: () => Promise<void>;
  fetchExplainability: () => Promise<void>;
  fetchModelRegistry: () => Promise<void>;
  fetchModelDrift: () => Promise<void>;
  fetchModelValidation: () => Promise<void>;
  fetchChampionChallenger: () => Promise<void>;
  fetchMfaStatus: () => Promise<void>;
  initMfaEnrollment: () => Promise<void>;
  verifyMfaEnrollment: (code: string) => Promise<void>;
  disableMfaEnrollment: (code: string) => Promise<void>;
  fetchAdminUsers: () => Promise<void>;
  updateUserRole: (username: string, role: DashboardRole, reason?: string, mfaCode?: string) => Promise<void>;
  fetchSecurityStatus: () => Promise<void>;
  rotateJwtSecret: (newSecret: string, mfaCode?: string) => Promise<void>;
  fetchSiemStatus: () => Promise<void>;
  flushSiemQueue: (maxBatches?: number) => Promise<void>;
  exportAuditTrail: (format?: "json" | "jsonl") => Promise<void>;
  fetchAlerts: () => Promise<void>;
  acknowledgeAlert: (alertId: string, acknowledgedBy: string) => Promise<void>;
  resolveAlert: (alertId: string) => Promise<void>;
  fetchLogs: () => Promise<void>;
  fetchAuditTrail: () => Promise<void>;
  fetchSloStatus: () => Promise<void>;
  fetchIncidents: () => Promise<void>;
  fetchIncidentTimeline: () => Promise<void>;
  fetchRunbooks: () => Promise<void>;
  fetchTradingStatus: () => Promise<void>;
  fetchJobs: () => Promise<void>;
  promoteChampion: (modelName: string, versionId: string, reason?: string, mfaCode?: string) => Promise<void>;
  startTrading: (payload: StartTradingPayload) => Promise<void>;
  stopTrading: () => Promise<void>;
  restartTrading: (payload: StartTradingPayload) => Promise<void>;
  activateKillSwitch: (reason: string, mfaCode?: string) => Promise<void>;
  resetKillSwitch: (authorizedBy: string, force?: boolean, overrideCode?: string, mfaCode?: string) => Promise<void>;
  createJob: (command: string, args: string[]) => Promise<void>;
  cancelJob: (jobId: string) => Promise<void>;
  executeRunbookAction: (action: string, mfaCode?: string) => Promise<void>;
  connectLiveChannels: () => void;
  disconnectLiveChannels: () => void;
};

const sockets: Record<string, WebSocket | null> = {
  portfolio: null,
  orders: null,
  signals: null,
  alerts: null,
};

const websocketBaseUrl = (() => {
  const base = String(api.defaults.baseURL ?? "http://127.0.0.1:8000");
  if (/^https?:\/\//i.test(base)) {
    return base.replace(/^http/i, "ws");
  }
  if (typeof window !== "undefined" && base.startsWith("/")) {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    return `${scheme}://${window.location.host}${base}`;
  }
  return "ws://127.0.0.1:8000";
})();

function buildIdempotencyKey(action: string) {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${action}-${crypto.randomUUID()}`;
  }
  return `${action}-${Date.now()}-${Math.round(Math.random() * 1e9)}`;
}

function isStatus(error: unknown, status: number) {
  if (!error || typeof error !== "object") {
    return false;
  }
  const maybeResponse = (error as { response?: { status?: number } }).response;
  return maybeResponse?.status === status;
}

function connectSocket(channel: keyof WsState, onMessage: (payload: any) => void, setWs: (connected: boolean) => void) {
  const ws = new WebSocket(`${websocketBaseUrl}/ws/${channel}`);
  sockets[channel] = ws;

  ws.onopen = () => setWs(true);
  ws.onclose = () => setWs(false);
  ws.onerror = () => setWs(false);
  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      onMessage(payload);
    } catch {
      // ignore malformed messages
    }
  };
}

export const useStore = create<DashboardState>((set, get) => ({
  token: getAccessToken(),
  user: null,
  role: "viewer",
  authError: null,
  isInitializing: false,
  isLoading: false,
  lastRefreshAt: null,

  health: null,
  portfolio: null,
  positions: [],
  orders: [],
  riskMetrics: null,
  tca: null,
  executionQuality: null,
  varData: null,
  riskConcentration: null,
  riskCorrelation: null,
  riskStress: null,
  riskAttribution: null,
  explainability: null,
  modelRegistry: null,
  modelDrift: null,
  modelValidation: null,
  championChallenger: null,
  mfaStatus: null,
  mfaEnrollment: null,
  adminUsers: [],
  securityStatus: null,
  ssoStatus: null,
  siemStatus: null,
  alerts: [],
  logs: [],
  auditTrail: [],
  sloStatus: null,
  incidents: [],
  incidentTimeline: [],
  runbooks: [],
  tradingStatus: null,
  jobs: [],
  ws: { portfolio: false, orders: false, signals: false, alerts: false },
  error: null,

  hasPermission: (permission: string) => hasPermissionForRole(get().role, permission),

  initialize: async () => {
    const token = get().token;
    if (!token) {
      return;
    }
    set({ isInitializing: true, authError: null });
    try {
      const me = await api.get<UserInfo>("/auth/me");
      const role = resolveRole(me.data);
      set({ user: me.data, role });
      await get().fetchSnapshot();
      get().connectLiveChannels();
    } catch {
      clearAccessToken();
      set({ token: null, user: null, role: "viewer", authError: "Session expired. Please login again." });
    } finally {
      set({ isInitializing: false });
    }
  },

  login: async (username: string, password: string) => {
    set({ authError: null, isLoading: true });
    try {
      const response = await api.post<LoginResponse>("/auth/login", { username, password });
      setAccessToken(response.data.access_token);
      const me = await api.get<UserInfo>("/auth/me");
      const role = resolveRole(me.data);
      set({ token: response.data.access_token, user: me.data, role });
      await get().fetchSnapshot();
      get().connectLiveChannels();
      return true;
    } catch {
      set({ authError: "Login failed. Check username/password or use SSO." });
      return false;
    } finally {
      set({ isLoading: false });
    }
  },

  fetchSsoStatus: async () => {
    try {
      const response = await api.get<SsoStatus>("/auth/sso/status");
      set({ ssoStatus: response.data });
    } catch {
      set({ ssoStatus: null });
    }
  },

  startSsoLogin: () => {
    const returnTo = typeof window !== "undefined" ? `${window.location.origin}/login` : "/login";
    const ssoUrl = buildApiUrl("/auth/sso/start", {
      redirect: true,
      return_to: returnTo,
    });
    window.location.assign(ssoUrl);
  },

  completeSsoLoginFromHash: async (hash: string) => {
    if (!hash) {
      return false;
    }
    const fragment = hash.startsWith("#") ? hash.slice(1) : hash;
    const params = new URLSearchParams(fragment);
    const token = params.get("access_token");
    const error = params.get("error");
    const errorDescription = params.get("error_description");

    if (!token) {
      if (error) {
        set({ authError: errorDescription || error });
      }
      if (typeof window !== "undefined" && window.location.hash) {
        window.history.replaceState(null, document.title, window.location.pathname + window.location.search);
      }
      return false;
    }

    set({ authError: null, isLoading: true });
    try {
      setAccessToken(token);
      const me = await api.get<UserInfo>("/auth/me");
      const role = resolveRole(me.data);
      set({ token, user: me.data, role });
      await get().fetchSnapshot();
      get().connectLiveChannels();
      return true;
    } catch {
      clearAccessToken();
      set({ token: null, user: null, role: "viewer", authError: "SSO session validation failed." });
      return false;
    } finally {
      set({ isLoading: false });
      if (typeof window !== "undefined" && window.location.hash) {
        window.history.replaceState(null, document.title, window.location.pathname + window.location.search);
      }
    }
  },

  logout: () => {
    get().disconnectLiveChannels();
    clearAccessToken();
    set({
      token: null,
      user: null,
      role: "viewer",
      health: null,
      portfolio: null,
      positions: [],
      orders: [],
      riskMetrics: null,
      tca: null,
      executionQuality: null,
      varData: null,
      riskConcentration: null,
      riskCorrelation: null,
      riskStress: null,
      riskAttribution: null,
      explainability: null,
      modelRegistry: null,
      modelDrift: null,
      modelValidation: null,
      championChallenger: null,
      mfaStatus: null,
      mfaEnrollment: null,
      adminUsers: [],
      securityStatus: null,
      ssoStatus: null,
      siemStatus: null,
      alerts: [],
      logs: [],
      auditTrail: [],
      sloStatus: null,
      incidents: [],
      incidentTimeline: [],
      runbooks: [],
      jobs: [],
      tradingStatus: null,
      authError: null,
      error: null,
    });
  },

  fetchSnapshot: async () => {
    set({ isLoading: true, error: null });
    try {
      const tasks: Array<Promise<void>> = [
        get().fetchHealth(),
        get().fetchPortfolio(),
        get().fetchPositions(),
        get().fetchOrders(),
        get().fetchRiskMetrics(),
        get().fetchTCA(),
        get().fetchExecutionQuality(),
        get().fetchVar(),
        get().fetchExplainability(),
        get().fetchModelRegistry(),
        get().fetchModelDrift(),
        get().fetchModelValidation(),
        get().fetchChampionChallenger(),
        get().fetchMfaStatus(),
        get().fetchSecurityStatus(),
        get().fetchSiemStatus(),
        get().fetchAdminUsers(),
        get().fetchAlerts(),
        get().fetchLogs(),
        get().fetchSloStatus(),
        get().fetchIncidents(),
        get().fetchIncidentTimeline(),
        get().fetchRunbooks(),
        get().fetchTradingStatus(),
        get().fetchJobs(),
      ];
      await Promise.all(tasks);
      set({ lastRefreshAt: new Date().toISOString() });
    } catch {
      set({ error: "Failed to refresh dashboard snapshot" });
    } finally {
      set({ isLoading: false });
    }
  },

  fetchHealth: async () => {
    const response = await api.get<SystemHealth>("/health/detailed");
    set({ health: response.data });
  },

  fetchPortfolio: async () => {
    const response = await api.get<Portfolio>("/portfolio");
    set({ portfolio: response.data });
  },

  fetchPositions: async () => {
    const response = await api.get<Position[]>("/positions");
    set({ positions: response.data });
  },

  fetchOrders: async () => {
    const response = await api.get<Order[]>("/orders?limit=250");
    set({ orders: response.data });
  },

  fetchRiskMetrics: async () => {
    const response = await api.get<RiskMetrics>("/risk");
    set({ riskMetrics: response.data });
  },

  fetchTCA: async () => {
    const response = await api.get<TCAResponse>("/execution/tca");
    set({ tca: response.data });
  },

  fetchExecutionQuality: async () => {
    const response = await api.get<ExecutionQualityResponse>("/execution/quality");
    set({ executionQuality: response.data });
  },

  fetchVar: async () => {
    const response = await api.get<VaRResponse>("/risk/var");
    set({ varData: response.data });
  },

  fetchRiskConcentration: async () => {
    const response = await api.get<RiskConcentrationResponse>("/risk/concentration");
    set({ riskConcentration: response.data });
  },

  fetchRiskCorrelation: async () => {
    const response = await api.get<RiskCorrelationResponse>("/risk/correlation");
    set({ riskCorrelation: response.data });
  },

  fetchRiskStress: async () => {
    const response = await api.get<RiskStressResponse>("/risk/stress");
    set({ riskStress: response.data });
  },

  fetchRiskAttribution: async () => {
    const response = await api.get<RiskAttributionResponse>("/risk/attribution");
    set({ riskAttribution: response.data });
  },

  fetchExplainability: async () => {
    const response = await api.get<ExplainabilityResponse>("/models/explainability");
    set({ explainability: response.data });
  },

  fetchModelRegistry: async () => {
    if (!get().hasPermission("models.governance.read")) {
      set({ modelRegistry: null });
      return;
    }
    try {
      const response = await api.get<ModelRegistryResponse>("/models/registry");
      set({ modelRegistry: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ modelRegistry: null });
        return;
      }
      throw error;
    }
  },

  fetchModelDrift: async () => {
    if (!get().hasPermission("models.governance.read")) {
      set({ modelDrift: null });
      return;
    }
    try {
      const response = await api.get<ModelDriftResponse>("/models/drift");
      set({ modelDrift: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ modelDrift: null });
        return;
      }
      throw error;
    }
  },

  fetchModelValidation: async () => {
    if (!get().hasPermission("models.governance.read")) {
      set({ modelValidation: null });
      return;
    }
    try {
      const response = await api.get<ModelValidationGateResponse>("/models/validation-gates");
      set({ modelValidation: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ modelValidation: null });
        return;
      }
      throw error;
    }
  },

  fetchChampionChallenger: async () => {
    if (!get().hasPermission("models.governance.read")) {
      set({ championChallenger: null });
      return;
    }
    try {
      const response = await api.get<ChampionChallengerResponse>("/models/champion-challenger");
      set({ championChallenger: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ championChallenger: null });
        return;
      }
      throw error;
    }
  },

  fetchMfaStatus: async () => {
    try {
      const response = await api.get<MfaStatus>("/auth/mfa/status");
      set({ mfaStatus: response.data });
    } catch (error) {
      if (isStatus(error, 401) || isStatus(error, 403)) {
        set({ mfaStatus: null });
        return;
      }
      throw error;
    }
  },

  initMfaEnrollment: async () => {
    const response = await api.post<MfaEnrollInitResponse>("/auth/mfa/enroll/init");
    set({ mfaEnrollment: response.data });
    await get().fetchMfaStatus();
  },

  verifyMfaEnrollment: async (code: string) => {
    await api.post("/auth/mfa/enroll/verify", { code });
    set({ mfaEnrollment: null });
    await get().fetchMfaStatus();
  },

  disableMfaEnrollment: async (code: string) => {
    await api.post("/auth/mfa/disable", { code });
    set({ mfaEnrollment: null });
    await get().fetchMfaStatus();
  },

  fetchAdminUsers: async () => {
    if (!get().hasPermission("admin.users.manage")) {
      set({ adminUsers: [] });
      return;
    }
    try {
      const response = await api.get<AdminUserRecord[]>("/admin/users");
      set({ adminUsers: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ adminUsers: [] });
        return;
      }
      throw error;
    }
  },

  updateUserRole: async (username: string, role: DashboardRole, reason = "admin_update", mfaCode?: string) => {
    await api.post(
      `/admin/users/${encodeURIComponent(username)}/role`,
      {
        role,
        reason,
        mfa_code: mfaCode ?? null,
      },
      {
        headers: {
          "Idempotency-Key": buildIdempotencyKey("admin-role-update"),
          ...(mfaCode ? { "X-MFA-Code": mfaCode } : {}),
        },
      },
    );
    await get().fetchAdminUsers();
  },

  fetchSecurityStatus: async () => {
    if (!get().hasPermission("auth.security.rotate")) {
      set({ securityStatus: null });
      return;
    }
    try {
      const response = await api.get<SecurityStatus>("/auth/security/status");
      set({ securityStatus: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ securityStatus: null });
        return;
      }
      throw error;
    }
  },

  rotateJwtSecret: async (newSecret: string, mfaCode?: string) => {
    await api.post(
      "/auth/security/rotate-jwt",
      {
        new_secret: newSecret,
        mfa_code: mfaCode ?? null,
      },
      {
        headers: {
          "Idempotency-Key": buildIdempotencyKey("jwt-rotate"),
          ...(mfaCode ? { "X-MFA-Code": mfaCode } : {}),
        },
      },
    );
    await get().fetchSecurityStatus();
  },

  fetchAlerts: async () => {
    const response = await api.get<AlertItem[]>("/alerts?limit=200");
    set({ alerts: response.data });
  },

  acknowledgeAlert: async (alertId: string, acknowledgedBy: string) => {
    await api.post(
      `/alerts/${alertId}/acknowledge?acknowledged_by=${encodeURIComponent(acknowledgedBy)}`,
      {},
      {
        headers: {
          "Idempotency-Key": buildIdempotencyKey("alerts-ack"),
        },
      },
    );
    await get().fetchAlerts();
  },

  resolveAlert: async (alertId: string) => {
    await api.post(
      `/alerts/${alertId}/resolve`,
      {},
      {
        headers: {
          "Idempotency-Key": buildIdempotencyKey("alerts-resolve"),
        },
      },
    );
    await get().fetchAlerts();
  },

  fetchLogs: async () => {
    const response = await api.get<LogEntry[]>("/logs?limit=200");
    set({ logs: response.data });
  },

  fetchAuditTrail: async () => {
    if (!get().hasPermission("control.audit.read")) {
      set({ auditTrail: [] });
      return;
    }
    try {
      const response = await api.get<AuditRecord[]>("/control/audit?limit=200");
      set({ auditTrail: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ auditTrail: [] });
        return;
      }
      throw error;
    }
  },

  fetchSiemStatus: async () => {
    if (!get().hasPermission("control.audit.read")) {
      set({ siemStatus: null });
      return;
    }
    try {
      const response = await api.get<SiemStatus>("/control/audit/siem/status");
      set({ siemStatus: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ siemStatus: null });
        return;
      }
      throw error;
    }
  },

  flushSiemQueue: async (maxBatches = 3) => {
    if (!get().hasPermission("control.audit.manage")) {
      return;
    }
    await api.post(`/control/audit/siem/flush?max_batches=${encodeURIComponent(String(maxBatches))}`);
    await get().fetchSiemStatus();
  },

  exportAuditTrail: async (format = "jsonl") => {
    if (format === "json") {
      const response = await api.get<{ generated_at: string; count: number; records: AuditRecord[] }>(
        "/control/audit/export?limit=5000&format=json",
      );
      const payload = JSON.stringify(response.data, null, 2);
      const blob = new Blob([payload], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `dashboard_audit_${Date.now()}.json`;
      anchor.click();
      URL.revokeObjectURL(url);
      return;
    }
    const response = await api.get<string>("/control/audit/export?limit=5000&format=jsonl", {
      responseType: "text",
    });
    const blob = new Blob([response.data], { type: "application/x-ndjson;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `dashboard_audit_${Date.now()}.ndjson`;
    anchor.click();
    URL.revokeObjectURL(url);
  },

  fetchSloStatus: async () => {
    if (!get().hasPermission("operations.sre.read")) {
      set({ sloStatus: null });
      return;
    }
    try {
      const response = await api.get<SloStatus>("/sre/slo");
      set({ sloStatus: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ sloStatus: null });
        return;
      }
      throw error;
    }
  },

  fetchIncidents: async () => {
    if (!get().hasPermission("operations.sre.read")) {
      set({ incidents: [] });
      return;
    }
    try {
      const response = await api.get<IncidentRecord[]>("/sre/incidents?limit=200");
      set({ incidents: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ incidents: [] });
        return;
      }
      throw error;
    }
  },

  fetchIncidentTimeline: async () => {
    if (!get().hasPermission("operations.sre.read")) {
      set({ incidentTimeline: [] });
      return;
    }
    try {
      const response = await api.get<IncidentTimelineEvent[]>("/sre/incidents/timeline?limit=250");
      set({ incidentTimeline: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ incidentTimeline: [] });
        return;
      }
      throw error;
    }
  },

  fetchRunbooks: async () => {
    if (!get().hasPermission("operations.sre.read")) {
      set({ runbooks: [] });
      return;
    }
    try {
      const response = await api.get<RunbookRecord[]>("/sre/runbooks");
      set({ runbooks: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ runbooks: [] });
        return;
      }
      throw error;
    }
  },

  fetchTradingStatus: async () => {
    if (!get().hasPermission("control.trading.status")) {
      set({ tradingStatus: null });
      return;
    }
    try {
      const response = await api.get<TradingStatus>("/control/trading/status");
      set({ tradingStatus: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ tradingStatus: null });
        return;
      }
      throw error;
    }
  },

  fetchJobs: async () => {
    if (!get().hasPermission("control.jobs.read")) {
      set({ jobs: [] });
      return;
    }
    try {
      const response = await api.get<CommandJob[]>("/control/jobs?limit=100");
      set({ jobs: response.data });
    } catch (error) {
      if (isStatus(error, 403)) {
        set({ jobs: [] });
        return;
      }
      throw error;
    }
  },

  promoteChampion: async (modelName: string, versionId: string, reason = "manual_promotion", mfaCode?: string) => {
    await api.post(
      "/models/champion/promote",
      {
        model_name: modelName,
        version_id: versionId,
        reason,
        mfa_code: mfaCode ?? null,
      },
      {
        headers: {
          "Idempotency-Key": buildIdempotencyKey("model-promote"),
          ...(mfaCode ? { "X-MFA-Code": mfaCode } : {}),
        },
      },
    );
    await Promise.all([get().fetchModelRegistry(), get().fetchChampionChallenger()]);
  },

  startTrading: async (payload) => {
    await api.post("/control/trading/start", payload, {
      headers: {
        "Idempotency-Key": buildIdempotencyKey("trading-start"),
        ...(payload.mfa_code ? { "X-MFA-Code": payload.mfa_code } : {}),
      },
    });
    await get().fetchTradingStatus();
  },

  stopTrading: async () => {
    await api.post(
      "/control/trading/stop",
      {},
      {
        headers: {
          "Idempotency-Key": buildIdempotencyKey("trading-stop"),
        },
      },
    );
    await get().fetchTradingStatus();
  },

  restartTrading: async (payload) => {
    await api.post("/control/trading/restart", payload, {
      headers: {
        "Idempotency-Key": buildIdempotencyKey("trading-restart"),
        ...(payload.mfa_code ? { "X-MFA-Code": payload.mfa_code } : {}),
      },
    });
    await get().fetchTradingStatus();
  },

  activateKillSwitch: async (reason: string, mfaCode?: string) => {
    await api.post(
      "/control/risk/kill-switch/activate",
      { reason, mfa_code: mfaCode ?? null },
      {
        headers: {
          "Idempotency-Key": buildIdempotencyKey("kill-switch-activate"),
          ...(mfaCode ? { "X-MFA-Code": mfaCode } : {}),
        },
      },
    );
  },

  resetKillSwitch: async (authorizedBy: string, force = false, overrideCode?: string, mfaCode?: string) => {
    await api.post(
      "/control/risk/kill-switch/reset",
      {
        authorized_by: authorizedBy,
        force,
        override_code: overrideCode ?? null,
        mfa_code: mfaCode ?? null,
      },
      {
        headers: {
          "Idempotency-Key": buildIdempotencyKey("kill-switch-reset"),
          ...(mfaCode ? { "X-MFA-Code": mfaCode } : {}),
        },
      },
    );
  },

  createJob: async (command: string, args: string[]) => {
    await api.post(
      "/control/jobs",
      { command, args },
      {
        headers: {
          "Idempotency-Key": buildIdempotencyKey("jobs-create"),
        },
      },
    );
    await get().fetchJobs();
  },

  cancelJob: async (jobId: string) => {
    await api.post(
      `/control/jobs/${jobId}/cancel`,
      {},
      {
        headers: {
          "Idempotency-Key": buildIdempotencyKey("jobs-cancel"),
        },
      },
    );
    await get().fetchJobs();
  },

  executeRunbookAction: async (action: string, mfaCode?: string) => {
    await api.post(
      "/sre/runbooks/execute",
      {
        action,
        mfa_code: mfaCode ?? null,
      },
      {
        headers: {
          "Idempotency-Key": buildIdempotencyKey("runbook-action"),
          ...(mfaCode ? { "X-MFA-Code": mfaCode } : {}),
        },
      },
    );
    await get().fetchJobs();
  },

  connectLiveChannels: () => {
    const token = get().token;
    if (!token) {
      return;
    }
    get().disconnectLiveChannels();

    const updateWs = (channel: keyof WsState, connected: boolean) => {
      set((state) => ({ ws: { ...state.ws, [channel]: connected } }));
    };

    connectSocket(
      "portfolio",
      (payload) => {
        if (payload?.type === "snapshot" || payload?.type === "portfolio_update") {
          set({ portfolio: payload.data });
        }
      },
      (connected) => updateWs("portfolio", connected),
    );

    connectSocket(
      "orders",
      (payload) => {
        if (payload?.type === "snapshot" && Array.isArray(payload.data)) {
          set({ orders: payload.data });
          return;
        }
        if (payload?.type === "order_update" && payload.data?.order_id) {
          set((state) => {
            const idx = state.orders.findIndex((x) => x.order_id === payload.data.order_id);
            if (idx < 0) {
              return { orders: [payload.data, ...state.orders].slice(0, 250) };
            }
            const next = [...state.orders];
            next[idx] = { ...next[idx], ...payload.data };
            return { orders: next };
          });
        }
      },
      (connected) => updateWs("orders", connected),
    );

    connectSocket(
      "signals",
      (payload) => {
        if (payload?.type === "signal" && payload.data) {
          // signals are shown on overview via explainability and alerts; no-op for now
          return;
        }
      },
      (connected) => updateWs("signals", connected),
    );

    connectSocket(
      "alerts",
      (payload) => {
        if (payload?.type === "snapshot" && Array.isArray(payload.data)) {
          set({ alerts: payload.data });
          return;
        }
        if (payload?.type === "alert" && payload.data) {
          set((state) => ({
            alerts: [payload.data, ...state.alerts].slice(0, 200),
          }));
        }
      },
      (connected) => updateWs("alerts", connected),
    );
  },

  disconnectLiveChannels: () => {
    (Object.keys(sockets) as Array<keyof typeof sockets>).forEach((key) => {
      if (sockets[key]) {
        sockets[key]?.close();
        sockets[key] = null;
      }
    });
    set({ ws: { portfolio: false, orders: false, signals: false, alerts: false } });
  },
}));
