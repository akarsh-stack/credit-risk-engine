// lib/api.ts
// API client for the Credit Risk Engine FastAPI backend

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface LoanFeatures {
  fico_score: number;
  annual_income: number;
  dti_ratio: number;
  loan_amount: number;
  loan_term: number;
  interest_rate: number;
  employment_length_years: number | null;
  home_ownership: string;
  loan_purpose: string;
  credit_history_length_years: number;
  num_delinquencies_2yrs: number;
  revolving_utilization_pct: number | null;
  loan_grade: string;
  verification_status: string;
  num_open_accounts: number;
  num_derogatory_marks: number;
}

export interface SHAPFeature {
  feature_name: string;
  feature_value: number;
  shap_value: number;
  direction: 'increases_risk' | 'decreases_risk';
}

export interface PredictionResponse {
  default_probability: number;
  risk_grade: string;
  risk_grade_label: string;
  top_shap_features: SHAPFeature[];
  model_version: string;
}

export interface PortfolioVaRResponse {
  n_loans: number;
  total_exposure: number;
  mean_pd: number;
  expected_loss: number;
  var_95: number;
  var_99: number;
  es_95: number;
  loss_distribution: number[];
  lgd: number;
  n_scenarios: number;
}

export async function predictLoan(loan: LoanFeatures): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(loan),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

export async function getPortfolioVaR(): Promise<PortfolioVaRResponse> {
  const response = await fetch(`${API_BASE}/portfolio-var`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ loans: null, n_scenarios: 10000, lgd: 0.6 }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

export interface StressTestRequest {
  unemployment_shock_pct: number;
  interest_rate_bump_bps: number;
  collateral_haircut_pct: number;
}

export interface StressTestResponse {
  baseline_expected_loss: number;
  stressed_expected_loss: number;
  baseline_var_95: number;
  stressed_var_95: number;
  baseline_var_99: number;
  stressed_var_99: number;
  loss_increase_pct: number;
  loss_distribution_baseline: number[];
  loss_distribution_stressed: number[];
}

export interface EntityProfile {
  entity_id: string;
  name: string;
  sector: string;
  total_exposure: number;
  pd_score: number;
  risk_grade: string;
  fico_score: number;
  dti_ratio: number;
  num_delinquencies: number;
  open_lines: number;
  status: string;
  recent_events: string[];
}

export interface LogEntry {
  timestamp: string;
  level: string;
  module: string;
  message: string;
}

export async function runStressTest(params: StressTestRequest): Promise<StressTestResponse> {
  const response = await fetch(`${API_BASE}/stress-test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

export async function getEntities(): Promise<EntityProfile[]> {
  const response = await fetch(`${API_BASE}/entities`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

export async function getLogs(): Promise<LogEntry[]> {
  const response = await fetch(`${API_BASE}/logs`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

