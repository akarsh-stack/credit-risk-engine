"use client";

import React, { useState, useEffect } from 'react';
import { predictLoan, getPortfolioVaR, LoanFeatures, PredictionResponse, PortfolioVaRResponse } from '@/lib/api';
import { ProbabilityGauge } from '@/components/ui/gauge';
import { WaterfallChart } from '@/components/ui/waterfall';
import { Activity, ShieldAlert, BarChart3, TrendingDown, DollarSign } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<'single' | 'portfolio'>('single');

  // Single Loan State
  const [loanInput, setLoanInput] = useState<LoanFeatures>({
    fico_score: 720,
    annual_income: 75000,
    dti_ratio: 18.5,
    loan_amount: 15000,
    loan_term: 36,
    interest_rate: 11.5,
    employment_length_years: 5,
    home_ownership: "MORTGAGE",
    loan_purpose: "debt_consolidation",
    credit_history_length_years: 12,
    num_delinquencies_2yrs: 0,
    revolving_utilization_pct: 28.0,
    loan_grade: "B",
    verification_status: "Verified",
    num_open_accounts: 8,
    num_derogatory_marks: 0
  });
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loadingPred, setLoadingPred] = useState(false);
  const [predError, setPredError] = useState('');

  // Portfolio State
  const [portfolioVar, setPortfolioVar] = useState<PortfolioVaRResponse | null>(null);
  const [loadingVar, setLoadingVar] = useState(false);
  const [varError, setVarError] = useState('');

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoadingPred(true);
    setPredError('');
    try {
      const result = await predictLoan(loanInput);
      setPrediction(result);
    } catch (err: any) {
      setPredError(err.message);
    } finally {
      setLoadingPred(false);
    }
  };

  const loadPortfolio = async () => {
    if (portfolioVar) return;
    setLoadingVar(true);
    setVarError('');
    try {
      const result = await getPortfolioVaR();
      setPortfolioVar(result);
    } catch (err: any) {
      setVarError(err.message);
    } finally {
      setLoadingVar(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'portfolio') {
      loadPortfolio();
    }
  }, [activeTab]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    let parsedValue: any = value;
    if (type === 'number') {
      parsedValue = value === '' ? null : Number(value);
    }
    setLoanInput({ ...loanInput, [name]: parsedValue });
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-6 font-sans">
      <header className="mb-8 border-b border-slate-800 pb-4 flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-2 tracking-tight">
            <Activity className="text-emerald-500" /> 
            Credit Risk Engine
          </h1>
          <p className="text-slate-400 mt-1">XGBoost & Monte Carlo Risk Analytics</p>
        </div>
        
        <div className="flex gap-2">
          <button 
            onClick={() => setActiveTab('single')}
            className={`px-4 py-2 rounded-md transition-colors ${activeTab === 'single' ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}`}
          >
            Single Underwriting
          </button>
          <button 
            onClick={() => setActiveTab('portfolio')}
            className={`px-4 py-2 rounded-md transition-colors ${activeTab === 'portfolio' ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}`}
          >
            Portfolio VaR
          </button>
        </div>
      </header>

      {activeTab === 'single' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Form */}
          <div className="lg:col-span-5 bg-slate-800/50 p-6 rounded-xl border border-slate-700">
            <h2 className="text-xl font-semibold mb-6 flex items-center gap-2 border-b border-slate-700 pb-2">
              <ShieldAlert className="text-indigo-400 w-5 h-5" />
              Loan Application
            </h2>
            <form onSubmit={handlePredict} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">FICO Score</label>
                  <input type="number" name="fico_score" value={loanInput.fico_score} onChange={handleInputChange} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-indigo-500 outline-none" required />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Annual Income ($)</label>
                  <input type="number" name="annual_income" value={loanInput.annual_income} onChange={handleInputChange} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-indigo-500 outline-none" required />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">DTI Ratio (%)</label>
                  <input type="number" name="dti_ratio" step="0.1" value={loanInput.dti_ratio} onChange={handleInputChange} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-indigo-500 outline-none" required />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Loan Amount ($)</label>
                  <input type="number" name="loan_amount" value={loanInput.loan_amount} onChange={handleInputChange} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-indigo-500 outline-none" required />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Loan Term (mo)</label>
                  <select name="loan_term" value={loanInput.loan_term} onChange={handleInputChange} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-indigo-500 outline-none">
                    <option value={36}>36</option>
                    <option value={60}>60</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Interest Rate (%)</label>
                  <input type="number" name="interest_rate" step="0.1" value={loanInput.interest_rate} onChange={handleInputChange} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-indigo-500 outline-none" required />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Home Ownership</label>
                  <select name="home_ownership" value={loanInput.home_ownership} onChange={handleInputChange} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-indigo-500 outline-none">
                    <option value="MORTGAGE">MORTGAGE</option>
                    <option value="RENT">RENT</option>
                    <option value="OWN">OWN</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Loan Grade</label>
                  <select name="loan_grade" value={loanInput.loan_grade} onChange={handleInputChange} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-indigo-500 outline-none">
                    {['A', 'B', 'C', 'D', 'E', 'F', 'G'].map(g => <option key={g} value={g}>{g}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Revolving Util (%)</label>
                  <input type="number" name="revolving_utilization_pct" step="0.1" value={loanInput.revolving_utilization_pct || ''} onChange={handleInputChange} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-indigo-500 outline-none" />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Delinquencies (2y)</label>
                  <input type="number" name="num_delinquencies_2yrs" value={loanInput.num_delinquencies_2yrs} onChange={handleInputChange} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-indigo-500 outline-none" required />
                </div>
              </div>
              
              <button 
                type="submit" 
                disabled={loadingPred}
                className="w-full mt-6 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3 px-4 rounded-md transition-colors shadow-lg shadow-indigo-500/20 disabled:opacity-50"
              >
                {loadingPred ? 'Running Model...' : 'Score Application'}
              </button>
            </form>
            {predError && <div className="mt-4 p-3 bg-red-900/30 border border-red-800 text-red-200 rounded">{predError}</div>}
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-7 flex flex-col gap-6">
            {!prediction && !loadingPred && (
              <div className="flex-1 flex flex-col items-center justify-center border-2 border-dashed border-slate-800 rounded-xl text-slate-500 bg-slate-900/50">
                <ShieldAlert className="w-12 h-12 mb-3 opacity-20" />
                <p>Submit an application to view risk profile</p>
              </div>
            )}
            
            {prediction && (
              <>
                <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col md:flex-row items-center gap-8 justify-around">
                  <ProbabilityGauge probability={prediction.default_probability} grade={prediction.risk_grade} />
                  <div className="flex flex-col">
                    <span className="text-slate-400 uppercase tracking-wider text-sm font-semibold mb-1">Model Grade</span>
                    <div className="flex items-end gap-3">
                      <span className={`text-6xl font-black ${
                        ['A','B'].includes(prediction.risk_grade) ? 'text-emerald-500' :
                        prediction.risk_grade === 'C' ? 'text-amber-500' : 'text-red-500'
                      }`}>
                        {prediction.risk_grade}
                      </span>
                      <span className="text-xl text-slate-300 pb-2">{prediction.risk_grade_label}</span>
                    </div>
                  </div>
                </div>

                <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex-1">
                  <h3 className="text-lg font-semibold mb-4 border-b border-slate-700 pb-2">SHAP Explainability (Top Drivers)</h3>
                  <WaterfallChart features={prediction.top_shap_features} />
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {activeTab === 'portfolio' && (
        <div className="space-y-6">
          {loadingVar && <div className="p-8 text-center text-slate-400 animate-pulse">Loading Monte Carlo Simulation...</div>}
          {varError && <div className="p-4 bg-red-900/30 border border-red-800 text-red-200 rounded">{varError}</div>}
          
          {portfolioVar && (
            <>
              {/* Stat Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700">
                  <div className="text-slate-400 text-sm uppercase tracking-wider font-semibold mb-1">Total Exposure</div>
                  <div className="text-2xl font-bold text-white">${(portfolioVar.total_exposure / 1e6).toFixed(2)}M</div>
                  <div className="text-xs text-slate-500 mt-1">{portfolioVar.n_loans} loans</div>
                </div>
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700">
                  <div className="text-slate-400 text-sm uppercase tracking-wider font-semibold mb-1">Expected Loss</div>
                  <div className="text-2xl font-bold text-emerald-400">${(portfolioVar.expected_loss / 1e6).toFixed(2)}M</div>
                  <div className="text-xs text-slate-500 mt-1">{(portfolioVar.expected_loss / portfolioVar.total_exposure * 100).toFixed(1)}% of exposure</div>
                </div>
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700">
                  <div className="text-slate-400 text-sm uppercase tracking-wider font-semibold mb-1">VaR (99%)</div>
                  <div className="text-2xl font-bold text-red-400">${(portfolioVar.var_99 / 1e6).toFixed(2)}M</div>
                  <div className="text-xs text-slate-500 mt-1">1-in-100 downside</div>
                </div>
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700">
                  <div className="text-slate-400 text-sm uppercase tracking-wider font-semibold mb-1">Expected Shortfall</div>
                  <div className="text-2xl font-bold text-purple-400">${(portfolioVar.es_95 / 1e6).toFixed(2)}M</div>
                  <div className="text-xs text-slate-500 mt-1">CVaR (95%)</div>
                </div>
              </div>

              {/* Histogram */}
              <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700">
                <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
                  <BarChart3 className="text-indigo-400" />
                  Portfolio Loss Distribution (Monte Carlo, N={portfolioVar.n_scenarios.toLocaleString()})
                </h2>
                <div className="h-[400px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={(() => {
                        // Bin the sample data for Recharts
                        const data = portfolioVar.loss_distribution;
                        const min = Math.min(...data);
                        const max = Math.max(...data);
                        const bins = 50;
                        const step = (max - min) / bins;
                        const binned = Array(bins).fill(0);
                        data.forEach(val => {
                          let idx = Math.floor((val - min) / step);
                          if (idx >= bins) idx = bins - 1;
                          binned[idx]++;
                        });
                        return binned.map((count, i) => ({
                          loss: (min + i * step + step/2) / 1e6, // In Millions
                          freq: count
                        }));
                      })()}
                      margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                      <XAxis 
                        dataKey="loss" 
                        stroke="#94a3b8" 
                        tickFormatter={(val) => `$${val.toFixed(1)}M`}
                        type="number"
                        domain={['dataMin', 'dataMax']}
                      />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }}
                        formatter={(val: number) => [val, 'Scenarios']}
                        labelFormatter={(val: number) => `Loss: $${val.toFixed(2)}M`}
                      />
                      <Bar dataKey="freq" fill="#3b82f6" opacity={0.8} name="Frequency" />
                      <ReferenceLine x={portfolioVar.expected_loss/1e6} stroke="#10b981" strokeWidth={2} label={{ position: 'top', value: 'Expected', fill: '#10b981', fontSize: 12 }} />
                      <ReferenceLine x={portfolioVar.var_95/1e6} stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" label={{ position: 'top', value: 'VaR95', fill: '#f59e0b', fontSize: 12 }} />
                      <ReferenceLine x={portfolioVar.var_99/1e6} stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" label={{ position: 'insideTopLeft', value: 'VaR99', fill: '#ef4444', fontSize: 12 }} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
