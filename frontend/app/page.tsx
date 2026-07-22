"use client";

import React, { useState, useEffect } from 'react';
import {
  predictLoan, getPortfolioVaR, runStressTest, getEntities, getLogs,
  LoanFeatures, PredictionResponse, PortfolioVaRResponse,
  StressTestRequest, StressTestResponse, EntityProfile, LogEntry
} from '@/lib/api';
import { WaterfallChart } from '@/components/ui/waterfall';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';

export default function Dashboard() {
  const [activeNav, setActiveNav] = useState<'dashboard' | 'portfolio' | 'entity' | 'simulation' | 'logs'>('portfolio');
  const [activeTopNav, setActiveTopNav] = useState<'exposure' | 'var' | 'limits' | 'alerts'>('var');
  const [isExecuting, setIsExecuting] = useState(false);

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
  const [loadingVar, setLoadingVar] = useState(true);

  // Entity View State
  const [entities, setEntities] = useState<EntityProfile[]>([]);
  const [selectedEntity, setSelectedEntity] = useState<EntityProfile | null>(null);
  const [entitySearch, setEntitySearch] = useState('');

  // Stress Test State
  const [stressInput, setStressInput] = useState<StressTestRequest>({
    unemployment_shock_pct: 2.5,
    interest_rate_bump_bps: 150,
    collateral_haircut_pct: 15.0
  });
  const [stressResult, setStressResult] = useState<StressTestResponse | null>(null);
  const [loadingStress, setLoadingStress] = useState(false);

  // Logs State
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logFilterLevel, setLogFilterLevel] = useState<string>('ALL');
  const [logSearch, setLogSearch] = useState('');

  const loadPortfolio = async () => {
    try {
      setLoadingVar(true);
      const result = await getPortfolioVaR();
      setPortfolioVar(result);
    } catch (err: any) {
      console.error("Failed to load portfolio var:", err);
    } finally {
      setLoadingVar(false);
    }
  };

  const loadEntities = async () => {
    try {
      const data = await getEntities();
      setEntities(data);
      if (data.length > 0 && !selectedEntity) {
        setSelectedEntity(data[0]);
      }
    } catch (err) {
      console.error("Failed to load entities:", err);
    }
  };

  const loadLogs = async () => {
    try {
      const data = await getLogs();
      setLogs(data);
    } catch (err) {
      console.error("Failed to load logs:", err);
    }
  };

  const handleRunStressTest = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoadingStress(true);
    try {
      const res = await runStressTest(stressInput);
      setStressResult(res);
    } catch (err) {
      console.error("Stress test failed:", err);
    } finally {
      setLoadingStress(false);
    }
  };

  useEffect(() => {
    loadPortfolio();
    loadEntities();
    loadLogs();
  }, []);

  const handlePredict = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
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

  const handleExecuteRun = () => {
    setIsExecuting(true);
    if (activeNav === 'dashboard') {
      handlePredict();
    } else if (activeNav === 'portfolio') {
      loadPortfolio();
    } else if (activeNav === 'simulation') {
      handleRunStressTest();
    } else if (activeNav === 'logs') {
      loadLogs();
    } else if (activeNav === 'entity') {
      loadEntities();
    }
    setTimeout(() => setIsExecuting(false), 1000);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    let parsedValue: any = value;
    if (type === 'number') {
      parsedValue = value === '' ? null : Number(value);
    }
    setLoanInput({ ...loanInput, [name]: parsedValue });
  };

  const loadEntityIntoUnderwriter = (entity: EntityProfile) => {
    setLoanInput({
      fico_score: entity.fico_score,
      annual_income: Math.round(entity.total_exposure * 0.45),
      dti_ratio: entity.dti_ratio,
      loan_amount: Math.round(entity.total_exposure * 0.3),
      loan_term: 36,
      interest_rate: entity.risk_grade === 'A' ? 7.5 : entity.risk_grade === 'B' ? 11.2 : 16.8,
      employment_length_years: 8,
      home_ownership: "MORTGAGE",
      loan_purpose: "debt_consolidation",
      credit_history_length_years: 15,
      num_delinquencies_2yrs: entity.num_delinquencies,
      revolving_utilization_pct: entity.dti_ratio * 1.5,
      loan_grade: entity.risk_grade,
      verification_status: "Verified",
      num_open_accounts: entity.open_lines,
      num_derogatory_marks: entity.num_delinquencies > 0 ? 1 : 0
    });
    setActiveNav('dashboard');
  };

  const totalExposure = portfolioVar ? portfolioVar.total_exposure : 0;
  const expectedLoss = portfolioVar ? portfolioVar.expected_loss : 0;
  const var95 = portfolioVar ? portfolioVar.var_95 : 0;
  const var99 = portfolioVar ? portfolioVar.var_99 : 0;
  const es95 = portfolioVar ? portfolioVar.es_95 : 0;
  const probDef = portfolioVar && totalExposure > 0 ? (expectedLoss / totalExposure) : 0;

  const formatCurrency = (val: number) => {
    if (val >= 1e6) {
      return `$${(val / 1e6).toFixed(2)}M`;
    }
    return `$${val.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  };

  const navItems = [
    { id: 'dashboard', icon: 'dashboard', label: 'DASHBOARD' },
    { id: 'portfolio', icon: 'account_balance', label: 'PORTFOLIO' },
    { id: 'entity', icon: 'business_center', label: 'ENTITY-VIEW' },
    { id: 'simulation', icon: 'analytics', label: 'SIMULATION' },
    { id: 'logs', icon: 'receipt_long', label: 'LOGS' },
  ];

  const topNavItems = ['EXPOSURE', 'VAR', 'LIMITS', 'ALERTS'];

  const filteredEntities = entities.filter(ent => 
    ent.name.toLowerCase().includes(entitySearch.toLowerCase()) ||
    ent.entity_id.toLowerCase().includes(entitySearch.toLowerCase()) ||
    ent.sector.toLowerCase().includes(entitySearch.toLowerCase())
  );

  const filteredLogs = logs.filter(log => {
    const matchesLevel = logFilterLevel === 'ALL' || log.level === logFilterLevel;
    const matchesSearch = log.message.toLowerCase().includes(logSearch.toLowerCase()) ||
                          log.module.toLowerCase().includes(logSearch.toLowerCase());
    return matchesLevel && matchesSearch;
  });

  return (
    <div className="flex h-screen w-screen overflow-hidden text-[#E8E6DF] font-sans bg-background">
      {/* SideNavBar */}
      <aside className="hidden md:flex flex-col h-full w-64 bg-surface-container-low border-r border-outline-variant shrink-0">
        <div className="p-container-margin border-b border-outline-variant">
          <h1 className="font-headline-sm text-headline-sm text-primary tracking-tighter">CREDIT-RISK-ENGINE</h1>
          <div className="mt-stack-compact flex flex-col">
            <span className="font-label-caps text-label-caps text-on-surface-variant">OP-042</span>
            <span className="font-body-sm text-body-sm text-secondary">SECTOR-ALPHA</span>
          </div>
        </div>
        
        <nav className="flex-1 overflow-y-auto scrollbar-terminal py-stack-default">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setActiveNav(item.id as any)}
              className={`w-full flex items-center px-4 py-3 transition-all duration-150 ${
                activeNav === item.id 
                  ? 'text-primary bg-surface-container-highest border-l-2 border-primary' 
                  : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container-highest border-l-2 border-transparent'
              }`}
            >
              <span className="material-symbols-outlined mr-3 text-[18px]">{item.icon}</span>
              <span className="font-label-caps text-label-caps">{item.label}</span>
            </button>
          ))}
        </nav>
        
        <div className="p-container-margin border-t border-outline-variant space-y-4">
          <button 
            onClick={handleExecuteRun}
            className={`w-full py-2 bg-primary-container text-on-primary-fixed font-label-caps text-label-caps font-bold hover:bg-primary transition-colors flex items-center justify-center gap-2 ${isExecuting ? 'animate-pulse' : ''}`}
          >
            {isExecuting ? (
              <><span className="material-symbols-outlined text-[14px] animate-spin">refresh</span> EXECUTING...</>
            ) : 'EXECUTE-RUN'}
          </button>
          <div className="flex flex-col space-y-2">
            <button className="flex items-center text-on-surface-variant font-label-caps text-label-caps hover:text-primary transition-colors">
              <span className="material-symbols-outlined mr-2 text-[14px]">help</span> DOCS
            </button>
            <button className="flex items-center text-on-surface-variant font-label-caps text-label-caps hover:text-primary transition-colors">
              <span className="material-symbols-outlined mr-2 text-[14px]">contact_support</span> SUPPORT
            </button>
          </div>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="flex-1 flex flex-col min-w-0 relative overflow-hidden">
        {/* Ticker Tape */}
        <div className="h-6 w-full bg-black border-b border-outline-variant flex items-center overflow-hidden z-10 shrink-0">
          <div className="ticker-content flex space-x-12 px-container-margin items-center">
            <div className="flex items-center space-x-2">
              <span className="text-[10px] text-primary-container font-bold">TOTAL EXPOSURE:</span>
              <span className="text-[10px] text-white">{formatCurrency(totalExposure)}</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] text-primary-container font-bold">EXPECTED LOSS:</span>
              <span className="text-[10px] text-white font-mono-data">{formatCurrency(expectedLoss)}</span>
              <span className="text-[10px] text-secondary font-mono-data">[{(probDef * 100).toFixed(1)}%]</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] text-primary-container font-bold">VaR95:</span>
              <span className="text-[10px] text-white">{formatCurrency(var95)}</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] text-primary-container font-bold">VaR99:</span>
              <span className="text-[10px] text-white">{formatCurrency(var99)}</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] text-primary-container font-bold">SHARPE:</span>
              <span className="text-[10px] text-secondary">2.41</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] text-primary-container font-bold">TOTAL EXPOSURE:</span>
              <span className="text-[10px] text-white">{formatCurrency(totalExposure)}</span>
            </div>
          </div>
        </div>

        {/* TopAppBar */}
        <header className="flex justify-between items-center w-full px-container-margin py-stack-compact border-b border-outline-variant bg-background shrink-0">
          <div className="flex items-center space-x-8">
            <nav className="flex space-x-6">
              {topNavItems.map(item => (
                <button
                  key={item}
                  onClick={() => setActiveTopNav(item.toLowerCase() as any)}
                  className={`font-label-caps text-label-caps transition-colors pb-1 ${
                    activeTopNav === item.toLowerCase()
                      ? 'text-primary border-b-2 border-primary font-bold'
                      : 'text-on-surface-variant hover:text-primary border-b-2 border-transparent'
                  }`}
                >
                  {item}
                </button>
              ))}
            </nav>
          </div>
          
          <div className="flex items-center space-x-6">
            <div className="relative">
              <span className="absolute left-2 top-1/2 -translate-y-1/2 material-symbols-outlined text-on-surface-variant text-[16px]">search</span>
              <input 
                type="text" 
                placeholder="CMD_SEARCH" 
                value={entitySearch}
                onChange={(e) => setEntitySearch(e.target.value)}
                className="bg-surface-container-low border border-outline-variant pl-8 pr-4 py-1 text-[11px] font-mono-data text-on-surface focus:outline-none focus:border-primary w-48 transition-colors"
              />
            </div>
            <div className="flex items-center space-x-4">
              <button className="material-symbols-outlined text-on-surface-variant text-[18px] hover:text-primary transition-colors">settings</button>
              <button className="material-symbols-outlined text-on-surface-variant text-[18px] hover:text-primary transition-colors">terminal</button>
              <button className="material-symbols-outlined text-[#ef4444] text-[18px] hover:text-white transition-colors">power_settings_new</button>
            </div>
          </div>
        </header>

        {/* View Controls & Breadcrumb */}
        <div className="px-container-margin py-4 border-b border-outline-variant bg-surface-container-low flex justify-between items-center shrink-0">
          <div className="flex items-center font-mono-data text-mono-data tracking-tight">
            <span className="text-on-surface-variant">CREDIT-RISK-ENGINE</span>
            <span className="mx-2 text-primary-container">&gt;</span>
            <span className="text-white font-bold">{activeNav.toUpperCase()}</span>
            <span className="ml-1 w-[8px] h-[14px] bg-primary-container cursor-pulse inline-block align-middle"></span>
          </div>
          <div className="flex font-label-caps text-label-caps">
            <button 
              onClick={() => setActiveNav('dashboard')}
              className={`px-4 py-1.5 border transition-colors ${activeNav === 'dashboard' ? 'border-primary bg-primary/10 text-primary font-bold' : 'border-outline-variant text-on-surface-variant hover:border-primary'}`}
            >
              SINGLE UNDERWRITING
            </button>
            <button 
              onClick={() => setActiveNav('portfolio')}
              className={`px-4 py-1.5 border ml-[-1px] transition-colors ${activeNav === 'portfolio' ? 'border-primary bg-primary/10 text-primary font-bold z-10' : 'border-outline-variant text-on-surface-variant hover:border-primary'}`}
            >
              PORTFOLIO VAR
            </button>
          </div>
        </div>

        {/* Dynamic Content Based on activeNav */}
        <div className="flex-1 overflow-y-auto p-gutter scrollbar-terminal grid grid-cols-12 gap-gutter bg-background">
          
          {/* PORTFOLIO VAR TAB */}
          {activeNav === 'portfolio' && (
            <>
              <div className="col-span-12 md:col-span-3 bg-surface-container p-4 flex flex-col justify-between hairline h-32">
                <span className="font-label-caps text-label-caps text-on-surface-variant">TOTAL-EXPOSURE</span>
                <div className="mt-2">
                  <span className="font-display-lg text-display-lg text-white">{formatCurrency(totalExposure)}</span>
                  <div className="flex items-center mt-1 text-on-surface-variant">
                    <span className="font-mono-data text-[10px]">LOB_SECURED: {formatCurrency(totalExposure * 0.53)}</span>
                  </div>
                </div>
              </div>
              
              <div className="col-span-12 md:col-span-3 bg-surface-container p-4 flex flex-col justify-between hairline h-32">
                <span className="font-label-caps text-label-caps text-on-surface-variant">EXPECTED-LOSS</span>
                <div className="mt-2">
                  <span className="font-display-lg text-display-lg text-secondary">{formatCurrency(expectedLoss)}</span>
                  <div className="font-mono-data text-[10px] text-on-surface-variant mt-1">PROB_DEF: {probDef.toFixed(3)}</div>
                </div>
              </div>
              
              <div className="col-span-12 md:col-span-3 bg-surface-container p-4 flex flex-col justify-between hairline h-32">
                <span className="font-label-caps text-label-caps text-on-surface-variant">VAR (95%)</span>
                <div className="mt-2">
                  <span className="font-display-lg text-display-lg text-primary-container">{formatCurrency(var95)}</span>
                  <div className="font-mono-data text-[10px] text-on-surface-variant mt-1">CI: 95.00000000</div>
                </div>
              </div>
              
              <div className="col-span-12 md:col-span-3 bg-surface-container p-4 flex flex-col justify-between hairline h-32">
                <span className="font-label-caps text-label-caps text-on-surface-variant">CVAR (95%)</span>
                <div className="mt-2">
                  <span className="font-display-lg text-display-lg text-primary">{formatCurrency(es95)}</span>
                  <div className="font-mono-data text-[10px] text-on-surface-variant mt-1">EXPECTED_SHORTFALL</div>
                </div>
              </div>

              <div className="col-span-12 md:col-span-9 bg-surface-container p-6 hairline flex flex-col h-[500px]">
                <div className="flex justify-between items-center mb-6 shrink-0">
                  <div className="flex items-center space-x-2">
                    <span className="material-symbols-outlined text-secondary text-[20px]">bar_chart</span>
                    <h2 className="font-headline-sm text-headline-sm uppercase tracking-widest text-white">Portfolio Loss Distribution</h2>
                  </div>
                  <div className="font-mono-data text-on-surface-variant text-[10px]">
                      MONTE_CARLO: N={portfolioVar ? portfolioVar.n_scenarios.toLocaleString() : "..."} | STATUS: {loadingVar ? 'CALCULATING' : 'STABLE'}
                  </div>
                </div>
                
                <div className="flex-1 w-full relative min-h-[350px]">
                  {loadingVar && !portfolioVar ? (
                    <div className="absolute inset-0 flex items-center justify-center text-secondary font-mono-data animate-pulse">
                      LOADING_SIMULATION_DATA...
                    </div>
                  ) : portfolioVar ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={(() => {
                          const data = portfolioVar.loss_distribution;
                          if (!data || data.length === 0) return [];
                          const min = Math.min(...data);
                          const max = Math.max(...data);
                          const bins = 40;
                          const step = (max - min) / bins;
                          const binned = Array(bins).fill(0);
                          data.forEach(val => {
                            let idx = Math.floor((val - min) / step);
                            if (idx >= bins) idx = bins - 1;
                            binned[idx]++;
                          });
                          return binned.map((count, i) => ({
                            loss: (min + i * step + step/2) / 1e6,
                            freq: count
                          }));
                        })()}
                        margin={{ top: 20, right: 30, left: 10, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1D" vertical={false} />
                        <XAxis 
                          dataKey="loss" 
                          stroke="#6B6B63" 
                          tickFormatter={(val) => `$${val.toFixed(1)}M`}
                          type="number"
                          domain={['dataMin', 'dataMax']}
                          tick={{ fill: '#d7c4ac', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                          axisLine={{ stroke: '#1F1F1D' }}
                        />
                        <YAxis 
                          stroke="#6B6B63" 
                          tick={{ fill: '#d7c4ac', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                          axisLine={{ stroke: '#1F1F1D' }}
                        />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#131312', border: '1px solid #1F1F1D', borderRadius: 0 }}
                          itemStyle={{ color: '#47f5db', fontFamily: 'JetBrains Mono', fontSize: 12 }}
                          labelStyle={{ color: '#d7c4ac', fontFamily: 'JetBrains Mono', fontSize: 10 }}
                          formatter={(val: number) => [val, 'FREQ']}
                          labelFormatter={(val: number) => `LOSS: $${val.toFixed(2)}M`}
                          cursor={{ fill: '#1c1c1a' }}
                        />
                        <Bar dataKey="freq" fill="#47f5db" opacity={0.8} />
                        <ReferenceLine x={portfolioVar.expected_loss/1e6} stroke="#e5e2de" strokeWidth={1} label={{ position: 'top', value: 'EXPECTED', fill: '#e5e2de', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                        <ReferenceLine x={portfolioVar.var_95/1e6} stroke="#ffb000" strokeWidth={1} strokeDasharray="4 4" label={{ position: 'top', value: 'VaR95', fill: '#ffb000', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                        <ReferenceLine x={portfolioVar.var_99/1e6} stroke="#ffd0cd" strokeWidth={1} strokeDasharray="4 4" label={{ position: 'insideTopLeft', value: 'VaR99', fill: '#ffd0cd', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : null}
                </div>
              </div>

              <div className="col-span-12 md:col-span-3 bg-surface-container hairline overflow-hidden flex flex-col h-[500px]">
                <div className="p-3 border-b border-outline-variant bg-surface-container-low flex justify-between items-center shrink-0">
                  <span className="font-label-caps text-label-caps text-on-surface-variant">RISK_FEED.LOG</span>
                  <span className="material-symbols-outlined text-[14px] text-secondary">sensors</span>
                </div>
                
                <div className="flex-1 overflow-y-auto font-mono-data text-[11px] p-3 space-y-3">
                  <div className="border-l-2 border-secondary pl-2 py-1 bg-surface-container-low">
                    <span className="text-on-surface-variant">[14:22:01]</span> <span className="text-secondary">CALC_COMPLETE:</span> Monte Carlo engine finished {portfolioVar ? portfolioVar.n_scenarios.toLocaleString() : "..."} iterations.
                  </div>
                  <div className="border-l-2 border-primary-container pl-2 py-1">
                    <span className="text-on-surface-variant">[14:21:55]</span> <span className="text-primary-container">ALERT_LIMIT:</span> Exposure in Sector-Alpha approaching 95% of limit.
                  </div>
                  <div className="border-l-2 border-outline-variant pl-2 py-1">
                    <span className="text-on-surface-variant">[14:20:12]</span> <span className="text-on-surface">SYS_UPDATE:</span> Asset correlation matrix updated for Q4 projection.
                  </div>
                  <div className="border-l-2 border-outline-variant pl-2 py-1">
                    <span className="text-on-surface-variant">[14:18:45]</span> <span className="text-on-surface">DATA_INGEST:</span> {portfolioVar ? portfolioVar.n_loans : "..."} loans processed from node DX-9.
                  </div>
                </div>
                
                <div className="p-3 border-t border-outline-variant mt-auto shrink-0">
                  <span className="font-label-caps text-label-caps text-on-surface-variant block mb-2">QUICK_ACTIONS</span>
                  <div className="grid grid-cols-2 gap-2">
                    <button onClick={loadPortfolio} className="p-2 border border-outline-variant text-[10px] text-on-surface font-bold hover:bg-surface-container-highest transition-colors">RECALCULATE</button>
                    <button onClick={() => setActiveNav('logs')} className="p-2 border border-outline-variant text-[10px] text-on-surface font-bold hover:bg-surface-container-highest transition-colors">VIEW_ALL_LOGS</button>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* DASHBOARD TAB (SINGLE UNDERWRITING) */}
          {activeNav === 'dashboard' && (
            <>
              <div className="col-span-12 lg:col-span-4 bg-surface-container hairline p-6 flex flex-col min-h-[650px]">
                <h2 className="font-headline-sm text-headline-sm text-white uppercase tracking-widest mb-6 flex items-center gap-2 border-b border-outline-variant pb-3">
                  <span className="material-symbols-outlined text-primary-container text-[20px]">assignment</span>
                  LOAN APPLICATION
                </h2>
                
                <form onSubmit={handlePredict} className="flex-1 flex flex-col">
                  <div className="grid grid-cols-2 gap-4 flex-1 content-start">
                    <div>
                      <label className="block font-label-caps text-label-caps text-on-surface-variant mb-1">FICO SCORE</label>
                      <input type="number" name="fico_score" value={loanInput.fico_score} onChange={handleInputChange} className="w-full bg-surface-container-low border border-outline-variant p-2 font-mono-data text-[12px] text-white focus:outline-none focus:border-secondary transition-colors" required />
                    </div>
                    <div>
                      <label className="block font-label-caps text-label-caps text-on-surface-variant mb-1">ANNUAL INCOME ($)</label>
                      <input type="number" name="annual_income" value={loanInput.annual_income} onChange={handleInputChange} className="w-full bg-surface-container-low border border-outline-variant p-2 font-mono-data text-[12px] text-white focus:outline-none focus:border-secondary transition-colors" required />
                    </div>
                    <div>
                      <label className="block font-label-caps text-label-caps text-on-surface-variant mb-1">DTI RATIO (%)</label>
                      <input type="number" name="dti_ratio" step="0.1" value={loanInput.dti_ratio} onChange={handleInputChange} className="w-full bg-surface-container-low border border-outline-variant p-2 font-mono-data text-[12px] text-white focus:outline-none focus:border-secondary transition-colors" required />
                    </div>
                    <div>
                      <label className="block font-label-caps text-label-caps text-on-surface-variant mb-1">LOAN AMOUNT ($)</label>
                      <input type="number" name="loan_amount" value={loanInput.loan_amount} onChange={handleInputChange} className="w-full bg-surface-container-low border border-outline-variant p-2 font-mono-data text-[12px] text-white focus:outline-none focus:border-secondary transition-colors" required />
                    </div>
                    <div>
                      <label className="block font-label-caps text-label-caps text-on-surface-variant mb-1">LOAN TERM (MO)</label>
                      <select name="loan_term" value={loanInput.loan_term} onChange={handleInputChange} className="w-full bg-surface-container-low border border-outline-variant p-2 font-mono-data text-[12px] text-white focus:outline-none focus:border-secondary transition-colors">
                        <option value={36}>36</option>
                        <option value={60}>60</option>
                      </select>
                    </div>
                    <div>
                      <label className="block font-label-caps text-label-caps text-on-surface-variant mb-1">INTEREST RATE (%)</label>
                      <input type="number" name="interest_rate" step="0.1" value={loanInput.interest_rate} onChange={handleInputChange} className="w-full bg-surface-container-low border border-outline-variant p-2 font-mono-data text-[12px] text-white focus:outline-none focus:border-secondary transition-colors" required />
                    </div>
                    <div>
                      <label className="block font-label-caps text-label-caps text-on-surface-variant mb-1">HOME OWNERSHIP</label>
                      <select name="home_ownership" value={loanInput.home_ownership} onChange={handleInputChange} className="w-full bg-surface-container-low border border-outline-variant p-2 font-mono-data text-[12px] text-white focus:outline-none focus:border-secondary transition-colors">
                        <option value="MORTGAGE">MORTGAGE</option>
                        <option value="RENT">RENT</option>
                        <option value="OWN">OWN</option>
                      </select>
                    </div>
                    <div>
                      <label className="block font-label-caps text-label-caps text-on-surface-variant mb-1">LOAN GRADE</label>
                      <select name="loan_grade" value={loanInput.loan_grade} onChange={handleInputChange} className="w-full bg-surface-container-low border border-outline-variant p-2 font-mono-data text-[12px] text-white focus:outline-none focus:border-secondary transition-colors">
                        {['A', 'B', 'C', 'D', 'E', 'F', 'G'].map(g => <option key={g} value={g}>{g}</option>)}
                      </select>
                    </div>
                    <div>
                      <label className="block font-label-caps text-label-caps text-on-surface-variant mb-1">REVOLVING UTIL (%)</label>
                      <input type="number" name="revolving_utilization_pct" step="0.1" value={loanInput.revolving_utilization_pct || ''} onChange={handleInputChange} className="w-full bg-surface-container-low border border-outline-variant p-2 font-mono-data text-[12px] text-white focus:outline-none focus:border-secondary transition-colors" />
                    </div>
                    <div>
                      <label className="block font-label-caps text-label-caps text-on-surface-variant mb-1">DELINQUENCIES (2Y)</label>
                      <input type="number" name="num_delinquencies_2yrs" value={loanInput.num_delinquencies_2yrs} onChange={handleInputChange} className="w-full bg-surface-container-low border border-outline-variant p-2 font-mono-data text-[12px] text-white focus:outline-none focus:border-secondary transition-colors" required />
                    </div>
                  </div>
                  
                  {predError && <div className="mt-4 p-2 bg-[#ef4444]/20 border border-[#ef4444] text-[#ef4444] font-mono-data text-[11px]">{predError}</div>}
                  
                  <button 
                    type="submit" 
                    disabled={loadingPred}
                    className="w-full mt-6 py-3 border border-primary-container text-primary-container font-label-caps text-label-caps font-bold hover:bg-primary-container hover:text-black transition-colors disabled:opacity-50"
                  >
                    {loadingPred ? 'PROCESSING...' : 'SCORE APPLICATION'}
                  </button>
                </form>
              </div>

              <div className="col-span-12 lg:col-span-8 bg-surface-container hairline p-6 flex flex-col min-h-[650px]">
                {!prediction && !loadingPred && (
                  <div className="flex-1 flex flex-col items-center justify-center text-on-surface-variant font-mono-data opacity-50">
                    <span className="material-symbols-outlined text-[48px] mb-4">memory</span>
                    AWAITING_DATA_INPUT
                  </div>
                )}
                
                {prediction && (
                  <div className="flex flex-col h-full">
                    <div className="grid grid-cols-2 gap-8 mb-8">
                      <div>
                        <span className="font-label-caps text-label-caps text-on-surface-variant block mb-2">MODEL GRADE</span>
                        <div className="flex items-center gap-4">
                          <span className={`font-display-lg text-[64px] leading-none ${
                            ['A','B'].includes(prediction.risk_grade) ? 'text-secondary' :
                            prediction.risk_grade === 'C' ? 'text-primary-container' : 'text-[#ef4444]'
                          }`}>
                            {prediction.risk_grade}
                          </span>
                          <span className="font-headline-sm text-primary tracking-widest uppercase">{prediction.risk_grade_label}</span>
                        </div>
                      </div>
                      
                      <div className="flex flex-col justify-center border-l border-outline-variant pl-8">
                         <span className="font-label-caps text-label-caps text-on-surface-variant block mb-2">DEFAULT PROB</span>
                         <span className="font-display-lg text-[48px] text-white leading-none">{(prediction.default_probability * 100).toFixed(1)}%</span>
                         <span className="font-mono-data text-[10px] text-primary-container mt-2">CONFIDENCE: {((1 - prediction.default_probability) * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    <div className="flex-1 flex flex-col border-t border-outline-variant pt-6">
                      <h3 className="font-label-caps text-label-caps text-white mb-6 uppercase tracking-widest">SHAP EXPLAINABILITY (TOP DRIVERS)</h3>
                      <div className="flex-1 min-h-[300px]">
                        <WaterfallChart features={prediction.top_shap_features} />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          {/* ENTITY-VIEW TAB */}
          {activeNav === 'entity' && (
            <>
              {/* Directory Sidebar */}
              <div className="col-span-12 lg:col-span-4 bg-surface-container hairline p-4 flex flex-col h-[650px]">
                <div className="mb-4">
                  <span className="font-label-caps text-label-caps text-on-surface-variant block mb-2">SEARCH BORROWERS / ENTITIES</span>
                  <input 
                    type="text" 
                    placeholder="Search Entity ID, Name or Sector..."
                    value={entitySearch}
                    onChange={(e) => setEntitySearch(e.target.value)}
                    className="w-full bg-surface-container-low border border-outline-variant p-2 font-mono-data text-[12px] text-white focus:outline-none focus:border-secondary"
                  />
                </div>

                <div className="flex-1 overflow-y-auto space-y-2 pr-1 scrollbar-terminal">
                  {filteredEntities.map((ent) => (
                    <button
                      key={ent.entity_id}
                      onClick={() => setSelectedEntity(ent)}
                      className={`w-full text-left p-3 border transition-all ${
                        selectedEntity?.entity_id === ent.entity_id
                          ? 'border-primary bg-surface-container-highest'
                          : 'border-outline-variant bg-surface-container-low hover:border-on-surface-variant'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-1">
                        <span className="font-mono-data text-[12px] font-bold text-white">{ent.name}</span>
                        <span className={`px-2 py-0.5 text-[10px] font-bold ${
                          ent.risk_grade === 'A' ? 'bg-secondary/20 text-secondary border border-secondary' :
                          ent.risk_grade === 'B' ? 'bg-primary-container/20 text-primary-container border border-primary-container' :
                          'bg-[#ef4444]/20 text-[#ef4444] border border-[#ef4444]'
                        }`}>
                          {ent.risk_grade}
                        </span>
                      </div>
                      <div className="flex justify-between font-mono-data text-[10px] text-on-surface-variant">
                        <span>{ent.entity_id} | {ent.sector}</span>
                        <span>{(ent.pd_score * 100).toFixed(1)}% PD</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Entity Detail Inspection */}
              <div className="col-span-12 lg:col-span-8 bg-surface-container hairline p-6 flex flex-col h-[650px] overflow-y-auto scrollbar-terminal">
                {selectedEntity ? (
                  <div className="space-y-6">
                    <div className="flex justify-between items-start border-b border-outline-variant pb-4">
                      <div>
                        <div className="flex items-center space-x-3">
                          <h2 className="font-display-lg text-[24px] text-white">{selectedEntity.name}</h2>
                          <span className="font-mono-data text-[12px] text-primary-container px-2 py-0.5 border border-primary-container">
                            {selectedEntity.entity_id}
                          </span>
                        </div>
                        <span className="font-mono-data text-[12px] text-on-surface-variant">{selectedEntity.sector}</span>
                      </div>

                      <button
                        onClick={() => loadEntityIntoUnderwriter(selectedEntity)}
                        className="px-4 py-2 border border-secondary bg-secondary/10 text-secondary font-label-caps text-label-caps font-bold hover:bg-secondary hover:text-black transition-all flex items-center gap-2"
                      >
                        <span className="material-symbols-outlined text-[14px]">bolt</span>
                        UNDERWRITE THIS ENTITY
                      </button>
                    </div>

                    {/* Entity Key Risk Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-surface-container-low p-3 border border-outline-variant">
                        <span className="font-label-caps text-label-caps text-on-surface-variant block mb-1">TOTAL EXPOSURE</span>
                        <span className="font-mono-data text-[16px] text-white font-bold">{formatCurrency(selectedEntity.total_exposure)}</span>
                      </div>
                      <div className="bg-surface-container-low p-3 border border-outline-variant">
                        <span className="font-label-caps text-label-caps text-on-surface-variant block mb-1">PD SCORE</span>
                        <span className="font-mono-data text-[16px] text-secondary font-bold">{(selectedEntity.pd_score * 100).toFixed(2)}%</span>
                      </div>
                      <div className="bg-surface-container-low p-3 border border-outline-variant">
                        <span className="font-label-caps text-label-caps text-on-surface-variant block mb-1">FICO / CREDIT SCORE</span>
                        <span className="font-mono-data text-[16px] text-white font-bold">{selectedEntity.fico_score}</span>
                      </div>
                      <div className="bg-surface-container-low p-3 border border-outline-variant">
                        <span className="font-label-caps text-label-caps text-on-surface-variant block mb-1">STATUS</span>
                        <span className={`font-mono-data text-[12px] font-bold ${
                          selectedEntity.status === 'PERFORMING' ? 'text-secondary' : 'text-[#ef4444]'
                        }`}>
                          {selectedEntity.status}
                        </span>
                      </div>
                    </div>

                    {/* Financial Ratios & Credit Health */}
                    <div className="bg-surface-container-low p-4 border border-outline-variant space-y-3">
                      <h3 className="font-label-caps text-label-caps text-white uppercase tracking-widest">CREDIT & LEVERAGE PROFILE</h3>
                      <div className="grid grid-cols-3 gap-4 font-mono-data text-[12px]">
                        <div>
                          <span className="text-on-surface-variant block text-[10px]">DTI RATIO</span>
                          <span className="text-white font-bold">{selectedEntity.dti_ratio}%</span>
                        </div>
                        <div>
                          <span className="text-on-surface-variant block text-[10px]">DELINQUENCIES (2Y)</span>
                          <span className="text-white font-bold">{selectedEntity.num_delinquencies}</span>
                        </div>
                        <div>
                          <span className="text-on-surface-variant block text-[10px]">OPEN CREDIT LINES</span>
                          <span className="text-white font-bold">{selectedEntity.open_lines} lines</span>
                        </div>
                      </div>
                    </div>

                    {/* Audit Timeline */}
                    <div className="bg-surface-container-low p-4 border border-outline-variant space-y-3">
                      <h3 className="font-label-caps text-label-caps text-white uppercase tracking-widest">CREDIT EVENT TIMELINE</h3>
                      <div className="space-y-2 font-mono-data text-[11px]">
                        {selectedEntity.recent_events.map((event, idx) => (
                          <div key={idx} className="border-l-2 border-primary-container pl-3 py-1 bg-surface-container">
                            <span className="text-on-surface-variant">{event}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-on-surface-variant font-mono-data">
                    Select an entity from the directory to inspect
                  </div>
                )}
              </div>
            </>
          )}

          {/* SIMULATION (STRESS TESTING ENGINE) TAB */}
          {activeNav === 'simulation' && (
            <>
              {/* Stress Test Config Panel */}
              <div className="col-span-12 lg:col-span-4 bg-surface-container hairline p-6 flex flex-col h-[650px]">
                <h2 className="font-headline-sm text-headline-sm text-white uppercase tracking-widest mb-6 flex items-center gap-2 border-b border-outline-variant pb-3">
                  <span className="material-symbols-outlined text-secondary text-[20px]">tune</span>
                  MACRO STRESS CONFIG
                </h2>

                <form onSubmit={handleRunStressTest} className="flex-1 flex flex-col space-y-6">
                  <div>
                    <div className="flex justify-between font-label-caps text-label-caps mb-2">
                      <span className="text-on-surface-variant">UNEMPLOYMENT SHOCK</span>
                      <span className="text-secondary">+{stressInput.unemployment_shock_pct}%</span>
                    </div>
                    <input 
                      type="range" min="0" max="10" step="0.5"
                      value={stressInput.unemployment_shock_pct}
                      onChange={(e) => setStressInput({ ...stressInput, unemployment_shock_pct: parseFloat(e.target.value) })}
                      className="w-full accent-secondary"
                    />
                    <span className="font-mono-data text-[10px] text-on-surface-variant block mt-1">Simulates labor market contraction</span>
                  </div>

                  <div>
                    <div className="flex justify-between font-label-caps text-label-caps mb-2">
                      <span className="text-on-surface-variant">INTEREST RATE BUMP</span>
                      <span className="text-primary-container">+{stressInput.interest_rate_bump_bps} BPS</span>
                    </div>
                    <input 
                      type="range" min="0" max="500" step="25"
                      value={stressInput.interest_rate_bump_bps}
                      onChange={(e) => setStressInput({ ...stressInput, interest_rate_bump_bps: parseFloat(e.target.value) })}
                      className="w-full accent-primary-container"
                    />
                    <span className="font-mono-data text-[10px] text-on-surface-variant block mt-1">Simulates Central Bank rate hike cycle</span>
                  </div>

                  <div>
                    <div className="flex justify-between font-label-caps text-label-caps mb-2">
                      <span className="text-on-surface-variant">COLLATERAL / HOUSING HAIRCUT</span>
                      <span className="text-[#ef4444]">-{stressInput.collateral_haircut_pct}%</span>
                    </div>
                    <input 
                      type="range" min="0" max="50" step="2.5"
                      value={stressInput.collateral_haircut_pct}
                      onChange={(e) => setStressInput({ ...stressInput, collateral_haircut_pct: parseFloat(e.target.value) })}
                      className="w-full accent-[#ef4444]"
                    />
                    <span className="font-mono-data text-[10px] text-on-surface-variant block mt-1">Simulates real estate / asset devaluation</span>
                  </div>

                  <button
                    type="submit"
                    disabled={loadingStress}
                    className="w-full mt-auto py-3 border border-secondary bg-secondary/10 text-secondary font-label-caps text-label-caps font-bold hover:bg-secondary hover:text-black transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    <span className="material-symbols-outlined text-[16px]">play_arrow</span>
                    {loadingStress ? 'RUNNING STRESS ENGINE...' : 'RUN STRESS SIMULATION'}
                  </button>
                </form>
              </div>

              {/* Stress Results Comparison */}
              <div className="col-span-12 lg:col-span-8 bg-surface-container hairline p-6 flex flex-col h-[650px] overflow-y-auto scrollbar-terminal">
                {stressResult ? (
                  <div className="space-y-6">
                    <div className="flex justify-between items-center border-b border-outline-variant pb-4">
                      <h2 className="font-headline-sm text-headline-sm text-white uppercase tracking-widest">STRESS TEST IMPACT REPORT</h2>
                      <span className="font-mono-data text-[12px] text-[#ef4444] px-3 py-1 bg-[#ef4444]/10 border border-[#ef4444] font-bold">
                        +{stressResult.loss_increase_pct}% LOSS SURGE
                      </span>
                    </div>

                    {/* Comparative Metric Cards */}
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-surface-container-low p-4 border border-outline-variant">
                        <span className="font-label-caps text-label-caps text-on-surface-variant block mb-1">EXPECTED LOSS</span>
                        <div className="flex flex-col font-mono-data">
                          <span className="text-[11px] text-on-surface-variant line-through">{formatCurrency(stressResult.baseline_expected_loss)}</span>
                          <span className="text-[18px] text-[#ef4444] font-bold">{formatCurrency(stressResult.stressed_expected_loss)}</span>
                        </div>
                      </div>

                      <div className="bg-surface-container-low p-4 border border-outline-variant">
                        <span className="font-label-caps text-label-caps text-on-surface-variant block mb-1">STRESSED VaR (95%)</span>
                        <div className="flex flex-col font-mono-data">
                          <span className="text-[11px] text-on-surface-variant line-through">{formatCurrency(stressResult.baseline_var_95)}</span>
                          <span className="text-[18px] text-primary-container font-bold">{formatCurrency(stressResult.stressed_var_95)}</span>
                        </div>
                      </div>

                      <div className="bg-surface-container-low p-4 border border-outline-variant">
                        <div className="font-label-caps text-label-caps text-on-surface-variant block mb-1">STRESSED VaR (99%)</div>
                        <div className="flex flex-col font-mono-data">
                          <span className="text-[11px] text-on-surface-variant line-through">{formatCurrency(stressResult.baseline_var_99)}</span>
                          <span className="text-[18px] text-white font-bold">{formatCurrency(stressResult.stressed_var_99)}</span>
                        </div>
                      </div>
                    </div>

                    {/* Stressed Distribution Chart */}
                    <div className="bg-surface-container-low p-4 border border-outline-variant h-[360px] flex flex-col">
                      <h3 className="font-label-caps text-label-caps text-white mb-4 uppercase tracking-widest">LOSS DISTRIBUTION: BASELINE VS STRESSED</h3>
                      <div className="flex-1 w-full min-h-[260px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={stressResult.loss_distribution_baseline.map((baseVal, i) => ({
                              index: i,
                              baseline: baseVal / 1e6,
                              stressed: (stressResult.loss_distribution_stressed[i] || baseVal) / 1e6,
                            }))}
                            margin={{ top: 10, right: 30, left: 10, bottom: 20 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1D" vertical={false} />
                            <XAxis dataKey="index" stroke="#6B6B63" tick={false} />
                            <YAxis 
                              stroke="#6B6B63" 
                              tickFormatter={(val) => `$${val.toFixed(1)}M`}
                              tick={{ fill: '#d7c4ac', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                            />
                            <Tooltip
                              contentStyle={{ backgroundColor: '#131312', border: '1px solid #1F1F1D' }}
                              formatter={(val: number, name: string) => [`$${val.toFixed(2)}M`, name.toUpperCase()]}
                            />
                            <Bar dataKey="baseline" fill="#47f5db" opacity={0.6} name="Baseline Loss" />
                            <Bar dataKey="stressed" fill="#ef4444" opacity={0.7} name="Stressed Loss" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-on-surface-variant font-mono-data opacity-60">
                    <span className="material-symbols-outlined text-[48px] mb-2">analytics</span>
                    Adjust macro parameters and click RUN STRESS SIMULATION
                  </div>
                )}
              </div>
            </>
          )}

          {/* LOGS TAB */}
          {activeNav === 'logs' && (
            <div className="col-span-12 bg-surface-container hairline p-6 flex flex-col h-[650px]">
              <div className="flex justify-between items-center mb-4 pb-3 border-b border-outline-variant shrink-0">
                <div className="flex items-center space-x-4">
                  <h2 className="font-headline-sm text-headline-sm text-white uppercase tracking-widest flex items-center gap-2">
                    <span className="material-symbols-outlined text-secondary text-[20px]">receipt_long</span>
                    SYSTEM AUDIT LOG TERMINAL
                  </h2>
                  <span className="font-mono-data text-[10px] text-secondary px-2 py-0.5 border border-secondary">
                    {filteredLogs.length} EVENTS LOGGED
                  </span>
                </div>

                {/* Log Level Filters */}
                <div className="flex space-x-2 font-label-caps text-label-caps">
                  {['ALL', 'INFO', 'WARN', 'ALERT', 'SYSTEM'].map(lvl => (
                    <button
                      key={lvl}
                      onClick={() => setLogFilterLevel(lvl)}
                      className={`px-3 py-1 border transition-colors ${
                        logFilterLevel === lvl
                          ? 'border-primary bg-primary/20 text-primary font-bold'
                          : 'border-outline-variant text-on-surface-variant hover:border-on-surface'
                      }`}
                    >
                      {lvl}
                    </button>
                  ))}
                </div>
              </div>

              {/* Log Search */}
              <div className="mb-4 shrink-0">
                <input 
                  type="text" 
                  placeholder="Filter logs by module or message query..."
                  value={logSearch}
                  onChange={(e) => setLogSearch(e.target.value)}
                  className="w-full bg-surface-container-low border border-outline-variant p-2 font-mono-data text-[12px] text-white focus:outline-none focus:border-secondary"
                />
              </div>

              {/* Log Terminal List */}
              <div className="flex-1 overflow-y-auto space-y-2 pr-2 font-mono-data text-[11px] scrollbar-terminal">
                {filteredLogs.map((log, idx) => (
                  <div 
                    key={idx} 
                    className={`p-3 border-l-2 flex justify-between items-start transition-colors ${
                      log.level === 'ALERT' ? 'border-[#ef4444] bg-[#ef4444]/10' :
                      log.level === 'WARN' ? 'border-primary-container bg-primary-container/10' :
                      log.level === 'SYSTEM' ? 'border-primary bg-primary/10' :
                      'border-secondary bg-surface-container-low'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <span className="text-on-surface-variant">[{log.timestamp}]</span>
                      <span className={`px-1.5 py-0.5 text-[9px] font-bold ${
                        log.level === 'ALERT' ? 'bg-[#ef4444] text-white' :
                        log.level === 'WARN' ? 'bg-primary-container text-black' :
                        log.level === 'SYSTEM' ? 'bg-primary text-black' :
                        'bg-secondary text-black'
                      }`}>
                        {log.level}
                      </span>
                      <span className="text-white font-bold">[{log.module}]</span>
                      <span className="text-on-surface">{log.message}</span>
                    </div>

                    <span className="text-on-surface-variant text-[10px] shrink-0">NODE_DX9</span>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>

        {/* Footer */}
        <footer className="flex justify-between px-container-margin py-1 text-[10px] uppercase tracking-widest border-t border-outline-variant bg-surface-container-lowest shrink-0">
          <div className="flex items-center space-x-6 text-on-surface-variant font-mono-data">
            <span className="text-secondary font-bold">SYS-VERSION: 4.8.0-STABLE</span>
            <span className="hover:text-primary cursor-crosshair transition-colors">SYS-STATUS: OK</span>
            <span className="hover:text-primary cursor-crosshair transition-colors">LATENCY: 12MS</span>
            <span className="hover:text-primary cursor-crosshair transition-colors">NODE: DX-9</span>
          </div>
          <div className="font-label-caps text-label-caps text-secondary flex items-center">
            <span className="material-symbols-outlined text-[12px] mr-1">security</span>
            SECURED CONNECTION ESTABLISHED
          </div>
        </footer>
      </main>
    </div>
  );
}
