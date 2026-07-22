import React from 'react';

interface GaugeProps {
  probability: number; // 0 to 1
  grade: string;
}

export function ProbabilityGauge({ probability, grade }: GaugeProps) {
  // SVG properties
  const size = 200;
  const strokeWidth = 20;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  // Arc is half a circle (180 degrees)
  const arcLength = circumference / 2;
  
  // Calculate dash offset based on probability
  const fillPct = Math.min(Math.max(probability, 0), 1);
  const strokeDashoffset = arcLength - (fillPct * arcLength);

  // Determine color based on grade
  let color = '#22c55e'; // Green for A/B
  if (['C'].includes(grade)) color = '#f59e0b'; // Amber for C
  if (['D', 'E'].includes(grade)) color = '#ef4444'; // Red for D/E

  return (
    <div className="flex flex-col items-center justify-center relative w-[200px] h-[120px]">
      <svg
        width={size}
        height={size / 2 + strokeWidth}
        viewBox={`0 0 ${size} ${size / 2 + strokeWidth}`}
        className="transform -rotate-180 origin-center"
      >
        {/* Background Arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#334155"
          strokeWidth={strokeWidth}
          strokeDasharray={`${arcLength} ${circumference}`}
          strokeDashoffset="0"
          strokeLinecap="round"
        />
        {/* Fill Arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={`${arcLength} ${circumference}`}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      
      {/* Text inside gauge */}
      <div className="absolute bottom-2 flex flex-col items-center">
        <span className="text-4xl font-bold text-white tracking-tighter">
          {(probability * 100).toFixed(1)}%
        </span>
        <span className="text-sm text-slate-400 mt-1 uppercase tracking-wider font-semibold">
          Default Prob
        </span>
      </div>
    </div>
  );
}
