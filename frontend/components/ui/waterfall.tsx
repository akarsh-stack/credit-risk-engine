import React from 'react';
import { SHAPFeature } from '@/lib/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

interface WaterfallProps {
  features: SHAPFeature[];
}

export function WaterfallChart({ features }: WaterfallProps) {
  // Map SHAP features to data recharts can plot
  const data = features.map((f) => {
    const val = f.shap_value;
    return {
      name: f.feature_name,
      value: val,
      fill: val > 0 ? '#ef4444' : '#22c55e', // Red increases risk, Green decreases
    };
  });

  return (
    <div className="w-full h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 10, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={true} vertical={false} />
          <XAxis type="number" stroke="#94a3b8" />
          <YAxis 
            dataKey="name" 
            type="category" 
            stroke="#94a3b8" 
            width={120}
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            cursor={{ fill: '#1e293b' }}
            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }}
            formatter={(value: number) => [value.toFixed(4), 'SHAP Value']}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
