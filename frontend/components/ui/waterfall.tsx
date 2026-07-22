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
      name: f.feature_name.toUpperCase(),
      value: val,
      fill: val > 0 ? '#ef4444' : '#47f5db', // Red increases risk, Cyan (Secondary) decreases
    };
  });

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 10, right: 30, left: 10, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1D" horizontal={true} vertical={false} />
          <XAxis 
            type="number" 
            stroke="#6B6B63" 
            tick={{ fill: '#d7c4ac', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            axisLine={{ stroke: '#1F1F1D' }}
          />
          <YAxis 
            dataKey="name" 
            type="category" 
            stroke="#6B6B63" 
            width={140}
            tick={{ fill: '#d7c4ac', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            axisLine={{ stroke: '#1F1F1D' }}
          />
          <Tooltip
            cursor={{ fill: '#1c1c1a' }}
            contentStyle={{ backgroundColor: '#131312', borderColor: '#1F1F1D', borderRadius: 0 }}
            itemStyle={{ fontFamily: 'JetBrains Mono', fontSize: 12 }}
            labelStyle={{ color: '#d7c4ac', fontFamily: 'JetBrains Mono', fontSize: 10 }}
            formatter={(value: number, name: string, props: any) => [
              value > 0 ? `+${value.toFixed(4)}` : value.toFixed(4), 
              'SHAP IMPACT'
            ]}
          />
          <Bar dataKey="value" radius={[0, 0, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
