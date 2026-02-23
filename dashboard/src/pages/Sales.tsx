import { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Download, Plus } from 'lucide-react';

export default function Sales() {
    const [revenueData, setRevenueData] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    const API_TOKEN = 'dev_secret_token_123';
    const HEADERS = { 'Authorization': `Bearer ${API_TOKEN}` };

    useEffect(() => {
        const fetchRevenue = async () => {
            try {
                const resp = await fetch('http://localhost:8000/api/v1/dashboard/revenue-by-month', { headers: HEADERS });
                const data = await resp.json();

                // Format for Recharts
                const formatted = data.labels.map((label: string, i: number) => ({
                    month: label,
                    revenue: data.values[i]
                }));

                setRevenueData(formatted);
            } catch (err) {
                console.error("Failed to fetch revenue data:", err);
            } finally {
                setLoading(false);
            }
        };

        fetchRevenue();
    }, []);

    return (
        <div className="p-6 md:p-12 animate-fade-in w-full h-full flex flex-col">
            <header className="flex justify-between items-center mb-10">
                <div>
                    <h1 className="text-3xl font-bold text-zinc-100">Sales Pipeline</h1>
                    <p className="text-zinc-400 mt-2">Track revenue and manage active deals.</p>
                </div>
                <div className="flex space-x-3">
                    <button className="flex items-center space-x-2 bg-white/5 border border-white/10 hover:bg-white/10 text-white px-4 py-2 rounded-lg font-medium transition-colors cursor-not-allowed opacity-50">
                        <Download className="w-4 h-4" />
                        <span>Export CSV</span>
                    </button>
                    <button className="flex items-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium transition-colors cursor-not-allowed opacity-50">
                        <Plus className="w-4 h-4" />
                        <span>New Deal</span>
                    </button>
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8 w-full mx-auto">
                <div className="lg:col-span-2 glass rounded-2xl p-6 h-96 flex flex-col">
                    <h2 className="text-lg font-semibold mb-6 text-zinc-200">Revenue Performance</h2>
                    <div className="flex-1 w-full mt-2">
                        {loading ? (
                            <div className="flex items-center justify-center h-full text-zinc-500 animate-pulse">Loading revenue charts...</div>
                        ) : revenueData.length === 0 ? (
                            <div className="flex items-center justify-center h-full text-zinc-500 italic">No revenue data available yet.</div>
                        ) : (
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={revenueData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                    <defs>
                                        <linearGradient id="colorRevenue" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                                        </linearGradient>
                                    </defs>
                                    <XAxis dataKey="month" stroke="#52525b" fontSize={12} tickLine={false} axisLine={false} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px' }}
                                        itemStyle={{ color: '#e4e4e7' }}
                                        formatter={(val: any) => [`$${Number(val).toLocaleString()}`, 'Revenue']}
                                    />
                                    <Area type="monotone" dataKey="revenue" stroke="#34d399" strokeWidth={3} fillOpacity={1} fill="url(#colorRevenue)" />
                                </AreaChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>

                <div className="glass rounded-2xl p-6 h-96 flex flex-col justify-between">
                    <div>
                        <h2 className="text-lg font-semibold mb-2 text-zinc-200">Pipeline Velocity</h2>
                        <p className="text-zinc-400 text-sm mb-6">Average time to close deals across stages.</p>

                        <div className="space-y-4">
                            <div className="flex justify-between text-sm">
                                <span className="text-zinc-300">Contact to Qualify</span>
                                <span className="font-semibold text-white">4 days</span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-zinc-300">Qualify to KYC</span>
                                <span className="font-semibold text-white">12 days</span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-zinc-300">KYC to Closed Won</span>
                                <span className="font-semibold text-emerald-400">8 days</span>
                            </div>
                        </div>
                    </div>

                    <div className="p-4 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
                        <p className="text-sm text-indigo-200 text-center">
                            Your pipeline velocity is <strong>14% faster</strong> than the 30-day average. 🚀
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
