import { useState, useEffect } from 'react';
import { Users, DollarSign, TrendingUp, Bell } from 'lucide-react';
import { AreaChart, Area, XAxis, Tooltip, ResponsiveContainer } from 'recharts';

const funnelData = [
  { stage: 'New', leads: 400 },
  { stage: 'Contacted', leads: 280 },
  { stage: 'Qualified', leads: 150 },
  { stage: 'KYC / Neg.', leads: 80 },
  { stage: 'Closed Won', leads: 45 },
];


export default function Dashboard() {
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [funnel, setFunnel] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const API_TOKEN = 'dev_secret_token_123'; // In production this would be from context/storage
  const HEADERS = { 'Authorization': `Bearer ${API_TOKEN}` };

  const fetchData = async () => {
    try {
      const [statsRes, funnelRes] = await Promise.all([
        fetch('http://localhost:8000/api/v1/dashboard', { headers: HEADERS }),
        fetch('http://localhost:8000/api/v1/dashboard/conversion-funnel', { headers: HEADERS })
      ]);

      const statsData = await statsRes.json();
      const funnelData = await funnelRes.json();

      setStats(statsData);
      setFunnel(funnelData);
    } catch (err) {
      console.error("Dashboard fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();

    // Connect to FastAPI WebSocket
    const wsUrl = `ws://localhost:8000/api/v1/ws/dashboard`;
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("WebSocket event:", data);
        setEvents(prev => [data, ...prev].slice(0, 50));
        // Refresh stats on any significant event
        fetchData();
      } catch (e) {
        console.error("Failed to parse WS message", e);
      }
    };

    return () => socket.close();
  }, []);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-zinc-500">
        <div className="animate-pulse">Loading real-time metrics...</div>
      </div>
    );
  }

  // Format funnel for chart
  const funnelChartData = funnel?.leads_funnel?.labels.map((label: string, i: number) => ({
    stage: label,
    leads: funnel.leads_funnel.values[i]
  })) || [];

  return (
    <div className="text-foreground p-6 md:p-12 animate-fade-in relative z-10 w-full h-full">
      <header className="flex justify-between items-center mb-10">
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
            Ascend Edge CRM
          </h1>
          <p className="text-zinc-400 mt-1">Real-time Command Center</p>
        </div>
        <div className="flex items-center gap-4">
          <div className={`px-3 py-1 rounded-full text-xs font-medium border ${connected ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20'}`}>
            {connected ? '● LIVE' : '○ CONNECTING...'}
          </div>
          <button className="p-2 rounded-full bg-white/5 hover:bg-white/10 transition cursor-pointer">
            <Bell className="w-5 h-5 text-zinc-300" />
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 w-full max-w-7xl mx-auto">
        <MetricCard
          title="Total Leads"
          value={stats?.leads?.total?.toLocaleString() || "0"}
          icon={Users}
          trend={stats?.leads?.new > 0 ? `+${stats?.leads?.new}` : "0"}
        />
        <MetricCard
          title="Conversion Rate"
          value={`${stats?.conversion_rate || 0}%`}
          icon={TrendingUp}
          trend={stats?.conversion_rate > 20 ? "+2.4" : "0"}
        />
        <MetricCard
          title="Pipeline Value"
          value={`$${(stats?.total_revenue || 0).toLocaleString()}`}
          icon={DollarSign}
          trend="+8"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full max-w-7xl mx-auto">
        <div className="lg:col-span-2 glass rounded-2xl p-6 h-96">
          <h2 className="text-lg font-semibold mb-6 text-zinc-200">Pipeline Funnel</h2>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={funnelChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorLeads" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="stage" stroke="#52525b" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px' }}
                  itemStyle={{ color: '#e4e4e7' }}
                />
                <Area type="monotone" dataKey="leads" stroke="#60a5fa" strokeWidth={3} fillOpacity={1} fill="url(#colorLeads)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass rounded-2xl p-6 h-96 flex flex-col">
          <h2 className="text-lg font-semibold mb-4 text-zinc-200">Live AI Events</h2>
          <div className="flex-1 overflow-auto space-y-4 pr-2">
            {events.length === 0 ? (
              <p className="text-zinc-500 text-sm italic">Listening for live events...</p>
            ) : (
              events.map((ev, i) => (
                <EventItem
                  key={i}
                  title={ev.event === "lead_created" ? "New Lead" : ev.event === "lead_updated" ? "Lead Updated" : ev.event}
                  desc={ev.stage ? `Moved to ${ev.stage.toUpperCase()}` : `ID: ${ev.lead_id || ev.sale_id || '?'}`}
                  time="Just now"
                  type={ev.stage === "paid" ? "revenue" : ev.event?.includes("high_value") ? "hot" : "info"}
                />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function MetricCard({ title, value, icon: Icon, trend }: any) {
  return (
    <div className="glass p-6 rounded-2xl relative overflow-hidden group hover:border-blue-500/30 transition-colors">
      <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full blur-2xl -mr-10 -mt-10 transition-transform group-hover:scale-150" />
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-zinc-400 font-medium">{title}</h3>
        <div className="p-2 bg-white/5 rounded-lg">
          <Icon className="w-5 h-5 text-blue-400" />
        </div>
      </div>
      <div className="flex items-end gap-3">
        <div className="text-3xl font-bold">{value}</div>
        <div className="text-emerald-400 text-sm font-medium mb-1">{trend}%</div>
      </div>
    </div>
  )
}

function EventItem({ title, desc, time, type }: any) {
  const config = {
    hot: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    revenue: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    info: 'bg-blue-500/20 text-blue-400 border-blue-500/30'
  }
  // @ts-ignore
  const typeStyle = config[type] || config.info;
  return (
    <div className={`flex items-start gap-4 p-3 rounded-xl bg-white/5 hover:bg-white/10 transition border border-transparent`}>
      <div className={`mt-1 flex-shrink-0 w-2 h-2 rounded-full ${type === 'hot' ? 'bg-orange-400 shadow-[0_0_8px_rgba(251,146,60,0.8)]' : type === 'revenue' ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]' : 'bg-blue-400'} animate-pulse`} />
      <div className="flex-1">
        <h4 className="text-sm font-semibold text-zinc-200">{title}</h4>
        <p className="text-xs text-zinc-400 mt-1">{desc}</p>
      </div>
      <div className="text-[10px] text-zinc-500 uppercase tracking-widest">{time}</div>
    </div>
  )
}
