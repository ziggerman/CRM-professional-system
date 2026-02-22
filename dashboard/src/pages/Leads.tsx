import { useState, useEffect } from 'react';
import { Search, Filter, MoreVertical, User } from 'lucide-react';

export default function Leads() {
    const [search, setSearch] = useState('');
    const [leads, setLeads] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [total, setTotal] = useState(0);

    const API_TOKEN = 'dev_secret_token_123';
    const HEADERS = { 'Authorization': `Bearer ${API_TOKEN}` };

    const fetchLeads = async () => {
        try {
            // Fetch first page, 50 items
            const resp = await fetch('http://localhost:8000/api/v1/leads?page=1&page_size=100', { headers: HEADERS });
            const data = await resp.json();
            setLeads(data.items || []);
            setTotal(data.total || 0);
        } catch (err) {
            console.error("Failed to fetch leads:", err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchLeads();
    }, []);

    const filteredLeads = leads.filter(lead =>
        (lead.full_name?.toLowerCase().includes(search.toLowerCase())) ||
        (lead.company?.toLowerCase().includes(search.toLowerCase())) ||
        (lead.email?.toLowerCase().includes(search.toLowerCase()))
    );

    return (
        <div className="p-6 md:p-12 animate-fade-in w-full h-full flex flex-col">
            <header className="flex justify-between items-center mb-10">
                <div>
                    <h1 className="text-3xl font-bold text-zinc-100">Leads Management</h1>
                    <p className="text-zinc-400 mt-2">Manage {total} pipeline contacts.</p>
                </div>
                <button className="flex items-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium transition-colors cursor-not-allowed opacity-50">
                    <span>+ Add Lead</span>
                </button>
            </header>

            <div className="flex items-center space-x-4 mb-6">
                <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-zinc-500" />
                    <input
                        type="text"
                        placeholder="Search leads..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="w-full bg-[#18181b] border border-white/10 rounded-lg pl-10 pr-4 py-2.5 text-zinc-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                    />
                </div>
                <button className="flex items-center space-x-2 bg-[#18181b] border border-white/10 text-zinc-300 px-4 py-2.5 rounded-lg hover:bg-white/5 transition-colors">
                    <Filter className="w-5 h-5" />
                    <span>Filters</span>
                </button>
            </div>

            <div className="glass rounded-xl border border-white/10 overflow-hidden flex-1 flex flex-col">
                <div className="overflow-x-auto">
                    {loading ? (
                        <div className="p-24 text-center text-zinc-500 animate-pulse">Fetching leads from database...</div>
                    ) : filteredLeads.length === 0 ? (
                        <div className="p-24 text-center text-zinc-500">No leads found.</div>
                    ) : (
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr className="border-b border-white/10 bg-white/5 text-zinc-400 text-sm">
                                    <th className="py-4 px-6 font-medium">Lead Info</th>
                                    <th className="py-4 px-6 font-medium">Source / Stage</th>
                                    <th className="py-4 px-6 font-medium text-center">AI Score</th>
                                    <th className="py-4 px-6 font-medium text-right">Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {filteredLeads.map(lead => (
                                    <tr key={lead.id} className="border-b border-white/5 hover:bg-white/5 transition-colors group">
                                        <td className="py-4 px-6">
                                            <div className="flex items-center space-x-3">
                                                <div className="w-8 h-8 rounded-full bg-zinc-800 flex items-center justify-center text-zinc-500">
                                                    <User className="w-4 h-4" />
                                                </div>
                                                <div className="flex flex-col">
                                                    <span className="font-medium text-zinc-200">{lead.full_name || 'Anonymous'}</span>
                                                    <span className="text-sm text-zinc-500">{lead.company || lead.email || 'No company info'}</span>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="py-2 px-6">
                                            <div className="flex flex-col gap-1">
                                                <span className="text-[10px] text-zinc-500 uppercase tracking-wider">{lead.source}</span>
                                                <span className="inline-flex items-center w-fit px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                                                    {lead.stage}
                                                </span>
                                            </div>
                                        </td>
                                        <td className="py-4 px-6 text-center">
                                            {lead.ai_score !== null ? (
                                                <div className="flex items-center justify-center space-x-2">
                                                    <div className={`w-2 h-2 rounded-full ${lead.ai_score >= 0.6 ? 'bg-emerald-400' : 'bg-rose-400 animate-pulse'}`} />
                                                    <span className={`font-semibold ${lead.ai_score >= 0.6 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                                        {Math.round(lead.ai_score * 100)}%
                                                    </span>
                                                </div>
                                            ) : (
                                                <span className="text-zinc-600 text-xs italic">Not analyzed</span>
                                            )}
                                        </td>
                                        <td className="py-4 px-6 text-right">
                                            <button className="text-zinc-500 hover:text-zinc-300 p-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                <MoreVertical className="w-5 h-5" />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            </div>
        </div>
    );
}
