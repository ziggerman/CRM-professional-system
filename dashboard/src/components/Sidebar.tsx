import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Users, DollarSign, Settings, Megaphone } from 'lucide-react';

interface SidebarProps {
    children?: ReactNode;
}

export function Sidebar({ children }: SidebarProps) {
    const navItems = [
        { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
        { to: '/leads', icon: Users, label: 'Leads' },
        { to: '/sales', icon: DollarSign, label: 'Sales Pipeline' },
        { to: '/broadcast', icon: Megaphone, label: 'Broadcast' },
        { to: '/settings', icon: Settings, label: 'Settings' },
    ];

    return (
        <div className="flex h-screen w-full bg-[#09090b] text-zinc-100 overflow-hidden">
            {/* Persistent Left Sidebar */}
            <aside className="w-64 border-r border-white/10 bg-[#09090b] flex flex-col shrink-0">
                <div className="h-16 flex items-center px-6 border-b border-white/10">
                    <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center mr-3">
                        <div className="w-4 h-4 rounded-sm bg-indigo-400" />
                    </div>
                    <span className="font-bold text-lg tracking-tight">AEL CRM</span>
                </div>

                <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
                    {navItems.map((item) => (
                        <NavLink
                            key={item.to}
                            to={item.to}
                            className={({ isActive }) =>
                                `flex items-center px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${isActive
                                    ? 'bg-white/10 text-white'
                                    : 'text-zinc-400 hover:text-white hover:bg-white/5'
                                }`
                            }
                        >
                            <item.icon className="w-5 h-5 mr-3 shrink-0" />
                            {item.label}
                        </NavLink>
                    ))}
                </nav>

                <div className="p-4 border-t border-white/10">
                    <div className="flex items-center px-3 py-2">
                        <div className="w-8 h-8 rounded-full bg-zinc-800 flex items-center justify-center text-sm font-medium shrink-0">
                            AD
                        </div>
                        <div className="ml-3 truncate">
                            <p className="text-sm font-medium text-white truncate">Admin User</p>
                            <p className="text-xs text-zinc-500 truncate">admin@ascendedge.com</p>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main Content Area */}
            <main className="flex-1 flex flex-col min-w-0 bg-[#0c0c0e]">
                <div className="flex-1 overflow-y-auto w-full relative">
                    <div className="absolute inset-0 bg-[radial-gradient(circle_500px_at_50%_-30%,rgba(120,119,198,0.1),transparent)] pointer-events-none" />
                    {children}
                </div>
            </main>
        </div>
    );
}
