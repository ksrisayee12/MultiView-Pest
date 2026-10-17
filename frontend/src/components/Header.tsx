import React from 'react';
import { LayoutDashboard, ShieldAlert, History, BarChart3, BookOpen, Sprout } from 'lucide-react';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'analysis', label: 'Field Analysis', icon: ShieldAlert },
    { id: 'history', label: 'Scan History', icon: History },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'methodology', label: 'Methodology', icon: BookOpen },
  ];

  return (
    <header className="bg-[#0D47A1] text-white shadow-md border-b border-[#1565C0]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
            <div className="bg-white/10 p-2 rounded-lg backdrop-blur-sm">
              <Sprout className="w-6 h-6 text-green-300" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                Plant Health Assessment
                <span className="text-xs bg-blue-500/30 text-blue-200 px-2 py-0.5 rounded border border-blue-400/30 font-normal">
                  Academic Prototype
                </span>
              </h1>
              <p className="text-xs text-blue-200">Agricultural AI & DIP Field Monitoring System</p>
            </div>
          </div>

          <nav className="flex space-x-1 sm:space-x-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-[#1565C0] text-white border-b-2 border-green-400 shadow-sm'
                      : 'text-blue-100 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden md:inline">{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>
      </div>
    </header>
  );
};
