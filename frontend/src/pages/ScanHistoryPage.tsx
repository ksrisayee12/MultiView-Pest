import React, { useEffect, useState } from 'react';
import { ScanSummary } from '../types';
import { History, ArrowRight, RefreshCw } from 'lucide-react';

interface ScanHistoryPageProps {
  onSelectScan: (scan: ScanSummary) => void;
  onNavigate: (tab: string) => void;
}

export const ScanHistoryPage: React.FC<ScanHistoryPageProps> = ({
  onSelectScan,
  onNavigate
}) => {
  const [scans, setScans] = useState<ScanSummary[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchScans = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/scans');
      if (response.ok) {
        const data = await response.json();
        setScans(data.scans || []);
      }
    } catch (err) {
      console.error('Fetch scans error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchScans();
  }, []);

  return (
    <div className="space-y-7">
      {/* Page Title */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold text-[#0D47A1]">
            Scan History Records
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Historical log of field drone composite scans and health assessments
          </p>
        </div>

        <button
          onClick={fetchScans}
          disabled={loading}
          className="bg-[#E3F2FD] hover:bg-[#BBDEFB] text-[#0D47A1] px-4 py-2 rounded-lg text-sm font-semibold transition-colors flex items-center gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh Log
        </button>
      </div>

      {/* History Table */}
      <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-[#F5F9FF] border-b border-gray-200 text-gray-600 text-xs uppercase font-bold tracking-wider">
              <tr>
                <th className="px-5 py-3.5">Scan ID</th>
                <th className="px-5 py-3.5">Timestamp</th>
                <th className="px-5 py-3.5">Crop</th>
                <th className="px-5 py-3.5">Healthy</th>
                <th className="px-5 py-3.5">Confirmed Disease</th>
                <th className="px-5 py-3.5">Disputed</th>
                <th className="px-5 py-3.5">Health Index</th>
                <th className="px-5 py-3.5 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {scans.length > 0 ? (
                scans.map((scanItem) => (
                  <tr key={scanItem.scan_id} className="hover:bg-blue-50/50 transition-colors">
                    <td className="px-5 py-4 font-bold text-[#0D47A1]">{scanItem.scan_id}</td>
                    <td className="px-5 py-4 text-xs text-gray-500">
                      {new Date(scanItem.created_at).toLocaleString()}
                    </td>
                    <td className="px-5 py-4 text-gray-700">{scanItem.crop}</td>
                    <td className="px-5 py-4 text-green-600 font-semibold">
                      {scanItem.healthy_tiles} / {scanItem.total_tiles}
                    </td>
                    <td className="px-5 py-4 text-red-600 font-semibold">
                      {scanItem.confirmed_disease_tiles}
                    </td>
                    <td className="px-5 py-4 text-amber-600 font-semibold">
                      {scanItem.disputed_tiles}
                    </td>
                    <td className="px-5 py-4 font-bold text-gray-800">
                      {scanItem.plant_health_index} / 100
                    </td>
                    <td className="px-5 py-4 text-right">
                      <button
                        onClick={() => {
                          onSelectScan(scanItem);
                          onNavigate('field-scan');
                        }}
                        className="bg-[#1565C0] hover:bg-[#0D47A1] text-white px-3 py-1.5 rounded text-xs font-semibold transition-colors inline-flex items-center gap-1.5"
                      >
                        View Field
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={8} className="px-5 py-8 text-center text-gray-500 text-sm">
                    No scan history records found. Run a field simulation to populate log.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
