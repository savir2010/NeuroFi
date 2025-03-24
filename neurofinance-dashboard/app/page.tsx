"use client"; // Mark this as a Client Component

import { useEffect, useState } from "react";
import { DashboardHeader } from "@/components/dashboard-header";
import { DashboardShell } from "@/components/dashboard-shell";
import { PolarChartDisplay } from "@/components/polar-chart-display";
import { DoubleLineGraph } from "@/components/double-line-graph";
import { Watchlist } from "@/components/watchlist";
import { FinancialAdvisor } from "@/components/financial-advisor";
import { NewsSection } from "@/components/news-section";
import { StartReadingButton } from "@/components/start-reading-button";
import { DonutChart } from "@/components/donut-chart";
import { fetchEEGData, fetchJsonData } from "@/lib/api";

export default function DashboardPage() {
  const [stressData, setStressData] = useState([
    { name: "Stress", value: 80, color: "#ef4444" },
    { name: "Normal", value: 20, color: "#22c55e" },
  ]);

  const [boredomData, setBoredomData] = useState([
    { name: "Boredom", value: 45, color: "#3b82f6" },
    { name: "Engagement", value: 55, color: "#ec4899" },
  ]);

  const [eegData, setEEGData] = useState<{
    stress_percentage: number;
    boredom_percentage: number;
    emotion_scores: {
      Anger: number;
      Anxiety: number;
      Calmness: number;
      Fear: number;
      Sadness: number;
    };
    data_log: string;
  } | null>(null);

  const [report, setReport] = useState(null);
  const [isLoading, setIsLoading] = useState(false); // Add loading state

  // Function to handle the "Start Reading" button click
  const handleStartReading = async () => {
    setIsLoading(true); // Set loading state to true
    try {
      const data = await fetchEEGData(); // Call the API
      setEEGData(data);

      // Update stress and boredom data based on the response
      setStressData([
        { name: "Stress", value: data.stress_percentage, color: "#ef4444" },
        { name: "Normal", value: 100 - data.stress_percentage, color: "#22c55e" },
      ]);
      setBoredomData([
        { name: "Boredom", value: data.boredom_percentage, color: "#3b82f6" },
        { name: "Engagement", value: 100 - data.boredom_percentage, color: "#ec4899" },
      ]);
    } catch (error) {
      console.error("Error fetching EEG data:", error);
    } finally {
      setIsLoading(false); // Set loading state to false
    }
  };

  // Fetch JSON data from the /get_json endpoint
  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await fetchJsonData(); // Call the API
        if (data.error) {
          console.error(data.error);
          return;
        }

        // Update stress and boredom data based on the response
        setStressData([
          { name: "Stress", value: data.stress_percentage, color: "#ef4444" },
          { name: "Normal", value: 100 - data.stress_percentage, color: "#22c55e" },
        ]);
        setBoredomData([
          { name: "Boredom", value: data.boredom_percentage, color: "#3b82f6" },
          { name: "Engagement", value: 100 - data.boredom_percentage, color: "#ec4899" },
        ]);

        // Set EEG data for the PolarChartDisplay
        setEEGData(data);
      } catch (error) {
        console.error("Error fetching JSON data:", error);
      }
    };

    fetchData();
  }, []);

  // Parse the data_log from the response
  const dataLog = eegData?.data_log ? JSON.parse(eegData.data_log) : null;

  // Format data for the line graph
  const lineGraphData = dataLog
    ? Object.values(dataLog).map((entry: any) => ({
        date: entry.Timestamp,
        stress: parseFloat(entry["Stress Percentage"]),
        boredom: parseFloat(entry["Boredom Percentage"]),
      }))
    : [];

  return (
    <div className="flex min-h-screen flex-col bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <DashboardHeader />
      <DashboardShell className="py-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Main Analytics Section */}
          <div className="col-span-12 lg:col-span-8">
            <div className="grid grid-cols-8 gap-6">
              {/* Top row - Primary charts */}
              <div className="col-span-3 bg-white dark:bg-gray-800 rounded-xl shadow-sm h-[350px]">
                {/* Pass emotion_scores to PolarChartDisplay */}
                <PolarChartDisplay data={eegData?.emotion_scores} />
              </div>
              <div className="col-span-5 bg-white dark:bg-gray-800 rounded-xl shadow-sm h-[350px]">
                {/* Use data_log for the line graph */}
                <DoubleLineGraph data={lineGraphData} />
              </div>

              {/* Bottom row - Financial tools */}
              <div className="col-span-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm h-[400px]">
                <Watchlist />
              </div>
              <div className="col-span-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm h-[400px]">
                <FinancialAdvisor />
              </div>
            </div>
          </div>

          {/* Right sidebar */}
          <div className="col-span-12 lg:col-span-4">
            <div className="grid grid-cols-4 gap-6">
              {/* Metrics overview */}
              <div className="col-span-2 bg-white dark:bg-gray-800 rounded-xl shadow-sm p-4">
                <DonutChart
                  title="Stress Levels"
                  description="Current distribution"
                  data={stressData}
                  height={200}
                  hideBottomText={true}
                />
              </div>
              <div className="col-span-2 bg-white dark:bg-gray-800 rounded-xl shadow-sm p-4">
                <DonutChart
                  title="Boredom Levels"
                  description="Current distribution"
                  data={boredomData}
                  height={200}
                  hideBottomText={true}
                />
              </div>

              {/* Action button - reduced margin */}
              <div className="col-span-4 flex justify-center my-2">
                <StartReadingButton onClick={handleStartReading} disabled={isLoading} />
              </div>

              {/* News feed - Now positioned higher */}
              <div className="col-span-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm h-[400px] -mt-2">
                <NewsSection />
              </div>
            </div>
          </div>
        </div>
      </DashboardShell>
    </div>
  );
}