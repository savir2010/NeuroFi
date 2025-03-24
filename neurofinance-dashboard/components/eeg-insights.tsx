"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Chart, ChartContainer, ChartDonut, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

export function EegInsights() {
  // Mock data for stress and boredom levels
  const stressData = [
    { name: "High", value: 35, color: "#ef4444" },
    { name: "Medium", value: 45, color: "#f97316" },
    { name: "Low", value: 20, color: "#22c55e" },
  ]

  const boredomData = [
    { name: "High", value: 15, color: "#3b82f6" },
    { name: "Medium", value: 30, color: "#8b5cf6" },
    { name: "Low", value: 55, color: "#ec4899" },
  ]

  return (
    <Card className="border-gray-800 bg-gray-950 shadow-lg shadow-blue-900/10">
      <CardHeader className="pb-3">
        <CardTitle className="text-xl font-bold text-white">EEG Insights</CardTitle>
        <CardDescription className="text-gray-400">
          Real-time neurological metrics during trading sessions
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="stress" className="w-full">
          <TabsList className="mb-4 grid w-full grid-cols-2 bg-gray-900">
            <TabsTrigger value="stress">Stress Levels</TabsTrigger>
            <TabsTrigger value="boredom">Boredom Levels</TabsTrigger>
          </TabsList>
          <TabsContent value="stress" className="mt-0">
            <div className="flex flex-col items-center justify-center gap-4 md:flex-row">
              <div className="w-full max-w-[240px]">
                <ChartContainer className="h-[240px]">
                  <Chart>
                    <ChartDonut data={stressData} dataKey="value" nameKey="name" innerRadius={60} outerRadius={80}>
                      <ChartTooltip>
                        <ChartTooltipContent />
                      </ChartTooltip>
                    </ChartDonut>
                  </Chart>
                </ChartContainer>
              </div>
              <div className="flex flex-col gap-2">
                <h4 className="text-lg font-medium text-white">Stress Analysis</h4>
                <p className="text-sm text-gray-400">
                  Your stress levels during trading are moderate. Consider implementing mindfulness techniques during
                  high-volatility periods.
                </p>
                <div className="mt-2 space-y-1">
                  {stressData.map((item) => (
                    <div key={item.name} className="flex items-center gap-2">
                      <div className="h-3 w-3 rounded-full" style={{ backgroundColor: item.color }} />
                      <span className="text-xs text-gray-300">
                        {item.name}: {item.value}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </TabsContent>
          <TabsContent value="boredom" className="mt-0">
            <div className="flex flex-col items-center justify-center gap-4 md:flex-row">
              <div className="w-full max-w-[240px]">
                <ChartContainer className="h-[240px]">
                  <Chart>
                    <ChartDonut data={boredomData} dataKey="value" nameKey="name" innerRadius={60} outerRadius={80}>
                      <ChartTooltip>
                        <ChartTooltipContent />
                      </ChartTooltip>
                    </ChartDonut>
                  </Chart>
                </ChartContainer>
              </div>
              <div className="flex flex-col gap-2">
                <h4 className="text-lg font-medium text-white">Boredom Analysis</h4>
                <p className="text-sm text-gray-400">
                  Low boredom levels indicate high engagement with your trading activities. This is optimal for
                  decision-making.
                </p>
                <div className="mt-2 space-y-1">
                  {boredomData.map((item) => (
                    <div key={item.name} className="flex items-center gap-2">
                      <div className="h-3 w-3 rounded-full" style={{ backgroundColor: item.color }} />
                      <span className="text-xs text-gray-300">
                        {item.name}: {item.value}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

