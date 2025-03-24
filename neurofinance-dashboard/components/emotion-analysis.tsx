"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Chart,
  ChartContainer,
  ChartPolarAngleAxis,
  ChartPolarGrid,
  ChartPolarRadiusAxis,
  ChartRadar,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

export function EmotionAnalysis() {
  // Mock data for emotional states
  const emotionData = [
    { emotion: "Anxiety", value: 65, fullMark: 100 },
    { emotion: "Fear", value: 40, fullMark: 100 },
    { emotion: "Sadness", value: 25, fullMark: 100 },
    { emotion: "Anger", value: 30, fullMark: 100 },
    { emotion: "Calm", value: 70, fullMark: 100 },
  ]

  return (
    <Card className="border-gray-800 bg-gray-950 shadow-lg shadow-purple-900/10">
      <CardHeader className="pb-3">
        <CardTitle className="text-xl font-bold text-white">Emotional State</CardTitle>
        <CardDescription className="text-gray-400">EEG-based emotional analysis during market activity</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center">
          <ChartContainer className="h-[300px] w-full">
            <Chart>
              <ChartPolarGrid stroke="rgba(255, 255, 255, 0.1)" />
              <ChartPolarAngleAxis dataKey="emotion" tick={{ fill: "#a3a3a3", fontSize: 12 }} />
              <ChartPolarRadiusAxis angle={90} domain={[0, 100]} tick={false} axisLine={false} />
              <ChartRadar
                dataKey="value"
                data={emotionData}
                fill="rgba(139, 92, 246, 0.6)"
                fillOpacity={0.6}
                stroke="#8b5cf6"
              >
                <ChartTooltip>
                  <ChartTooltipContent />
                </ChartTooltip>
              </ChartRadar>
            </Chart>
          </ChartContainer>
          <div className="mt-4 text-center">
            <p className="text-sm text-gray-400">
              Your emotional state suggests heightened anxiety but good calm levels. This balance is favorable for
              calculated risk-taking.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

