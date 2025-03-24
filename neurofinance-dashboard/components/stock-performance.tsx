"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Chart,
  ChartContainer,
  ChartLine,
  ChartTooltip,
  ChartTooltipContent,
  ChartXAxis,
  ChartYAxis,
  ChartGrid,
} from "@/components/ui/chart"

export function StockPerformance() {
  // Mock data for stock performance
  const performanceData = [
    { date: "Jan", value: 4000 },
    { date: "Feb", value: 3500 },
    { date: "Mar", value: 4500 },
    { date: "Apr", value: 5000 },
    { date: "May", value: 4800 },
    { date: "Jun", value: 5200 },
    { date: "Jul", value: 5800 },
    { date: "Aug", value: 6300 },
    { date: "Sep", value: 6100 },
    { date: "Oct", value: 6500 },
    { date: "Nov", value: 7000 },
    { date: "Dec", value: 7500 },
  ]

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle>Performance Analysis</CardTitle>
        <CardDescription>Your portfolio performance over time</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="yearly" className="w-full">
          <TabsList className="mb-4 grid w-full grid-cols-3">
            <TabsTrigger value="monthly">Monthly</TabsTrigger>
            <TabsTrigger value="quarterly">Quarterly</TabsTrigger>
            <TabsTrigger value="yearly">Yearly</TabsTrigger>
          </TabsList>
          <TabsContent value="monthly" className="mt-0">
            <ChartContainer className="h-[300px]">
              <Chart data={performanceData.slice(-3)}>
                <ChartGrid strokeDasharray="3 3" vertical={false} />
                <ChartXAxis dataKey="date" />
                <ChartYAxis />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLine
                  type="monotone"
                  dataKey="value"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={{ stroke: "hsl(var(--primary))", strokeWidth: 2, fill: "hsl(var(--primary))" }}
                />
              </Chart>
            </ChartContainer>
          </TabsContent>
          <TabsContent value="quarterly" className="mt-0">
            <ChartContainer className="h-[300px]">
              <Chart data={performanceData.filter((_, i) => i % 3 === 0)}>
                <ChartGrid strokeDasharray="3 3" vertical={false} />
                <ChartXAxis dataKey="date" />
                <ChartYAxis />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLine
                  type="monotone"
                  dataKey="value"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={{ stroke: "hsl(var(--primary))", strokeWidth: 2, fill: "hsl(var(--primary))" }}
                />
              </Chart>
            </ChartContainer>
          </TabsContent>
          <TabsContent value="yearly" className="mt-0">
            <ChartContainer className="h-[300px]">
              <Chart data={performanceData}>
                <ChartGrid strokeDasharray="3 3" vertical={false} />
                <ChartXAxis dataKey="date" />
                <ChartYAxis />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLine
                  type="monotone"
                  dataKey="value"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={{ stroke: "hsl(var(--primary))", strokeWidth: 2, fill: "hsl(var(--primary))" }}
                />
              </Chart>
            </ChartContainer>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

