"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

interface DonutChartProps {
  title: string
  description: string
  data: Array<{ name: string; value: number; color: string }>
  height?: number
  hideBottomText?: boolean
}

export function DonutChart({ title, description, data, height = 280, hideBottomText = false }: DonutChartProps) {
  const total = data.reduce((sum, item) => sum + item.value, 0)

  // Calculate the circumference of the circle
  const radius = 35 // Increased from 30 to make chart larger
  const circumference = 2 * Math.PI * radius

  // Calculate the stroke-dasharray and stroke-dashoffset for each segment
  let accumulatedOffset = 0
  const segments = data.map((item) => {
    const segmentLength = (item.value / total) * circumference
    const segment = {
      length: segmentLength,
      offset: accumulatedOffset,
      color: item.color,
      name: item.name,
      value: item.value,
    }
    accumulatedOffset += segmentLength
    return segment
  })

  return (
    <Card className="h-full">
      <CardHeader className="p-4 pb-0">
        <CardTitle className="text-lg">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="p-0 pt-2 flex items-center justify-center h-[calc(100%-60px)]">
        <div className="flex flex-col items-center">
          <div className="relative h-[160px] w-[160px]">
            <svg viewBox="0 0 100 100" className="absolute inset-0">
              {/* Background circle */}
              <circle
                cx="50"
                cy="50"
                r={radius}
                fill="none"
                stroke="hsl(var(--border))"
                strokeWidth="8"
                opacity="0.2"
              />

              {/* Donut segments */}
              {segments.map((segment, i) => (
                <circle
                  key={`segment-${i}`}
                  cx="50"
                  cy="50"
                  r={radius}
                  fill="none"
                  stroke={segment.color}
                  strokeWidth="8"
                  strokeDasharray={circumference}
                  strokeDashoffset={circumference - segment.offset}
                  transform="rotate(-90 50 50)"
                  strokeLinecap="round"
                />
              ))}

              {/* Center text */}
              <text
                x="50"
                y="50"
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="14"
                fontWeight="bold"
                fill="currentColor"
              >
                {data[0].value}%
              </text>
            </svg>
          </div>

          {/* Legend - only show if hideBottomText is false */}
          {!hideBottomText && (
            <div className="mt-4 w-full px-4">
              <div className="grid grid-cols-1 gap-2">
                {data.map((item, i) => (
                  <div key={`legend-${i}`} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="h-3 w-3 rounded-full" style={{ backgroundColor: item.color }} />
                      <span className="text-sm">{item.name}</span>
                    </div>
                    <span className="text-sm font-medium">{item.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

