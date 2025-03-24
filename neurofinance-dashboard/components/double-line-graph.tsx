"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Maximize2, Minimize2 } from "lucide-react"
import Image from "next/image"

export function DoubleLineGraph() {
  const [isExpanded, setIsExpanded] = useState(false)

  const toggleExpand = () => {
    setIsExpanded(!isExpanded)
  }

  return (
    <Card className={`h-full transition-all duration-300 ${isExpanded ? "fixed inset-4 z-50" : ""}`}>
      <CardHeader className="p-3 pb-0 flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-lg">Stress & Boredom Trends</CardTitle>
          <CardDescription>Annual tracking</CardDescription>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-4 text-sm mr-2">
            <div className="flex items-center gap-1">
              <div className="h-3 w-3 rounded-full bg-red-500" />
              <span>Stress</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="h-3 w-3 rounded-full bg-blue-500" />
              <span>Boredom</span>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={toggleExpand} className="h-8 w-8">
            {isExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            <span className="sr-only">{isExpanded ? "Minimize" : "Maximize"}</span>
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-3 flex items-center justify-center">
        <div className={`relative ${isExpanded ? "h-[calc(100vh-180px)]" : "h-[300px]"} w-full`}>
          {/* Using img tag instead of Next.js Image component */}
          <img
            src="/graph.png"
            alt="Stress & Boredom Trends Graph"
            className="w-full h-full object-contain"
          />
        </div>
      </CardContent>
    </Card>
  )
}