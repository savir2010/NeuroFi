"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Loader2, Search, AlertTriangle, BarChart2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"

export function NewsSection() {
  const [newsUrl, setNewsUrl] = useState("")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  type AnalysisResult = {
    title: string
    source: string
    date: string
    sentiment: "positive" | "negative" | "neutral"
    sentimentScore: number
    marketImpact: "high" | "medium" | "low"
    keyPoints: string[]
    biasDetection: {
      level: "high" | "medium" | "low"
      explanation: string
    }
    relatedStocks: Array<{
      symbol: string
      impact: "positive" | "negative" | "neutral"
    }>
  }

  const mockAnalysisResults: AnalysisResult[] = [
    {
      title: "Fed Signals Potential Rate Cut in Coming Months",
      source: "Financial Times",
      date: "2023-03-22",
      sentiment: "positive",
      sentimentScore: 0.78,
      marketImpact: "high",
      keyPoints: [
        "Federal Reserve indicates openness to rate cuts",
        "Inflation showing signs of cooling",
        "Markets responded positively to the announcement",
        "Analysts predict first cut could come by September",
      ],
      biasDetection: {
        level: "low",
        explanation: "Article presents balanced view with multiple expert opinions",
      },
      relatedStocks: [
        { symbol: "JPM", impact: "positive" },
        { symbol: "GS", impact: "positive" },
        { symbol: "MS", impact: "positive" },
      ],
    },
    {
      title: "Tech Layoffs Continue as Industry Giants Restructure",
      source: "TechCrunch",
      date: "2023-03-21",
      sentiment: "negative",
      sentimentScore: -0.65,
      marketImpact: "medium",
      keyPoints: [
        "Major tech companies announced new rounds of layoffs",
        "Cost-cutting measures in response to economic uncertainty",
        "AI implementation cited as factor in workforce reduction",
        "Tech sector unemployment rate rising",
      ],
      biasDetection: {
        level: "medium",
        explanation: "Article focuses heavily on negative aspects with limited discussion of industry adaptation",
      },
      relatedStocks: [
        { symbol: "MSFT", impact: "negative" },
        { symbol: "GOOGL", impact: "negative" },
        { symbol: "META", impact: "negative" },
      ],
    },
  ]

  const handleAnalyze = () => {
    if (!newsUrl.trim()) {
      setError("Please enter a news URL to analyze")
      return
    }

    setIsAnalyzing(true)
    setError(null)

    // Simulate API call with timeout
    setTimeout(() => {
      setIsAnalyzing(false)

      // For demo purposes, choose a random mock result
      const randomResult = mockAnalysisResults[Math.floor(Math.random() * mockAnalysisResults.length)]
      setAnalysisResult(randomResult)
    }, 2000)
  }

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case "positive":
        return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400"
      case "negative":
        return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400"
      case "neutral":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400"
      default:
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400"
    }
  }

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case "high":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400"
      case "medium":
        return "bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400"
      case "low":
        return "bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400"
      default:
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400"
    }
  }

  return (
    <Card className="h-full">
      <CardHeader className="p-3 pb-2">
        <CardTitle className="text-lg">News Analyzer</CardTitle>
        <CardDescription>Analyze financial news for market insights</CardDescription>
      </CardHeader>
      <CardContent className="p-3 pt-2 overflow-hidden">
        <div className="flex gap-2 mb-3">
          <Input
            placeholder="Paste news article URL..."
            value={newsUrl}
            onChange={(e) => setNewsUrl(e.target.value)}
            className="h-8 text-sm"
          />
          <Button size="sm" className="h-8 whitespace-nowrap" onClick={handleAnalyze} disabled={isAnalyzing}>
            {isAnalyzing ? (
              <>
                <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                Analyzing
              </>
            ) : (
              <>
                <Search className="mr-2 h-3 w-3" />
                Analyze
              </>
            )}
          </Button>
        </div>

        <div className="h-[280px] overflow-hidden">
          <ScrollArea className="h-[280px]">
            {error && (
              <div className="flex items-center gap-2 p-3 mb-3 rounded-md bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400">
                <AlertTriangle className="h-4 w-4" />
                <p className="text-sm">{error}</p>
              </div>
            )}

            {!analysisResult && !error && !isAnalyzing && (
              <div className="flex flex-col items-center justify-center h-full text-center p-4">
                <BarChart2 className="h-12 w-12 mb-2 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  Enter a news article URL and click Analyze to get AI-powered insights
                </p>
              </div>
            )}

            {analysisResult && (
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium mb-1">{analysisResult.title}</h3>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-muted-foreground">
                      {analysisResult.source} • {analysisResult.date}
                    </span>
                    <div className="flex gap-1">
                      <Badge
                        variant="secondary"
                        className={`text-xs px-2 py-0 h-5 ${getSentimentColor(analysisResult.sentiment)}`}
                      >
                        {analysisResult.sentiment}
                      </Badge>
                      <Badge
                        variant="secondary"
                        className={`text-xs px-2 py-0 h-5 ${getImpactColor(analysisResult.marketImpact)}`}
                      >
                        {analysisResult.marketImpact} impact
                      </Badge>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-medium mb-1">Key Points</h4>
                  <ul className="space-y-1">
                    {analysisResult.keyPoints.map((point, i) => (
                      <li key={i} className="text-xs flex items-start gap-1">
                        <span className="text-primary mt-0.5">•</span>
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="text-xs font-medium mb-1">Bias Detection</h4>
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant="outline" className="text-xs px-2 py-0 h-5">
                      {analysisResult.biasDetection.level} bias
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">{analysisResult.biasDetection.explanation}</p>
                </div>

                <div>
                  <h4 className="text-xs font-medium mb-1">Related Stocks</h4>
                  <div className="flex flex-wrap gap-1">
                    {analysisResult.relatedStocks.map((stock) => (
                      <Badge
                        key={stock.symbol}
                        variant="outline"
                        className={`text-xs px-2 py-0 h-5 ${
                          stock.impact === "positive"
                            ? "border-green-500 text-green-600"
                            : stock.impact === "negative"
                              ? "border-red-500 text-red-600"
                              : "border-blue-500 text-blue-600"
                        }`}
                      >
                        {stock.symbol}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </ScrollArea>
        </div>
      </CardContent>
    </Card>
  )
}

