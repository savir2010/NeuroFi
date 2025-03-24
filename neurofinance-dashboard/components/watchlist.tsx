"use client"

import { useState } from "react"
import { ArrowDown, ArrowUp, Maximize2, Minimize2, Star } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"

// Mock data for watchlist
const watchlistData = [
  { symbol: "AAPL", name: "Apple Inc.", price: 218.27, change: 4.17, changePercent: 1.95 },
  { symbol: "MSFT", name: "Microsoft Corp.", price: 391.26, change: 4.42, changePercent: 1.14 },
  { symbol: "GOOGL", name: "Alphabet Inc.", price: 166.25, change: 1.20, changePercent: 0.73 },
  { symbol: "AMZN", name: "Amazon.com Inc.", price: 196.21, change: 1.26, changePercent: 0.55 },
  { symbol: "TSLA", name: "Tesla Inc.", price: 248.71, change: 12.45, changePercent: 5.27 },
  { symbol: "META", name: "Meta Platforms Inc.", price: 596.25, change: 10.25, changePercent: 1.75 },
  { symbol: "NVDA", name: "NVIDIA Corp.", price: 117.70, change: -0.83, changePercent: -0.70},
  { symbol: "JPM", name: "JPMorgan Chase & Co.", price: 241.63, change: 2.62, changePercent: 1.10 },
  { symbol: "V", name: "Visa Inc.", price: 335.66, change: -3.84, changePercent: -1.13 },
  { symbol: "WMT", name: "Walmart Inc.", price: 85.98, change: 0.41, changePercent: 0.47 },
]

export function Watchlist() {
  const [favorites, setFavorites] = useState<string[]>([])
  const [isExpanded, setIsExpanded] = useState(false)
  const [searchTerm, setSearchTerm] = useState("")

  const toggleFavorite = (symbol: string) => {
    if (favorites.includes(symbol)) {
      setFavorites(favorites.filter((s) => s !== symbol))
    } else {
      setFavorites([...favorites, symbol])
    }
  }

  const toggleExpand = () => {
    setIsExpanded(!isExpanded)
  }

  const filteredStocks = watchlistData.filter(
    (stock) =>
      stock.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
      stock.name.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  return (
    <Card className={`h-full transition-all duration-300 ${isExpanded ? "fixed inset-4 z-50" : ""}`}>
      <CardHeader className="p-3 pb-2 flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-lg">Watchlist</CardTitle>
          <CardDescription>Track your favorite stocks</CardDescription>
        </div>
        <Button variant="ghost" size="icon" onClick={toggleExpand} className="h-8 w-8">
          {isExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          <span className="sr-only">{isExpanded ? "Minimize" : "Maximize"}</span>
        </Button>
      </CardHeader>
      <CardContent className="p-3 pt-2 overflow-hidden">
        <div className="mb-3 flex justify-between">
          <div className="w-full">
            <Input
              placeholder="Search stocks..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="h-8 text-sm w-full"
            />
          </div>
        </div>
        <div className="h-[280px] overflow-hidden">
          <ScrollArea className={`${isExpanded ? "h-[calc(100vh-180px)]" : "h-[280px]"}`}>
            <div className="space-y-3 pr-2">
              {filteredStocks.map((stock) => (
                <div
                  key={stock.symbol}
                  className="flex items-center justify-between rounded-md border p-3 transition-colors hover:bg-muted"
                >
                  <div className="flex items-center gap-3">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-muted-foreground hover:text-yellow-500"
                      onClick={() => toggleFavorite(stock.symbol)}
                    >
                      <Star
                        className={`h-4 w-4 ${favorites.includes(stock.symbol) ? "fill-yellow-500 text-yellow-500" : ""}`}
                      />
                      <span className="sr-only">Favorite</span>
                    </Button>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm">{stock.symbol}</span>
                        <span className="text-xs text-muted-foreground">{stock.name}</span>
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">${stock.price.toFixed(2)}</div>
                    <div className="flex items-center justify-end gap-1">
                      {stock.change > 0 ? (
                        <ArrowUp className="h-3 w-3 text-green-500" />
                      ) : (
                        <ArrowDown className="h-3 w-3 text-red-500" />
                      )}
                      <span className={`text-xs ${stock.change > 0 ? "text-green-500" : "text-red-500"}`}>
                        {stock.change > 0 ? "+" : ""}
                        {stock.change.toFixed(2)} ({stock.changePercent > 0 ? "+" : ""}
                        {stock.changePercent.toFixed(2)}%)
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </div>
      </CardContent>
    </Card>
  )
}

