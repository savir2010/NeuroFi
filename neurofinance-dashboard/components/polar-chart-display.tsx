"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Maximize2, Minimize2 } from "lucide-react"

interface EmotionData {
  emotion: string;
  value: number;
  color: string;
}

interface EmotionScores {
  Anxiety: number;
  Calmness: number;
  Sadness: number;
  Anger: number;
  Fear: number;
}

interface PolarChartDisplayProps {
  data?: EmotionScores; // Accept emotion_scores as a prop
}

const colorMap: Record<string, string> = {
  Anger: "#ef4444",
  Anxiety: "#8b5cf6",
  Fear: "#ec4899",
  Sadness: "#3b82f6",
  Calmness: "#10b981"
}

export function PolarChartDisplay({ data }: PolarChartDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [emotionData, setEmotionData] = useState<EmotionData[]>([])
  const [description, setDescription] = useState("")
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (data) {
      processData(data);
      setIsLoading(false);
    }
  }, [data]);

  const processData = (data: EmotionScores) => {
    // Convert emotion_scores to the format needed for the chart
    const processedData: EmotionData[] = Object.entries(data).map(([emotion, value]) => ({
      emotion,
      value: value, // Scale to 0-100 (assuming scores are 0-10)
      color: colorMap[emotion] || "#888888" // Default gray if no color is mapped
    }));

    setEmotionData(processedData);

    // Generate description based on the data
    const dominantEmotions = Object.entries(data)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 2)
      .map(([emotion]) => emotion.toLowerCase());

    const stressLevel = data.Anxiety > 70 ? "high" : data.Anxiety > 40 ? "moderate" : "low";
    
    setDescription(`${dominantEmotions.join(" and ")} with ${stressLevel} stress levels.`);
  };

  // Calculate chart dimensions and positions
  const centerX = 50;
  const centerY = 50;
  const maxRadius = 35;

  // Calculate points for the radar polygon
  const calculatePoint = (index: number, value: number) => {
    const angle = (Math.PI * 2 * index) / emotionData.length - Math.PI / 2;
    const radius = (value / 100) * maxRadius;
    return {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    };
  };

  const polygonPoints = emotionData
    .map((data, i) => {
      const point = calculatePoint(i, data.value);
      return `${point.x},${point.y}`;
    })
    .join(" ");

  // Calculate label positions
  const labelPositions = emotionData.map((data, i) => {
    const angle = (Math.PI * 2 * i) / emotionData.length - Math.PI / 2;
    const radius = maxRadius + 8;
    return {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
      anchor:
        angle > Math.PI / 2 && angle < (3 * Math.PI) / 2
          ? "end"
          : angle === Math.PI / 2 || angle === (3 * Math.PI) / 2
            ? "middle"
            : "start",
      emotion: data.emotion,
      value: data.value,
    };
  });

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <Card className={`h-full transition-all duration-300 ${isExpanded ? "fixed inset-4 z-50" : ""}`}>
      <CardHeader className="p-3 pb-0 flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-lg">Emotional State</CardTitle>
          <CardDescription>Analysis of emotional factors</CardDescription>
        </div>
        <Button variant="ghost" size="icon" onClick={toggleExpand} className="h-8 w-8">
          {isExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          <span className="sr-only">{isExpanded ? "Minimize" : "Maximize"}</span>
        </Button>
      </CardHeader>
      <CardContent className="p-0 pt-2">
        <div className="flex flex-col items-center h-full">
          {isLoading ? (
            <div className="flex items-center justify-center h-[250px] w-full">
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-purple-500"></div>
            </div>
          ) : (
            <>
              <div
                className={`${isExpanded ? "h-[calc(100vh-180px)]" : "h-[250px]"} w-full flex items-center justify-center`}
              >
                <div className={`relative ${isExpanded ? "h-[400px] w-[400px]" : "h-[200px] w-[200px]"}`}>
                  <svg className="absolute inset-0" viewBox="0 0 100 100" overflow="visible">
                    {/* Define gradients */}
                    <defs>
                      <linearGradient id="radarGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.8" />
                        <stop offset="100%" stopColor="#ec4899" stopOpacity="0.8" />
                      </linearGradient>
                      <radialGradient id="radarRadialGradient" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
                        <stop offset="0%" stopColor="rgba(139, 92, 246, 0.3)" />
                        <stop offset="100%" stopColor="rgba(236, 72, 153, 0.1)" />
                      </radialGradient>
                    </defs>

                    {/* Background circles */}
                    {[1, 0.75, 0.5, 0.25].map((ratio, i) => (
                      <circle
                        key={`circle-${i}`}
                        cx={centerX}
                        cy={centerY}
                        r={maxRadius * ratio}
                        fill="none"
                        stroke="hsl(var(--border))"
                        strokeWidth="0.3"
                        strokeOpacity="0.5"
                      />
                    ))}

                    {/* Spokes */}
                    {emotionData.map((_, i) => {
                      const angle = (Math.PI * 2 * i) / emotionData.length - Math.PI / 2;
                      const x = centerX + maxRadius * Math.cos(angle);
                      const y = centerY + maxRadius * Math.sin(angle);
                      return (
                        <line
                          key={`spoke-${i}`}
                          x1={centerX}
                          y1={centerY}
                          x2={x}
                          y2={y}
                          stroke="hsl(var(--border))"
                          strokeWidth="0.3"
                          strokeOpacity="0.5"
                        />
                      );
                    })}

                    {/* Data polygon with gradient fill */}
                    {emotionData.length > 0 && (
                      <polygon
                        points={polygonPoints}
                        fill="url(#radarRadialGradient)"
                        stroke="url(#radarGradient)"
                        strokeWidth="1.5"
                        strokeLinejoin="round"
                      />
                    )}

                    {/* Data points */}
                    {emotionData.map((data, i) => {
                      const point = calculatePoint(i, data.value);
                      return (
                        <circle
                          key={`point-${i}`}
                          cx={point.x}
                          cy={point.y}
                          r={isExpanded ? "2" : "1.5"}
                          fill="white"
                          stroke={data.color}
                          strokeWidth={isExpanded ? "1.5" : "1"}
                        />
                      );
                    })}

                    {/* Labels */}
                    {labelPositions.map((pos, i) => (
                      <g key={`label-${i}`}>
                        <text
                          x={pos.x}
                          y={pos.y}
                          textAnchor={pos.anchor}
                          dominantBaseline="middle"
                          fontSize={isExpanded ? "4.5" : "3.5"}
                          fontWeight="500"
                          fill="currentColor"
                        >
                          {pos.emotion}
                        </text>
                        <text
                          x={pos.x}
                          y={pos.y + 4}
                          textAnchor={pos.anchor}
                          dominantBaseline="middle"
                          fontSize={isExpanded ? "4" : "3"}
                          fill="currentColor"
                          opacity="0.7"
                        >
                          {pos.value.toFixed(0)}%
                        </text>
                      </g>
                    ))}
                  </svg>
                </div>
              </div>
              <div className="text-center px-4 mt-2">
                <p className="text-sm text-muted-foreground">{description}</p>
              </div>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}