"use client"

import { useState, useEffect } from "react"
import { Send } from "lucide-react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

type Message = {
  id: number
  content: string
  sender: "user" | "advisor"
  timestamp: number | Date
}

export function FinancialAdvisor() {
  const [input, setInput] = useState("")
  const [messages, setMessages] = useState<Message[]>([])
  const [isTyping, setIsTyping] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  // Fetch initial messages on component mount
  useEffect(() => {
    const fetchMessages = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5020/api/messages')
        const data = await response.json()
        setMessages(data.messages || [])
      } catch (error) {
        console.error('Error fetching messages:', error)
        // Fallback to initial message if server is unavailable
        setMessages([{
          id: 1,
          content: "Hello! I'm your EEG-integrated financial advisor. How can I help?",
          sender: "advisor",
          timestamp: new Date(),
        }])
      } finally {
        setIsLoading(false)
      }
    }
    
    fetchMessages()
  }, [])

  const handleSendMessage = async () => {
    if (!input.trim()) return

    // Add user message to UI immediately for better UX
    const userMessage: Message = {
      id: messages.length + 1,
      content: input,
      sender: "user",
      timestamp: new Date(),
    }

    setMessages([...messages, userMessage])
    setInput("")
    setIsTyping(true)

    try {
      // Send message to server
      const response = await fetch('http://127.0.0.1:5020/api/send-message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      })

      if (!response.ok) {
        throw new Error('Network response was not ok')
      }

      const data = await response.json()
      
      // Update messages with the server response
      setMessages(prev => {
        // Remove the temporary user message (we'll use the server version)
        const withoutTemp = prev.filter(msg => msg.id !== userMessage.id)
        return [...withoutTemp, data.userMessage, data.advisorMessage]
      })
    } catch (error) {
      console.error('Error sending message:', error)
      
      // Fallback to client-side response if server fails
      const advisorResponses = [
        "Your EEG shows stress with tech stocks. Consider diversifying.",
        "Neural patterns show confidence in renewable energy stocks.",
        "EEG indicates you handle volatility well. Consider growth stocks.",
        "EEG shows decision fatigue. Schedule trades earlier in the day.",
      ]

      const advisorMessage: Message = {
        id: messages.length + 2,
        content: advisorResponses[Math.floor(Math.random() * advisorResponses.length)],
        sender: "advisor",
        timestamp: new Date(),
      }

      setMessages(prev => [...prev, advisorMessage])
    } finally {
      setIsTyping(false)
    }
  }

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="p-3 pb-2">
        <CardTitle className="text-lg">Financial Advisor</CardTitle>
        <CardDescription>EEG-integrated insights</CardDescription>
      </CardHeader>
      <CardContent className="flex-1 p-0 pt-2 overflow-hidden">
        <div className="h-[280px] overflow-hidden px-3">
          <ScrollArea className="h-[280px]">
            <div className="space-y-4 pt-1">
              {isLoading ? (
                <div className="flex justify-center items-center h-full">
                  <p className="text-sm text-muted-foreground">Loading conversation...</p>
                </div>
              ) : (
                <>
                  {messages.map((message) => (
                    <div key={message.id} className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}>
                      <div
                        className={`flex max-w-[80%] gap-2 ${message.sender === "user" ? "flex-row-reverse" : "flex-row"}`}
                      >
                        {message.sender === "advisor" && (
                          <Avatar className="h-8 w-8">
                            <AvatarImage src="/placeholder.svg?height=32&width=32" alt="Advisor" />
                            <AvatarFallback className="bg-primary text-primary-foreground text-xs">AI</AvatarFallback>
                          </Avatar>
                        )}
                        <div
                          className={`rounded-lg p-3 text-sm ${
                            message.sender === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                          }`}
                        >
                          {message.content}
                        </div>
                      </div>
                    </div>
                  ))}
                </>
              )}
              {isTyping && (
                <div className="flex justify-start">
                  <div className="flex max-w-[80%] gap-2">
                    <Avatar className="h-8 w-8">
                      <AvatarImage src="/placeholder.svg?height=32&width=32" alt="Advisor" />
                      <AvatarFallback className="bg-primary text-primary-foreground text-xs">AI</AvatarFallback>
                    </Avatar>
                    <div className="rounded-lg bg-muted p-3">
                      <div className="flex space-x-1">
                        <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground"></div>
                        <div
                          className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground"
                          style={{ animationDelay: "0.2s" }}
                        ></div>
                        <div
                          className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground"
                          style={{ animationDelay: "0.4s" }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
      </CardContent>
      <CardFooter className="border-t p-3">
        <div className="flex w-full items-center gap-2">
          <Input
            placeholder="Ask for advice..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
            className="h-10 text-sm"
            disabled={isLoading}
          />
          <Button 
            size="icon" 
            className="h-10 w-10" 
            onClick={handleSendMessage}
            disabled={isLoading || !input.trim()}
          >
            <Send className="h-4 w-4" />
            <span className="sr-only">Send message</span>
          </Button>
        </div>
      </CardFooter>
    </Card>
  )
}