"use client"

import { useState } from "react"
import Link from "next/link"
import { Bell, Menu, Search, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

export function DashboardHeader() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background">
      <div className="container flex h-12 items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" className="h-8 w-8 md:hidden" onClick={() => setIsMenuOpen(!isMenuOpen)}>
            <Menu className="h-4 w-4" />
            <span className="sr-only">Toggle menu</span>
          </Button>
          <Link href="/" className="flex items-center gap-2">
            <div className="relative h-6 w-6 overflow-hidden rounded-full bg-primary">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="h-3 w-3 rounded-full bg-background"></div>
              </div>
            </div>
            <span className="text-lg font-bold">NeuroFi</span>
          </Link>
        </div>

        <div className="hidden md:flex md:flex-1 md:items-center md:justify-center md:px-6">
          <div className="relative w-full max-w-md">
            <Search className="absolute left-2.5 top-2 h-3 w-3 text-muted-foreground" />
            <Input type="search" placeholder="Search stocks..." className="w-full h-7 pl-8 text-xs" />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <Bell className="h-4 w-4" />
            <span className="sr-only">Notifications</span>
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <User className="h-4 w-4" />
            <span className="sr-only">Account</span>
          </Button>
        </div>
      </div>

      {isMenuOpen && (
        <div className="container border-t px-4 py-2 md:hidden">
          <div className="relative mb-2">
            <Search className="absolute left-2.5 top-2 h-3 w-3 text-muted-foreground" />
            <Input type="search" placeholder="Search stocks..." className="w-full h-7 pl-8 text-xs" />
          </div>
          <nav className="flex flex-col space-y-2">
            <Link href="#" className="text-xs font-medium hover:text-primary">
              Dashboard
            </Link>
            <Link href="#" className="text-xs font-medium text-muted-foreground hover:text-primary">
              Portfolio
            </Link>
            <Link href="#" className="text-xs font-medium text-muted-foreground hover:text-primary">
              Analysis
            </Link>
          </nav>
        </div>
      )}
    </header>
  )
}

