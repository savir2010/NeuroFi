import type React from "react"
interface DashboardShellProps {
  children: React.ReactNode
  className?: string
}

export function DashboardShell({ children, className = "" }: DashboardShellProps) {
  return (
    <main className={`container flex-1 px-4 pb-8 ${className}`}>
      <div className="flex flex-col gap-4">{children}</div>
    </main>
  )
}

