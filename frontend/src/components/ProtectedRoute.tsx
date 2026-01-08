import { useEffect, useState } from 'react'
import { Box, CircularProgress, Typography } from '@mui/material'

interface SessionStatus {
  authenticated: boolean
  session_info?: {
    created_at: string
    last_accessed: string
    client_ip: string
    is_admin: boolean
  }
  expires_in?: number
}

interface ProtectedRouteProps {
  children: React.ReactNode
}

/**
 * ProtectedRoute component that checks authentication before rendering children.
 * 
 * If the user is not authenticated, redirects to the login page with a return URL.
 * Shows a loading spinner while checking authentication status.
 */
export default function ProtectedRoute({ children }: ProtectedRouteProps) {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Check authentication status
    const checkAuth = async () => {
      try {
        const response = await fetch('/auth/status', {
          credentials: 'include', // Include cookies
        })

        if (response.ok) {
          const data: SessionStatus = await response.json()
          setIsAuthenticated(data.authenticated)

          // If not authenticated, redirect to login
          if (!data.authenticated) {
            const currentPath = window.location.pathname
            window.location.href = `/auth/login?redirect=${encodeURIComponent(currentPath)}`
          }
        } else {
          // If status check fails, assume not authenticated
          setIsAuthenticated(false)
          const currentPath = window.location.pathname
          window.location.href = `/auth/login?redirect=${encodeURIComponent(currentPath)}`
        }
      } catch (error) {
        console.error('Authentication check failed:', error)
        setIsAuthenticated(false)
        const currentPath = window.location.pathname
        window.location.href = `/auth/login?redirect=${encodeURIComponent(currentPath)}`
      } finally {
        setIsLoading(false)
      }
    }

    checkAuth()
  }, [])

  // Show loading spinner while checking authentication
  if (isLoading || isAuthenticated === null) {
    return (
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        minHeight="60vh"
        gap={2}
      >
        <CircularProgress size={60} />
        <Typography variant="body1" color="text.secondary">
          Checking authentication...
        </Typography>
      </Box>
    )
  }

  // If authenticated, render children
  if (isAuthenticated) {
    return <>{children}</>
  }

  // If not authenticated, show nothing (redirect is happening)
  return null
}
