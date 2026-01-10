import React, { useEffect, useState } from 'react'
import {
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Box,
  IconButton,
  Tooltip,
  Chip,
  BottomNavigation,
  BottomNavigationAction,
  Paper,
  useMediaQuery,
  useTheme,
} from '@mui/material'
import {
  Dashboard,
  Upload,
  BatchPrediction,
  Settings,
  Description,
  PlayArrow,
  Logout,
  Lock,
} from '@mui/icons-material'
import { useNavigate, useLocation } from 'react-router-dom'

const drawerWidth = 240

interface LayoutProps {
  children: React.ReactNode
}

interface SessionStatus {
  authenticated: boolean
  session_info?: {
    created_at: string
    last_accessed: string
    client_ip: string
    is_admin: boolean
  }
}

const menuItems = [
  { text: 'Demo', icon: <PlayArrow />, path: '/', protected: false },
  { text: 'Dashboard', icon: <Dashboard />, path: '/dashboard', protected: true },
  { text: 'Upload', icon: <Upload />, path: '/upload', protected: false },
  { text: 'Batch Processing', icon: <BatchPrediction />, path: '/batch', protected: true },
  { text: 'Configuration', icon: <Settings />, path: '/config', protected: true },
  { text: 'Documentation', icon: <Description />, path: '/documentation', protected: false },
]

// Mobile-only navigation items
const mobileMenuItems = [
  { text: 'Demo', icon: <PlayArrow />, path: '/' },
  { text: 'Documentation', icon: <Description />, path: '/documentation' },
  { text: 'Upload', icon: <Upload />, path: '/upload' },
]

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const navigate = useNavigate()
  const location = useLocation()
  const [authStatus, setAuthStatus] = useState<SessionStatus | null>(null)
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md')) // < 900px

  useEffect(() => {
    // Check authentication status
    const checkAuth = async () => {
      try {
        const response = await fetch('/auth/status', {
          credentials: 'include',
        })
        if (response.ok) {
          const data: SessionStatus = await response.json()
          setAuthStatus(data)
        }
      } catch (error) {
        console.error('Failed to check auth status:', error)
      }
    }

    checkAuth()
  }, [location.pathname]) // Re-check on route change

  const handleLogout = async () => {
    try {
      await fetch('/auth/logout', {
        method: 'POST',
        credentials: 'include',
      })
      // Redirect to home page
      window.location.href = '/'
    } catch (error) {
      console.error('Logout failed:', error)
    }
  }

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}
      >
        <Toolbar>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Children's Drawing Anomaly Detection
          </Typography>
          
          {/* Authentication status indicator */}
          {authStatus?.authenticated ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip
                label="Admin"
                size="small"
                color="success"
                icon={<Lock />}
              />
              <Tooltip title="Logout">
                <IconButton color="inherit" onClick={handleLogout}>
                  <Logout />
                </IconButton>
              </Tooltip>
            </Box>
          ) : null}
        </Toolbar>
      </AppBar>

      {/* Desktop Sidebar - Hidden on mobile */}
      {!isMobile && (
        <Drawer
          variant="permanent"
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: {
              width: drawerWidth,
              boxSizing: 'border-box',
            },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto' }}>
            <List>
              {menuItems.map((item) => (
                <ListItem key={item.text} disablePadding>
                  <ListItemButton
                    selected={location.pathname === item.path}
                    onClick={() => navigate(item.path)}
                  >
                    <ListItemIcon>
                      {item.icon}
                      {item.protected && !authStatus?.authenticated && (
                        <Lock sx={{ fontSize: 12, ml: -1, mt: -1 }} />
                      )}
                    </ListItemIcon>
                    <ListItemText 
                      primary={item.text}
                      secondary={item.protected && !authStatus?.authenticated ? 'Login required' : undefined}
                      secondaryTypographyProps={{ variant: 'caption' }}
                    />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>
      )}

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: isMobile ? 2 : 3,
          width: isMobile ? '100%' : { sm: `calc(100% - ${drawerWidth}px)` },
          pb: isMobile ? 10 : 3, // Extra padding at bottom for mobile nav
        }}
      >
        <Toolbar />
        {children}
      </Box>

      {/* Mobile Bottom Navigation - Only visible on mobile */}
      {isMobile && (
        <Paper
          sx={{
            position: 'fixed',
            bottom: 0,
            left: 0,
            right: 0,
            zIndex: (theme) => theme.zIndex.appBar,
          }}
          elevation={3}
        >
          <BottomNavigation
            value={location.pathname}
            onChange={(_event, newValue) => {
              navigate(newValue)
            }}
            showLabels
          >
            {mobileMenuItems.map((item) => (
              <BottomNavigationAction
                key={item.text}
                label={item.text}
                value={item.path}
                icon={item.icon}
                sx={{
                  '& .MuiBottomNavigationAction-label': {
                    fontSize: '0.75rem',
                  },
                  '& .MuiSvgIcon-root': {
                    fontSize: '2rem', // Large icons
                  },
                }}
              />
            ))}
          </BottomNavigation>
        </Paper>
      )}
    </Box>
  )
}

export default Layout
