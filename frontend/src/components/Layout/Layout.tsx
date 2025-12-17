import React from 'react'
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
} from '@mui/material'
import {
  Dashboard,
  Upload,
  BatchPrediction,
  Settings,
  Description,
} from '@mui/icons-material'
import { useNavigate, useLocation } from 'react-router-dom'

const drawerWidth = 240

interface LayoutProps {
  children: React.ReactNode
}

const menuItems = [
  { text: 'Dashboard', icon: <Dashboard />, path: '/' },
  { text: 'Upload', icon: <Upload />, path: '/upload' },
  { text: 'Batch Processing', icon: <BatchPrediction />, path: '/batch' },
  { text: 'Configuration', icon: <Settings />, path: '/config' },
  { text: 'Documentation', icon: <Description />, path: '/documentation' },
]

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}
      >
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            Children's Drawing Anomaly Detection
          </Typography>
        </Toolbar>
      </AppBar>
      
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
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
        }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  )
}

export default Layout