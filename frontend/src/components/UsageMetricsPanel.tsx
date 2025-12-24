import React from 'react'
import {
  Typography,
  Paper,
  Box,
  Grid,
  Card,
  CardContent,
  Avatar,
  Chip,
  LinearProgress,
  Divider,
} from '@mui/material'
import {
  People,
  Memory,
  Speed,
  Public,
  TrendingUp,
  AccessTime,
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts'
import axios from 'axios'

interface UsageMetrics {
  timestamp: string
  database: {
    total_drawings: number
    total_analyses: number
    anomaly_count: number
    normal_count: number
    recent_analyses_count: number
    age_groups_count: number
  }
  time_based: {
    daily_analyses: number
    weekly_analyses: number
    monthly_analyses: number
  }
  sessions: {
    active_sessions: number
    total_page_views: number
    total_session_analyses: number
  }
  system_health: {
    uptime_seconds: number
    uptime_percentage: number
    total_requests: number
    successful_requests: number
    failed_requests: number
    error_rate: number
    average_response_time: number
    memory_usage_mb: number
    cpu_usage_percent: number
    average_processing_time: number
  }
  geographic: Record<string, number>
  uptime_seconds: number
}

interface PerformanceMetrics {
  status: string
  analysis: {
    total_analyses: number
    average_processing_time: number
    recent_processing_times: number[]
    anomaly_count: number
    normal_count: number
  }
  system: {
    total_requests: number
    successful_requests: number
    failed_requests: number
    error_rate: number
    average_response_time: number
    recent_response_times: number[]
    memory_usage_mb: number
    cpu_usage_percent: number
  }
}

const UsageMetricsPanel: React.FC = () => {
  const { data: metrics, isLoading: metricsLoading } = useQuery<UsageMetrics>({
    queryKey: ['usage-metrics'],
    queryFn: async () => {
      const response = await axios.get('/api/v1/metrics/usage')
      return response.data.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const { data: performance, isLoading: performanceLoading } = useQuery<PerformanceMetrics>({
    queryKey: ['performance-metrics'],
    queryFn: async () => {
      const response = await axios.get('/api/v1/metrics/performance')
      return response.data
    },
    refetchInterval: 15000, // Refresh every 15 seconds
  })

  const formatUptime = (seconds: number): string => {
    const days = Math.floor(seconds / 86400)
    const hours = Math.floor((seconds % 86400) / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    
    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`
    } else if (hours > 0) {
      return `${hours}h ${minutes}m`
    } else {
      return `${minutes}m`
    }
  }

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes.toFixed(1)} MB`
    return `${(bytes / 1024).toFixed(1)} GB`
  }

  // Prepare chart data for processing times
  const processingTimeData = performance?.analysis.recent_processing_times.map((time, index) => ({
    index: index + 1,
    time: time * 1000, // Convert to milliseconds
  })) || []

  // Prepare chart data for response times
  const responseTimeData = performance?.system.recent_response_times.map((time, index) => ({
    index: index + 1,
    time: time * 1000, // Convert to milliseconds
  })) || []

  if (metricsLoading || performanceLoading) {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Usage Metrics
        </Typography>
        <LinearProgress />
      </Paper>
    )
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Usage Metrics & System Health
      </Typography>

      {/* Real-time Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Active Sessions
                  </Typography>
                  <Typography variant="h4">
                    {metrics?.sessions.active_sessions || 0}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'primary.main' }}>
                  <People />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    System Uptime
                  </Typography>
                  <Typography variant="h6">
                    {metrics ? formatUptime(metrics.uptime_seconds) : '0m'}
                  </Typography>
                  <Typography variant="body2" color="success.main">
                    {metrics?.system_health.uptime_percentage.toFixed(1)}% today
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'success.main' }}>
                  <AccessTime />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Memory Usage
                  </Typography>
                  <Typography variant="h6">
                    {metrics ? formatBytes(metrics.system_health.memory_usage_mb) : '0 MB'}
                  </Typography>
                  <Typography variant="body2" color="info.main">
                    CPU: {metrics?.system_health.cpu_usage_percent.toFixed(1)}%
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'info.main' }}>
                  <Memory />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Error Rate
                  </Typography>
                  <Typography variant="h4" color={(metrics?.system_health.error_rate || 0) > 0.05 ? 'error.main' : 'success.main'}>
                    {((metrics?.system_health.error_rate || 0) * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: (metrics?.system_health.error_rate || 0) > 0.05 ? 'error.main' : 'success.main' }}>
                  <TrendingUp />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Time-based Analysis Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Analysis Activity
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={4}>
                <Box textAlign="center">
                  <Typography variant="h4" color="primary.main">
                    {metrics?.time_based.daily_analyses || 0}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Today
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box textAlign="center">
                  <Typography variant="h4" color="secondary.main">
                    {metrics?.time_based.weekly_analyses || 0}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    This Week
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box textAlign="center">
                  <Typography variant="h4" color="success.main">
                    {metrics?.time_based.monthly_analyses || 0}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    This Month
                  </Typography>
                </Box>
              </Grid>
            </Grid>
            
            <Divider sx={{ my: 2 }} />
            
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="body2" color="textSecondary">
                Avg Processing Time
              </Typography>
              <Chip
                icon={<Speed />}
                label={`${(metrics?.system_health.average_processing_time || 0).toFixed(2)}s`}
                color="info"
                variant="outlined"
                size="small"
              />
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Geographic Distribution
            </Typography>
            {metrics?.geographic && Object.keys(metrics.geographic).length > 0 ? (
              <Box>
                {Object.entries(metrics.geographic)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 5)
                  .map(([location, count]) => (
                    <Box key={location} display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Box display="flex" alignItems="center" gap={1}>
                        <Public fontSize="small" color="action" />
                        <Typography variant="body2">
                          {location}
                        </Typography>
                      </Box>
                      <Chip
                        label={count}
                        size="small"
                        variant="outlined"
                      />
                    </Box>
                  ))}
              </Box>
            ) : (
              <Typography variant="body2" color="textSecondary" textAlign="center" py={2}>
                No active sessions
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Performance Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Processing Times
            </Typography>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={processingTimeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="index" />
                <YAxis />
                <Tooltip formatter={(value) => [`${(value as number).toFixed(0)}ms`, 'Processing Time']} />
                <Line type="monotone" dataKey="time" stroke="#1976d2" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Response Times
            </Typography>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={responseTimeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="index" />
                <YAxis />
                <Tooltip formatter={(value) => [`${(value as number).toFixed(0)}ms`, 'Response Time']} />
                <Area type="monotone" dataKey="time" stroke="#ff9800" fill="#ff9800" fillOpacity={0.3} />
              </AreaChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* System Health Summary */}
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          System Health Summary
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <Box textAlign="center">
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Total Requests
              </Typography>
              <Typography variant="h5">
                {metrics?.system_health.total_requests || 0}
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Box textAlign="center">
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Success Rate
              </Typography>
              <Typography variant="h5" color="success.main">
                {metrics ? (((metrics.system_health.successful_requests / metrics.system_health.total_requests) * 100) || 0).toFixed(1) : 0}%
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Box textAlign="center">
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Avg Response Time
              </Typography>
              <Typography variant="h5">
                {((metrics?.system_health.average_response_time || 0) * 1000).toFixed(0)}ms
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Box textAlign="center">
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Page Views
              </Typography>
              <Typography variant="h5">
                {metrics?.sessions.total_page_views || 0}
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  )
}

export default UsageMetricsPanel