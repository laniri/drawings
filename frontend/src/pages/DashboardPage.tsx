import React from 'react'
import {
  Typography,
  Paper,
  Box,
  Grid,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Chip,
  Button,
  LinearProgress,
} from '@mui/material'
import {
  TrendingUp,
  Warning,
  CheckCircle,
  Image as ImageIcon,
  Analytics,
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import axios from 'axios'

interface DashboardStats {
  total_drawings: number
  total_analyses: number
  anomaly_count: number
  normal_count: number
  recent_analyses: Array<{
    id: number
    drawing_id: number
    filename: string
    age_years: number
    anomaly_score: number
    is_anomaly: boolean
    analysis_timestamp: string
  }>
  age_distribution: Array<{
    age_group: string
    count: number
  }>
  model_status: {
    vision_model: string
    is_loaded: boolean
    last_updated: string
    active_age_groups: number
  }
}



const DashboardPage: React.FC = () => {
  const navigate = useNavigate()

  const { data: stats, isLoading } = useQuery<DashboardStats>({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      const response = await axios.get('/api/analysis/stats')
      return response.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const anomalyRate = stats
    ? ((stats.anomaly_count / (stats.anomaly_count + stats.normal_count)) * 100).toFixed(1)
    : '0'

  const pieData = stats
    ? [
        { name: 'Normal', value: stats.normal_count, color: '#00C49F' },
        { name: 'Anomaly', value: stats.anomaly_count, color: '#FF8042' },
      ]
    : []

  if (isLoading) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Dashboard
        </Typography>
        <LinearProgress />
      </Box>
    )
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Dashboard
      </Typography>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Total Drawings
                  </Typography>
                  <Typography variant="h4">
                    {stats?.total_drawings || 0}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'primary.main' }}>
                  <ImageIcon />
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
                    Analyses Run
                  </Typography>
                  <Typography variant="h4">
                    {stats?.total_analyses || 0}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'secondary.main' }}>
                  <Analytics />
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
                    Anomalies Detected
                  </Typography>
                  <Typography variant="h4" color="warning.main">
                    {stats?.anomaly_count || 0}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'warning.main' }}>
                  <Warning />
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
                    Anomaly Rate
                  </Typography>
                  <Typography variant="h4" color="error.main">
                    {anomalyRate}%
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'error.main' }}>
                  <TrendingUp />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Age Distribution Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Age Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={stats?.age_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age_group" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#1976d2" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Analysis Results Pie Chart */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Analysis Results
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Model Status */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Model Status
            </Typography>
            {stats?.model_status && (
              <Box>
                <Box display="flex" alignItems="center" gap={1} mb={2}>
                  <Chip
                    icon={stats.model_status.is_loaded ? <CheckCircle /> : <Warning />}
                    label={stats.model_status.is_loaded ? 'Loaded' : 'Not Loaded'}
                    color={stats.model_status.is_loaded ? 'success' : 'warning'}
                    variant="outlined"
                  />
                  <Typography variant="body2">
                    {stats.model_status.vision_model.toUpperCase()} Model
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Active Age Groups: {stats.model_status.active_age_groups}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Last Updated: {new Date(stats.model_status.last_updated).toLocaleString()}
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Recent Analyses */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">
                Recent Analyses
              </Typography>
              <Button
                size="small"
                onClick={() => navigate('/upload')}
              >
                Upload New
              </Button>
            </Box>
            <List>
              {stats?.recent_analyses?.slice(0, 5).map((analysis) => (
                <ListItem
                  key={analysis.id}
                  button
                  onClick={() => navigate(`/analysis/${analysis.id}`)}
                >
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: analysis.is_anomaly ? 'warning.main' : 'success.main' }}>
                      {analysis.is_anomaly ? <Warning /> : <CheckCircle />}
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={analysis.filename}
                    secondary={
                      <Box>
                        <Typography variant="body2" component="span">
                          Age: {analysis.age_years} years • Score: {analysis.anomaly_score.toFixed(3)}
                        </Typography>
                        <br />
                        <Typography variant="caption" color="text.secondary">
                          {new Date(analysis.analysis_timestamp).toLocaleString()}
                        </Typography>
                      </Box>
                    }
                  />
                  <Chip
                    size="small"
                    label={analysis.is_anomaly ? 'Anomaly' : 'Normal'}
                    color={analysis.is_anomaly ? 'warning' : 'success'}
                    variant="outlined"
                  />
                </ListItem>
              )) || (
                <ListItem>
                  <ListItemText
                    primary="No analyses yet"
                    secondary="Upload a drawing to get started"
                  />
                </ListItem>
              )}
            </List>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}

export default DashboardPage