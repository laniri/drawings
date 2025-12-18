import { useState, useEffect } from 'react'
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  Button,
  Alert,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel
} from '@mui/material'
import {
  Timeline as TimelineIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon,
  Assessment as AssessmentIcon,
  Visibility as ViewIcon,
  School as SchoolIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material'
import { XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Area, AreaChart, Line } from 'recharts'

interface HistoricalAnalysis {
  id: number
  drawing_id: number
  drawing_filename: string
  anomaly_score: number
  normalized_score: number
  is_anomaly: boolean
  confidence: number
  age_group: string
  analysis_timestamp: string
  child_age_at_time: number
  interpretability_available: boolean
}

interface LongitudinalPattern {
  pattern_type: 'improvement' | 'decline' | 'stable' | 'fluctuating'
  confidence: number
  description: string
  recommendations: string[]
  key_observations: string[]
}

interface DevelopmentalMilestone {
  age_range: string
  expected_patterns: string[]
  observed_patterns: string[]
  alignment_score: number
  concerns: string[]
}

interface HistoricalInterpretationTrackerProps {
  drawingId: number
  currentAnalysis: {
    id: number
    normalized_score: number
    is_anomaly: boolean
    analysis_timestamp: string
  }
  onAnalysisSelect?: (analysisId: number) => void
}

export default function HistoricalInterpretationTracker({
  drawingId,
  currentAnalysis,
  onAnalysisSelect
}: HistoricalInterpretationTrackerProps) {
  const [historicalData, setHistoricalData] = useState<HistoricalAnalysis[]>([])
  const [longitudinalPattern, setLongitudinalPattern] = useState<LongitudinalPattern | null>(null)
  const [developmentalMilestones, setDevelopmentalMilestones] = useState<DevelopmentalMilestone[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedAnalysis, setSelectedAnalysis] = useState<HistoricalAnalysis | null>(null)
  const [detailDialogOpen, setDetailDialogOpen] = useState(false)
  const [timeRange, setTimeRange] = useState<'all' | '6months' | '1year' | '2years'>('all')
  const [showTrendLine, setShowTrendLine] = useState(true)
  const [viewMode, setViewMode] = useState<'timeline' | 'chart' | 'milestones'>('timeline')

  useEffect(() => {
    loadHistoricalData()
  }, [drawingId, timeRange])

  const loadHistoricalData = async () => {
    try {
      setLoading(true)
      setError(null)

      // Load historical analyses for this drawing
      const response = await fetch(`/api/analysis/drawing/${drawingId}`)
      
      if (!response.ok) {
        throw new Error('Failed to load historical data')
      }

      const data = await response.json()
      
      // Transform the data to include additional metadata
      const enrichedData: HistoricalAnalysis[] = data.analyses.map((analysis: any, index: number) => ({
        ...analysis,
        drawing_filename: `drawing_${drawingId}`,
        child_age_at_time: 5 + (index * 0.5), // Mock progressive age
        interpretability_available: true
      }))

      // Filter by time range
      const filteredData = filterByTimeRange(enrichedData)
      setHistoricalData(filteredData)

      // Generate longitudinal pattern analysis
      if (filteredData.length > 1) {
        const pattern = analyzeLongitudinalPattern(filteredData)
        setLongitudinalPattern(pattern)
      }

      // Generate developmental milestones
      const milestones = generateDevelopmentalMilestones(filteredData)
      setDevelopmentalMilestones(milestones)

    } catch (err) {
      console.error('Error loading historical data:', err)
      setError(err instanceof Error ? err.message : 'Failed to load historical data')
    } finally {
      setLoading(false)
    }
  }

  const filterByTimeRange = (data: HistoricalAnalysis[]) => {
    if (timeRange === 'all') return data

    const now = new Date()
    const cutoffDate = new Date()
    
    switch (timeRange) {
      case '6months':
        cutoffDate.setMonth(now.getMonth() - 6)
        break
      case '1year':
        cutoffDate.setFullYear(now.getFullYear() - 1)
        break
      case '2years':
        cutoffDate.setFullYear(now.getFullYear() - 2)
        break
    }

    return data.filter(item => new Date(item.analysis_timestamp) >= cutoffDate)
  }

  const analyzeLongitudinalPattern = (data: HistoricalAnalysis[]): LongitudinalPattern => {
    const scores = data.map(d => d.normalized_score)
    const firstScore = scores[0]
    const lastScore = scores[scores.length - 1]
    const scoreDiff = lastScore - firstScore

    // Calculate trend
    let pattern_type: LongitudinalPattern['pattern_type']
    let confidence = 0.7
    let description = ''
    let recommendations: string[] = []
    let key_observations: string[] = []

    if (Math.abs(scoreDiff) < 5) {
      pattern_type = 'stable'
      description = 'Anomaly scores have remained relatively stable over time, indicating consistent developmental patterns.'
      recommendations = [
        'Continue regular monitoring',
        'Maintain current intervention strategies if any',
        'Document any environmental or developmental changes'
      ]
      key_observations = [
        `Score variation within ${Math.abs(scoreDiff).toFixed(1)} points`,
        'Consistent pattern maintenance',
        'No significant developmental shifts detected'
      ]
    } else if (scoreDiff < -10) {
      pattern_type = 'improvement'
      description = 'Anomaly scores show a positive trend with decreasing anomaly indicators over time.'
      recommendations = [
        'Continue current interventions as they appear effective',
        'Document successful strategies for future reference',
        'Consider gradual reduction in intervention intensity if appropriate'
      ]
      key_observations = [
        `${Math.abs(scoreDiff).toFixed(1)} point improvement in scores`,
        'Positive developmental trajectory',
        'Intervention effectiveness indicated'
      ]
    } else if (scoreDiff > 10) {
      pattern_type = 'decline'
      description = 'Anomaly scores show an increasing trend, suggesting emerging or worsening developmental concerns.'
      recommendations = [
        'Increase monitoring frequency',
        'Consider additional assessment or intervention',
        'Consult with developmental specialists',
        'Review environmental factors that may be contributing'
      ]
      key_observations = [
        `${scoreDiff.toFixed(1)} point increase in anomaly scores`,
        'Concerning developmental trajectory',
        'May require intervention adjustment'
      ]
    } else {
      pattern_type = 'fluctuating'
      description = 'Anomaly scores show variable patterns with both increases and decreases over time.'
      recommendations = [
        'Investigate factors contributing to variability',
        'Consider more frequent assessment periods',
        'Document contextual factors during each assessment'
      ]
      key_observations = [
        'Variable score patterns observed',
        'Multiple factors may be influencing development',
        'Requires careful contextual analysis'
      ]
    }

    // Adjust confidence based on data quality
    if (data.length >= 5) confidence = Math.min(0.9, confidence + 0.1)
    if (data.every(d => d.confidence > 0.7)) confidence = Math.min(0.95, confidence + 0.1)

    return {
      pattern_type,
      confidence,
      description,
      recommendations,
      key_observations
    }
  }

  const generateDevelopmentalMilestones = (data: HistoricalAnalysis[]): DevelopmentalMilestone[] => {
    // Group data by age ranges
    const ageGroups = new Map<string, HistoricalAnalysis[]>()
    
    data.forEach(item => {
      const ageRange = `${Math.floor(item.child_age_at_time)}-${Math.floor(item.child_age_at_time) + 1}`
      if (!ageGroups.has(ageRange)) {
        ageGroups.set(ageRange, [])
      }
      ageGroups.get(ageRange)!.push(item)
    })

    return Array.from(ageGroups.entries()).map(([ageRange, analyses]) => {
      const avgScore = analyses.reduce((sum, a) => sum + a.normalized_score, 0) / analyses.length
      const hasAnomalies = analyses.some(a => a.is_anomaly)

      // Define expected patterns based on age
      const age = parseInt(ageRange.split('-')[0])
      let expected_patterns: string[] = []
      let observed_patterns: string[] = []
      let alignment_score = 0.8
      let concerns: string[] = []

      if (age <= 4) {
        expected_patterns = [
          'Simple geometric shapes',
          'Basic spatial relationships',
          'Emerging symbolic representation'
        ]
      } else if (age <= 6) {
        expected_patterns = [
          'More complex drawings with details',
          'Human figures with body parts',
          'Improved spatial organization'
        ]
      } else {
        expected_patterns = [
          'Detailed and proportional drawings',
          'Advanced spatial relationships',
          'Creative and expressive elements'
        ]
      }

      // Analyze observed patterns
      if (avgScore < 30) {
        observed_patterns = ['Typical developmental patterns', 'Age-appropriate complexity', 'Normal variation range']
        alignment_score = 0.9
      } else if (avgScore < 60) {
        observed_patterns = ['Some atypical elements', 'Mixed developmental indicators', 'Requires monitoring']
        alignment_score = 0.6
        concerns = ['Mild developmental variations noted']
      } else {
        observed_patterns = ['Significant atypical patterns', 'Developmental concerns present', 'Intervention may be needed']
        alignment_score = 0.3
        concerns = ['Significant developmental concerns', 'Professional assessment recommended']
      }

      if (hasAnomalies) {
        concerns.push('Anomalous patterns detected in this age range')
        alignment_score = Math.max(0.1, alignment_score - 0.2)
      }

      return {
        age_range: ageRange,
        expected_patterns,
        observed_patterns,
        alignment_score,
        concerns
      }
    })
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    })
  }

  const getTrendIcon = (current: number, previous?: number) => {
    if (!previous) return <TimelineIcon />
    
    const diff = current - previous
    if (diff > 5) return <TrendingUpIcon color="error" />
    if (diff < -5) return <TrendingDownIcon color="success" />
    return <TrendingFlatIcon color="action" />
  }

  const getPatternColor = (pattern: LongitudinalPattern['pattern_type']) => {
    switch (pattern) {
      case 'improvement': return '#4caf50'
      case 'decline': return '#f44336'
      case 'stable': return '#2196f3'
      case 'fluctuating': return '#ff9800'
      default: return '#757575'
    }
  }

  const chartData = historicalData.map((item) => ({
    date: formatTimestamp(item.analysis_timestamp),
    score: item.normalized_score,
    age: item.child_age_at_time,
    isAnomaly: item.is_anomaly,
    confidence: item.confidence * 100
  }))

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
        <CircularProgress />
      </Box>
    )
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <TimelineIcon sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6">
            Historical Analysis Tracking
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value as any)}
              label="Time Range"
            >
              <MenuItem value="all">All Time</MenuItem>
              <MenuItem value="6months">6 Months</MenuItem>
              <MenuItem value="1year">1 Year</MenuItem>
              <MenuItem value="2years">2 Years</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>View</InputLabel>
            <Select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value as any)}
              label="View"
            >
              <MenuItem value="timeline">Timeline</MenuItem>
              <MenuItem value="chart">Chart</MenuItem>
              <MenuItem value="milestones">Milestones</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {historicalData.length === 0 ? (
        <Alert severity="info">
          No historical analysis data available for this drawing
        </Alert>
      ) : (
        <Box>
          {/* Longitudinal Pattern Summary */}
          {longitudinalPattern && (
            <Paper sx={{ p: 2, mb: 3, backgroundColor: '#f8f9fa' }}>
              <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <AssessmentIcon sx={{ mr: 1, color: getPatternColor(longitudinalPattern.pattern_type) }} />
                Longitudinal Pattern Analysis
                <Chip
                  label={longitudinalPattern.pattern_type.toUpperCase()}
                  size="small"
                  sx={{
                    ml: 2,
                    backgroundColor: getPatternColor(longitudinalPattern.pattern_type),
                    color: 'white'
                  }}
                />
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                {longitudinalPattern.description}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Confidence: {(longitudinalPattern.confidence * 100).toFixed(0)}%
              </Typography>
            </Paper>
          )}

          {/* Timeline View */}
          {viewMode === 'timeline' && (
            <Paper sx={{ p: 2 }}>
              <Box>
                {historicalData.map((item, index) => {
                  const isLatest = item.id === currentAnalysis.id
                  const previousScore = index > 0 ? historicalData[index - 1].normalized_score : undefined
                  
                  return (
                    <Box key={item.id} sx={{ display: 'flex', mb: 2 }}>
                      <Box sx={{ minWidth: 150, pr: 2 }}>
                        <Typography variant="body2" color="text.secondary">
                          {formatTimestamp(item.analysis_timestamp)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Age: {item.child_age_at_time} years
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
                        <Box
                          sx={{
                            width: 40,
                            height: 40,
                            borderRadius: '50%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            backgroundColor: item.is_anomaly ? 'error.main' : 'success.main',
                            color: 'white',
                            border: isLatest ? '3px solid #1976d2' : 'none'
                          }}
                        >
                          {getTrendIcon(item.normalized_score, previousScore)}
                        </Box>
                        {index < historicalData.length - 1 && (
                          <Box
                            sx={{
                              width: 2,
                              height: 60,
                              backgroundColor: 'divider',
                              ml: 2,
                              mt: 4
                            }}
                          />
                        )}
                      </Box>
                      <Box sx={{ flex: 1 }}>
                        <Card
                          sx={{
                            cursor: 'pointer',
                            border: isLatest ? '2px solid #1976d2' : '1px solid #e0e0e0',
                            '&:hover': { boxShadow: 2 }
                          }}
                          onClick={() => {
                            setSelectedAnalysis(item)
                            setDetailDialogOpen(true)
                          }}
                        >
                          <CardContent sx={{ pb: '16px !important' }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                              <Typography variant="subtitle2">
                                Analysis #{item.id}
                                {isLatest && <Chip label="Current" size="small" sx={{ ml: 1 }} />}
                              </Typography>
                              <Chip
                                label={item.is_anomaly ? 'Anomaly' : 'Normal'}
                                color={item.is_anomaly ? 'error' : 'success'}
                                size="small"
                              />
                            </Box>
                            <Typography variant="body2" color="text.secondary">
                              Score: {item.normalized_score.toFixed(1)}/100
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Confidence: {(item.confidence * 100).toFixed(0)}%
                            </Typography>
                          </CardContent>
                        </Card>
                      </Box>
                    </Box>
                  )
                })}
              </Box>
            </Paper>
          )}

          {/* Chart View */}
          {viewMode === 'chart' && (
            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="subtitle1">Score Progression Over Time</Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={showTrendLine}
                      onChange={(e) => setShowTrendLine(e.target.checked)}
                    />
                  }
                  label="Show Trend"
                />
              </Box>
              
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis domain={[0, 100]} />
                    <RechartsTooltip
                      formatter={(value: any, name: string) => [
                        name === 'score' ? `${value.toFixed(1)}/100` : value,
                        name === 'score' ? 'Anomaly Score' : name
                      ]}
                    />
                    <Area
                      type="monotone"
                      dataKey="score"
                      stroke="#1976d2"
                      fill="#1976d2"
                      fillOpacity={0.1}
                    />
                    {showTrendLine && (
                      <Line
                        type="linear"
                        dataKey="score"
                        stroke="#f44336"
                        strokeWidth={2}
                        dot={false}
                      />
                    )}
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            </Paper>
          )}

          {/* Developmental Milestones View */}
          {viewMode === 'milestones' && (
            <Box>
              {developmentalMilestones.map((milestone, index) => (
                <Paper key={index} sx={{ p: 2, mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <SchoolIcon sx={{ mr: 1, color: 'primary.main' }} />
                    Age {milestone.age_range} Years
                    <Chip
                      label={`${(milestone.alignment_score * 100).toFixed(0)}% Alignment`}
                      color={milestone.alignment_score > 0.7 ? 'success' : milestone.alignment_score > 0.4 ? 'warning' : 'error'}
                      size="small"
                      sx={{ ml: 2 }}
                    />
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle2" gutterBottom>
                        Expected Patterns
                      </Typography>
                      <List dense>
                        {milestone.expected_patterns.map((pattern, i) => (
                          <ListItem key={i}>
                            <ListItemIcon>
                              <CheckCircleIcon color="success" fontSize="small" />
                            </ListItemIcon>
                            <ListItemText primary={pattern} />
                          </ListItem>
                        ))}
                      </List>
                    </Grid>
                    
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle2" gutterBottom>
                        Observed Patterns
                      </Typography>
                      <List dense>
                        {milestone.observed_patterns.map((pattern, i) => (
                          <ListItem key={i}>
                            <ListItemIcon>
                              <InfoIcon color="primary" fontSize="small" />
                            </ListItemIcon>
                            <ListItemText primary={pattern} />
                          </ListItem>
                        ))}
                      </List>
                    </Grid>
                  </Grid>
                  
                  {milestone.concerns.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom color="error">
                        Areas of Concern
                      </Typography>
                      <List dense>
                        {milestone.concerns.map((concern, i) => (
                          <ListItem key={i}>
                            <ListItemIcon>
                              <WarningIcon color="error" fontSize="small" />
                            </ListItemIcon>
                            <ListItemText primary={concern} />
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}
                </Paper>
              ))}
            </Box>
          )}
        </Box>
      )}

      {/* Analysis Detail Dialog */}
      <Dialog
        open={detailDialogOpen}
        onClose={() => setDetailDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Analysis Details: {selectedAnalysis?.analysis_timestamp && formatTimestamp(selectedAnalysis.analysis_timestamp)}
        </DialogTitle>
        <DialogContent>
          {selectedAnalysis && (
            <Box>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Analysis Information
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Analysis ID: {selectedAnalysis.id}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Child Age: {selectedAnalysis.child_age_at_time} years
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Age Group Model: {selectedAnalysis.age_group}
                  </Typography>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Results
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Anomaly Score: {selectedAnalysis.anomaly_score.toFixed(3)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Normalized Score: {selectedAnalysis.normalized_score.toFixed(1)}/100
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Status: {selectedAnalysis.is_anomaly ? 'Anomaly Detected' : 'Normal'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Confidence: {(selectedAnalysis.confidence * 100).toFixed(0)}%
                  </Typography>
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailDialogOpen(false)}>
            Close
          </Button>
          {selectedAnalysis && onAnalysisSelect && (
            <Button
              variant="contained"
              startIcon={<ViewIcon />}
              onClick={() => {
                onAnalysisSelect(selectedAnalysis.id)
                setDetailDialogOpen(false)
              }}
            >
              View Full Analysis
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  )
}