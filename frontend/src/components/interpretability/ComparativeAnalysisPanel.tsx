import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Button,
  Divider,
  IconButton,
  Tooltip,
  Alert,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material'
import {
  Compare as CompareIcon,
  ZoomIn as ZoomInIcon,
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Timeline as TimelineIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material'

interface ComparisonExample {
  drawing_id: number
  filename: string
  age_years: number
  subject?: string
  anomaly_score: number
  normalized_score: number
  confidence: number
  analysis_timestamp: string
}

interface ComparisonData {
  normal_examples: ComparisonExample[]
  anomalous_examples: ComparisonExample[]
  explanation_context: string
  age_group: string
  total_available: number
}

interface AnalysisHistory {
  id: number
  anomaly_score: number
  normalized_score: number
  is_anomaly: boolean
  confidence: number
  analysis_timestamp: string
  age_group: string
}

interface ComparativeAnalysisPanelProps {
  currentAnalysis: {
    id: number
    drawing_id: number
    anomaly_score: number
    normalized_score: number
    is_anomaly: boolean
    confidence: number
    age_group: string
    analysis_timestamp: string
  }
  currentDrawing: {
    id: number
    filename: string
    age_years: number
    subject?: string
    file_path?: string
  }
  onExampleSelect?: (exampleId: number) => void
}

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`comparison-tabpanel-${index}`}
      aria-labelledby={`comparison-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  )
}

export default function ComparativeAnalysisPanel({
  currentAnalysis,
  currentDrawing,
  onExampleSelect
}: ComparativeAnalysisPanelProps) {
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null)
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistory[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedExample, setSelectedExample] = useState<ComparisonExample | null>(null)
  const [detailDialogOpen, setDetailDialogOpen] = useState(false)
  const [tabValue, setTabValue] = useState(0)

  // Load comparison examples
  useEffect(() => {
    loadComparisonExamples()
    loadAnalysisHistory()
  }, [currentAnalysis.age_group, currentDrawing.id])

  const loadComparisonExamples = async () => {
    try {
      setLoading(true)
      setError(null)

      // Try to get subject-specific examples first
      let url = `/api/interpretability/examples/${currentAnalysis.age_group}?example_type=both&limit=5`
      if (currentDrawing.subject) {
        url += `&subject=${encodeURIComponent(currentDrawing.subject)}`
      }

      const response = await fetch(url)

      if (!response.ok) {
        throw new Error('Failed to load comparison examples')
      }

      const data = await response.json()
      setComparisonData(data)
    } catch (err) {
      console.error('Error loading comparison examples:', err)
      setError(err instanceof Error ? err.message : 'Failed to load comparison examples')
    } finally {
      setLoading(false)
    }
  }

  const loadAnalysisHistory = async () => {
    try {
      const response = await fetch(`/api/analysis/drawing/${currentDrawing.id}`)
      
      if (!response.ok) {
        throw new Error('Failed to load analysis history')
      }

      const data = await response.json()
      setAnalysisHistory(data.analyses || [])
    } catch (err) {
      console.error('Error loading analysis history:', err)
      // Don't show error for history - it's supplementary data
    }
  }

  const handleExampleClick = (example: ComparisonExample) => {
    setSelectedExample(example)
    setDetailDialogOpen(true)
  }

  const handleViewExample = (exampleId: number) => {
    if (onExampleSelect) {
      onExampleSelect(exampleId)
    }
    setDetailDialogOpen(false)
  }

  const getScoreColor = (score: number, isAnomaly: boolean) => {
    if (isAnomaly) {
      return score >= 80 ? '#d32f2f' : score >= 60 ? '#f57c00' : '#ed6c02'
    }
    return '#2e7d32'
  }

  const getScoreLabel = (score: number, isAnomaly: boolean) => {
    if (isAnomaly) {
      if (score >= 80) return 'High Anomaly'
      if (score >= 60) return 'Moderate Anomaly'
      return 'Mild Anomaly'
    }
    return 'Normal'
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const renderExampleCard = (example: ComparisonExample, isNormal: boolean) => (
    <Card
      key={example.drawing_id}
      sx={{
        cursor: 'pointer',
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: 3
        },
        border: `2px solid ${getScoreColor(example.normalized_score, !isNormal)}`,
        borderRadius: 2
      }}
      onClick={() => handleExampleClick(example)}
    >
      <CardContent sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
          <Typography variant="subtitle2" noWrap sx={{ flex: 1, mr: 1 }}>
            {example.filename}
          </Typography>
          <Chip
            size="small"
            label={getScoreLabel(example.normalized_score, !isNormal)}
            sx={{
              backgroundColor: getScoreColor(example.normalized_score, !isNormal),
              color: 'white',
              fontSize: '0.7rem'
            }}
          />
        </Box>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Age: {example.age_years} years
          {example.subject && ` • Subject: ${example.subject}`}
        </Typography>
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            Score: {example.normalized_score.toFixed(1)}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {formatTimestamp(example.analysis_timestamp)}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  )

  const renderHistoryItem = (analysis: AnalysisHistory, index: number) => {
    const isLatest = index === 0
    const trend = index > 0 ? 
      (analysis.normalized_score > analysisHistory[index - 1].normalized_score ? 'up' : 
       analysis.normalized_score < analysisHistory[index - 1].normalized_score ? 'down' : 'stable') : 'none'

    return (
      <ListItem
        key={analysis.id}
        sx={{
          border: isLatest ? '2px solid #1976d2' : '1px solid #e0e0e0',
          borderRadius: 1,
          mb: 1,
          backgroundColor: isLatest ? '#f3f7ff' : 'white'
        }}
      >
        <ListItemIcon>
          {trend === 'up' && <TrendingUpIcon color="error" />}
          {trend === 'down' && <TrendingDownIcon color="success" />}
          {(trend === 'stable' || trend === 'none') && <TimelineIcon color="action" />}
        </ListItemIcon>
        <ListItemText
          primary={
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2">
                Score: {analysis.normalized_score.toFixed(1)}
                {isLatest && <Chip size="small" label="Current" sx={{ ml: 1 }} />}
              </Typography>
              <Chip
                size="small"
                label={analysis.is_anomaly ? 'Anomaly' : 'Normal'}
                color={analysis.is_anomaly ? 'error' : 'success'}
                variant="outlined"
              />
            </Box>
          }
          secondary={
            <Typography variant="caption" color="text.secondary">
              {formatTimestamp(analysis.analysis_timestamp)} • Confidence: {(analysis.confidence * 100).toFixed(0)}%
            </Typography>
          }
        />
      </ListItem>
    )
  }

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
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <CompareIcon sx={{ mr: 1, color: 'primary.main' }} />
        <Typography variant="h6">
          Comparative Analysis
        </Typography>
        <Tooltip title="Compare with similar drawings and view analysis history">
          <IconButton size="small" sx={{ ml: 1 }}>
            <InfoIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Current Analysis Summary */}
      <Paper sx={{ p: 2, mb: 3, backgroundColor: '#f8f9fa' }}>
        <Typography variant="subtitle1" gutterBottom>
          Current Analysis Summary
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              Drawing: {currentDrawing.filename}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Age: {currentDrawing.age_years} years
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              Score: {currentAnalysis.normalized_score.toFixed(1)}/100
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Status: {currentAnalysis.is_anomaly ? 'Anomaly Detected' : 'Normal'}
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Tabs for different comparison views */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
          <Tab label="Similar Examples" />
          <Tab label="Analysis History" />
          <Tab label="Age Group Context" />
        </Tabs>
      </Box>

      {/* Tab Panels */}
      <TabPanel value={tabValue} index={0}>
        {comparisonData && (
          <Box>
            {/* Normal Examples */}
            <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <TrendingDownIcon sx={{ mr: 1, color: 'success.main' }} />
              Normal Examples (Age {comparisonData.age_group}
              {currentDrawing.subject && `, Subject: ${currentDrawing.subject}`})
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              These drawings show typical patterns for this age group
              {currentDrawing.subject && ' and subject category'}
            </Typography>
            
            <Grid container spacing={2} sx={{ mb: 4 }}>
              {comparisonData.normal_examples.map((example) => (
                <Grid item xs={12} sm={6} md={4} key={example.drawing_id}>
                  {renderExampleCard(example, true)}
                </Grid>
              ))}
            </Grid>

            <Divider sx={{ my: 3 }} />

            {/* Anomalous Examples */}
            <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <TrendingUpIcon sx={{ mr: 1, color: 'error.main' }} />
              Anomalous Examples (Age {comparisonData.age_group}
              {currentDrawing.subject && `, Subject: ${currentDrawing.subject}`})
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              These drawings show patterns that deviate from typical development
              {currentDrawing.subject && ' for this subject category'}
            </Typography>
            
            <Grid container spacing={2}>
              {comparisonData.anomalous_examples.map((example) => (
                <Grid item xs={12} sm={6} md={4} key={example.drawing_id}>
                  {renderExampleCard(example, false)}
                </Grid>
              ))}
            </Grid>

            {comparisonData.normal_examples.length === 0 && comparisonData.anomalous_examples.length === 0 && (
              <Alert severity="info">
                No comparison examples available for age group {comparisonData.age_group}
              </Alert>
            )}
          </Box>
        )}
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <Box>
          <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            <TimelineIcon sx={{ mr: 1, color: 'primary.main' }} />
            Analysis History for {currentDrawing.filename}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Track changes in analysis results over time
          </Typography>

          {analysisHistory.length > 0 ? (
            <List sx={{ width: '100%' }}>
              {analysisHistory.map((analysis, index) => renderHistoryItem(analysis, index))}
            </List>
          ) : (
            <Alert severity="info">
              No previous analyses found for this drawing
            </Alert>
          )}
        </Box>
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        {comparisonData && (
          <Box>
            <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <AssessmentIcon sx={{ mr: 1, color: 'info.main' }} />
              Age Group Context
            </Typography>
            
            <Alert severity="info" sx={{ mb: 2 }}>
              {comparisonData.explanation_context}
            </Alert>

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Dataset Statistics
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total drawings in age group {comparisonData.age_group}: {comparisonData.total_available}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Normal examples shown: {comparisonData.normal_examples.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Anomalous examples shown: {comparisonData.anomalous_examples.length}
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Developmental Context
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Age {comparisonData.age_group} years represents a critical period for:
                  </Typography>
                  <Box component="ul" sx={{ mt: 1, pl: 2 }}>
                    <Typography component="li" variant="body2" color="text.secondary">
                      Spatial relationship development
                    </Typography>
                    <Typography component="li" variant="body2" color="text.secondary">
                      Fine motor skill refinement
                    </Typography>
                    <Typography component="li" variant="body2" color="text.secondary">
                      Symbolic representation growth
                    </Typography>
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          </Box>
        )}
      </TabPanel>

      {/* Example Detail Dialog */}
      <Dialog
        open={detailDialogOpen}
        onClose={() => setDetailDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Example Details: {selectedExample?.filename}
        </DialogTitle>
        <DialogContent>
          {selectedExample && (
            <Box>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Drawing Information
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Age: {selectedExample.age_years} years
                  </Typography>
                  {selectedExample.subject && (
                    <Typography variant="body2" color="text.secondary">
                      Subject: {selectedExample.subject}
                    </Typography>
                  )}
                  <Typography variant="body2" color="text.secondary">
                    Analysis Date: {formatTimestamp(selectedExample.analysis_timestamp)}
                  </Typography>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Analysis Results
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Anomaly Score: {selectedExample.anomaly_score.toFixed(3)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Normalized Score: {selectedExample.normalized_score.toFixed(1)}/100
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Confidence: {(selectedExample.confidence * 100).toFixed(0)}%
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
          {selectedExample && (
            <Button
              variant="contained"
              startIcon={<ZoomInIcon />}
              onClick={() => handleViewExample(selectedExample.drawing_id)}
            >
              View Full Analysis
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  )
}