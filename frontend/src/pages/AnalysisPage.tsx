import React, { useState } from 'react'
import {
  Typography,
  Paper,
  Box,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Chip,
  Button,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Slider,
  FormControlLabel,
  Switch,
  Divider,
} from '@mui/material'
import {
  Warning,
  CheckCircle,
  Error as ErrorIcon,
  Visibility,
  Compare,
  Analytics,
  TrendingUp,
} from '@mui/icons-material'
import { useParams } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import axios from 'axios'
import {
  InteractiveInterpretabilityViewer,
  ExplanationLevelToggle,
  ConfidenceIndicator,
  ComparativeAnalysisPanel,
  ExportToolbar,
  AnnotationTools,
  HistoricalInterpretationTracker,
} from '../components/interpretability'

interface Drawing {
  id: number
  filename: string
  age_years: number
  subject?: string
  expert_label?: string
  upload_timestamp: string
}

interface AnalysisResult {
  id: number
  anomaly_score: number
  normalized_score: number
  visual_anomaly_score?: number
  subject_anomaly_score?: number
  anomaly_attribution?: string
  analysis_type: string
  subject_category?: string
  is_anomaly: boolean
  confidence: number
  age_group: string
  method_used: string
  analysis_timestamp: string
}

interface InterpretabilityResult {
  saliency_map_url: string
  overlay_image_url: string
  explanation_text?: string
  importance_regions: Array<{
    x: number
    y: number
    width: number
    height: number
    importance: number
  }>
}

interface AnalysisData {
  drawing: Drawing
  analysis: AnalysisResult
  interpretability?: InterpretabilityResult
}

const AnalysisPage: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const [activeTab, setActiveTab] = useState(0)
  const [showSaliency, setShowSaliency] = useState(true)
  const [saliencyOpacity, setSaliencyOpacity] = useState(0.6)
  const [selectedRegionExplanation, setSelectedRegionExplanation] = useState<string>('')
  const [, setCurrentExplanationLevel] = useState<'technical' | 'simplified'>('simplified')

  // Fetch analysis results
  const { data: analysisData, isLoading, error } = useQuery<AnalysisData>({
    queryKey: ['analysis', id],
    queryFn: async () => {
      try {
        // First try to get analysis by analysis ID
        const response = await axios.get(`/api/analysis/${id}`)
        return response.data
      } catch (error: any) {
        if (error.response?.status === 404) {
          // If not found, try to get the latest analysis for this drawing ID
          try {
            const drawingAnalysesResponse = await axios.get(`/api/analysis/drawing/${id}`)
            const analyses = drawingAnalysesResponse.data.analyses
            if (analyses && analyses.length > 0) {
              // Get the latest analysis (first in the list since they're ordered by timestamp desc)
              const latestAnalysisId = analyses[0].id
              // Now fetch the complete analysis result
              const analysisResponse = await axios.get(`/api/analysis/${latestAnalysisId}`)
              return analysisResponse.data
            }
          } catch (drawingError) {
            // If both fail, throw the original error
            throw error
          }
        }
        throw error
      }
    },
    enabled: !!id,
  })

  // Trigger new analysis
  const analysisMutation = useMutation({
    mutationFn: async () => {
      if (!analysisData?.drawing?.id) {
        return Promise.reject(new Error('Drawing ID not available'))
      }
      const response = await axios.post(`/api/analysis/analyze/${analysisData.drawing.id}`)
      return response.data
    },
    onSuccess: () => {
      // Refetch analysis data
      window.location.reload()
    },
  })

  const getAnomalyStatusColor = (isAnomaly: boolean, confidence: number) => {
    if (!isAnomaly) return 'success'
    if (confidence > 0.8) return 'error'
    return 'warning'
  }

  const getAnomalyStatusIcon = (isAnomaly: boolean, confidence: number) => {
    if (!isAnomaly) return <CheckCircle />
    if (confidence > 0.8) return <ErrorIcon />
    return <Warning />
  }

  const getAnomalyStatusText = (isAnomaly: boolean, confidence: number) => {
    if (!isAnomaly) return 'Normal'
    if (confidence > 0.8) return 'High Anomaly'
    return 'Potential Anomaly'
  }

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    )
  }

  if (error || !analysisData) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Analysis Results
        </Typography>
        <Alert severity="error">
          Failed to load analysis results. Please try again.
        </Alert>
        <Button
          variant="contained"
          onClick={() => analysisMutation.mutate()}
          disabled={analysisMutation.isPending}
          sx={{ mt: 2 }}
        >
          {analysisMutation.isPending ? 'Analyzing...' : 'Run Analysis'}
        </Button>
      </Box>
    )
  }

  // Add null check before destructuring
  if (!analysisData) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Analysis Results
        </Typography>
        <Alert severity="error">
          No analysis data found. The analysis may not exist or failed to load.
        </Alert>
        <Button
          variant="contained"
          onClick={() => analysisMutation.mutate()}
          disabled={analysisMutation.isPending}
          sx={{ mt: 2 }}
        >
          {analysisMutation.isPending ? 'Analyzing...' : 'Run New Analysis'}
        </Button>
      </Box>
    )
  }

  // Safe to destructure after null check
  const { drawing, analysis, interpretability } = analysisData

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Analysis Results
      </Typography>

      <Grid container spacing={3}>
        {/* Drawing Information */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardMedia
              component="img"
              height="300"
              image={`/api/drawings/${drawing.id}/file`}
              alt={drawing.filename}
              sx={{ objectFit: 'contain' }}
            />
            <CardContent>
              <Typography variant="h6" gutterBottom>
                {drawing.filename}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Age: {drawing.age_years} years
              </Typography>
              {drawing.subject && (
                <Typography variant="body2" color="text.secondary">
                  Subject: {drawing.subject}
                </Typography>
              )}
              {analysis.subject_category && analysis.subject_category !== drawing.subject && (
                <Typography variant="body2" color="text.secondary">
                  Analysis Subject: {analysis.subject_category}
                </Typography>
              )}
              {drawing.expert_label && (
                <Chip
                  label={`Expert: ${drawing.expert_label}`}
                  size="small"
                  sx={{ mt: 1 }}
                />
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Analysis Results */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Box display="flex" alignItems="center" gap={2} mb={3}>
              <Typography variant="h5">Analysis Results</Typography>
              <Chip
                icon={getAnomalyStatusIcon(analysis.is_anomaly, analysis.confidence)}
                label={getAnomalyStatusText(analysis.is_anomaly, analysis.confidence)}
                color={getAnomalyStatusColor(analysis.is_anomaly, analysis.confidence)}
                variant="outlined"
              />
            </Box>

            <Grid container spacing={2} mb={3}>
              <Grid item xs={6} sm={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="primary">
                    {analysis.anomaly_score.toFixed(3)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Overall Score
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="secondary">
                    {(analysis.confidence * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Confidence
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Box textAlign="center">
                  <Typography variant="h4">
                    {analysis.normalized_score.toFixed(3)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Normalized Score
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Box textAlign="center">
                  <Typography variant="body1" fontWeight="bold">
                    {analysis.age_group}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Age Group
                  </Typography>
                </Box>
              </Grid>
            </Grid>

            {/* Subject-Aware Analysis Details */}
            {analysis.analysis_type === 'subject_aware' && (
              <Box mb={3}>
                <Typography variant="h6" gutterBottom>
                  Subject-Aware Analysis Details
                </Typography>
                <Grid container spacing={2}>
                  {analysis.visual_anomaly_score !== undefined && (
                    <Grid item xs={6} sm={3}>
                      <Box textAlign="center">
                        <Typography variant="h5" color="info.main">
                          {analysis.visual_anomaly_score.toFixed(3)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Visual Score
                        </Typography>
                      </Box>
                    </Grid>
                  )}
                  {analysis.subject_anomaly_score !== undefined && (
                    <Grid item xs={6} sm={3}>
                      <Box textAlign="center">
                        <Typography variant="h5" color="warning.main">
                          {analysis.subject_anomaly_score.toFixed(3)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Subject Score
                        </Typography>
                      </Box>
                    </Grid>
                  )}
                  {analysis.anomaly_attribution && (
                    <Grid item xs={6} sm={3}>
                      <Box textAlign="center">
                        <Chip
                          label={analysis.anomaly_attribution.charAt(0).toUpperCase() + analysis.anomaly_attribution.slice(1)}
                          color={
                            analysis.anomaly_attribution === 'visual' ? 'info' :
                            analysis.anomaly_attribution === 'subject' ? 'warning' :
                            analysis.anomaly_attribution === 'both' ? 'error' :
                            'default'
                          }
                          variant="filled"
                        />
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                          Primary Attribution
                        </Typography>
                      </Box>
                    </Grid>
                  )}
                  {analysis.subject_category && (
                    <Grid item xs={6} sm={3}>
                      <Box textAlign="center">
                        <Typography variant="body1" fontWeight="bold">
                          {analysis.subject_category}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Subject Category
                        </Typography>
                      </Box>
                    </Grid>
                  )}
                </Grid>
                
                {/* Attribution Explanation */}
                {analysis.anomaly_attribution && (
                  <Alert severity="info" sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      <strong>Attribution Explanation:</strong>{' '}
                      {analysis.anomaly_attribution === 'visual' && 
                        'The anomaly is primarily in the visual features of the drawing (shapes, lines, spatial relationships).'}
                      {analysis.anomaly_attribution === 'subject' && 
                        'The anomaly is primarily related to the subject category representation.'}
                      {analysis.anomaly_attribution === 'both' && 
                        'The anomaly involves both visual features and subject representation.'}
                      {analysis.anomaly_attribution === 'age' && 
                        'The drawing appears more typical for a different age group.'}
                    </Typography>
                  </Alert>
                )}
              </Box>
            )}

            <Divider sx={{ my: 2 }} />

            <Typography variant="body2" color="text.secondary" gutterBottom>
              Method: {analysis.method_used} | Type: {analysis.analysis_type} | Analyzed: {new Date(analysis.analysis_timestamp).toLocaleString()}
            </Typography>

            <Button
              variant="outlined"
              onClick={() => analysisMutation.mutate()}
              disabled={analysisMutation.isPending}
              startIcon={<Analytics />}
              sx={{ mt: 2 }}
            >
              {analysisMutation.isPending ? 'Re-analyzing...' : 'Re-analyze'}
            </Button>
          </Paper>
        </Grid>

        {/* Interpretability Results */}
        {interpretability && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>
                Interpretability Analysis
              </Typography>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
                  <Tab label="Interactive Analysis" icon={<Analytics />} />
                  <Tab label="Saliency Map" icon={<Visibility />} />
                  <Tab label="Comparison" icon={<Compare />} />
                  <Tab label="Confidence" icon={<TrendingUp />} />
                  <Tab label="History" icon={<TrendingUp />} />
                  <Tab label="Annotations" icon={<Analytics />} />
                </Tabs>
                
                <ExportToolbar
                  analysisId={analysis.id}
                  drawingFilename={drawing.filename}
                  onExportComplete={(result) => {
                    console.log('Export completed:', result)
                  }}
                />
              </Box>

              {activeTab === 0 && (
                <Box sx={{ mt: 3 }}>
                  <Grid container spacing={3}>
                    <Grid item xs={12} lg={8}>
                      <InteractiveInterpretabilityViewer
                        analysisId={analysis.id}
                        drawingImageUrl={`/api/drawings/${drawing.id}/file`}
                        saliencyMapUrl={interpretability.saliency_map_url}
                        onRegionClick={(_, explanation) => {
                          setSelectedRegionExplanation(explanation)
                        }}
                      />
                    </Grid>
                    <Grid item xs={12} lg={4}>
                      <Box mb={2}>
                        <ConfidenceIndicator
                          analysisId={analysis.id}
                          compact={true}
                        />
                      </Box>
                      <ExplanationLevelToggle
                        analysisId={analysis.id}
                        technicalExplanation={interpretability.explanation_text}
                        userRole="educator"
                        onExplanationChange={(level, explanation) => {
                          setCurrentExplanationLevel(level)
                          if (level === 'simplified') {
                            setSelectedRegionExplanation(explanation)
                          }
                        }}
                      />
                      {selectedRegionExplanation && (
                        <Paper sx={{ p: 2, mt: 2, backgroundColor: 'primary.light', color: 'primary.contrastText' }}>
                          <Typography variant="subtitle2" gutterBottom>
                            Selected Region Analysis:
                          </Typography>
                          <Typography variant="body2">
                            {selectedRegionExplanation}
                          </Typography>
                        </Paper>
                      )}
                    </Grid>
                  </Grid>
                </Box>
              )}

              {activeTab === 1 && (
                <Box sx={{ mt: 3 }}>
                  <Box display="flex" alignItems="center" gap={2} mb={2}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={showSaliency}
                          onChange={(e) => setShowSaliency(e.target.checked)}
                        />
                      }
                      label="Show Saliency Overlay"
                    />
                    {showSaliency && (
                      <Box sx={{ width: 200 }}>
                        <Typography variant="body2" gutterBottom>
                          Opacity: {Math.round(saliencyOpacity * 100)}%
                        </Typography>
                        <Slider
                          value={saliencyOpacity}
                          onChange={(_, value) => setSaliencyOpacity(value as number)}
                          min={0.1}
                          max={1}
                          step={0.1}
                          size="small"
                        />
                      </Box>
                    )}
                  </Box>

                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Original Drawing
                          </Typography>
                          <Box
                            component="img"
                            src={`/api/drawings/${drawing.id}/file`}
                            alt="Original"
                            sx={{
                              width: '100%',
                              height: 'auto',
                              maxHeight: 400,
                              objectFit: 'contain',
                            }}
                          />
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            {showSaliency ? 'Saliency Overlay' : 'Saliency Map'}
                          </Typography>
                          <Box sx={{ position: 'relative' }}>
                            <Box
                              component="img"
                              src={
                                showSaliency
                                  ? `/api/drawings/${drawing.id}/file`
                                  : interpretability.saliency_map_url
                              }
                              alt="Base"
                              sx={{
                                width: '100%',
                                height: 'auto',
                                maxHeight: 400,
                                objectFit: 'contain',
                              }}
                            />
                            {showSaliency && (
                              <Box
                                component="img"
                                src={interpretability.saliency_map_url}
                                alt="Saliency"
                                sx={{
                                  position: 'absolute',
                                  top: 0,
                                  left: 0,
                                  width: '100%',
                                  height: '100%',
                                  objectFit: 'contain',
                                  opacity: saliencyOpacity,
                                  mixBlendMode: 'multiply',
                                }}
                              />
                            )}
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>

                  {interpretability.explanation_text && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                      <Typography variant="body2">
                        {interpretability.explanation_text}
                      </Typography>
                    </Alert>
                  )}
                </Box>
              )}

              {activeTab === 2 && (
                <Box sx={{ mt: 3 }}>
                  <ComparativeAnalysisPanel
                    currentAnalysis={{
                      ...analysis,
                      drawing_id: drawing.id
                    }}
                    currentDrawing={drawing}
                    onExampleSelect={(exampleId) => {
                      // Navigate to the example analysis
                      window.open(`/analysis/${exampleId}`, '_blank')
                    }}
                  />
                </Box>
              )}

              {activeTab === 3 && (
                <Box sx={{ mt: 3 }}>
                  <ConfidenceIndicator
                    analysisId={analysis.id}
                    showTechnicalDetails={true}
                  />
                </Box>
              )}

              {activeTab === 4 && (
                <Box sx={{ mt: 3 }}>
                  <HistoricalInterpretationTracker
                    drawingId={drawing.id}
                    currentAnalysis={analysis}
                    onAnalysisSelect={(analysisId) => {
                      // Navigate to the historical analysis
                      window.location.href = `/analysis/${analysisId}`
                    }}
                  />
                </Box>
              )}

              {activeTab === 5 && (
                <Box sx={{ mt: 3 }}>
                  <AnnotationTools
                    analysisId={analysis.id}
                    regions={interpretability?.importance_regions?.map((region, index) => ({
                      region_id: `region_${index + 1}`,
                      bounding_box: [region.x, region.y, region.x + region.width, region.y + region.height],
                      spatial_location: `Region ${index + 1}`,
                      importance_score: region.importance
                    })) || []}
                    onAnnotationAdd={(annotation) => {
                      console.log('Annotation added:', annotation)
                    }}
                    onAnnotationUpdate={(annotation) => {
                      console.log('Annotation updated:', annotation)
                    }}
                    onAnnotationDelete={(annotationId) => {
                      console.log('Annotation deleted:', annotationId)
                    }}
                  />
                </Box>
              )}
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  )
}

export default AnalysisPage