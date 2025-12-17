import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Paper,
  Chip,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Button,

} from '@mui/material'
import {
  ExpandMore,
  Science,
  School,
  Lightbulb,
  Psychology,
  TuneRounded,
  AutoFixHigh,
  Visibility,

} from '@mui/icons-material'

interface ExplanationConfig {
  complexity: number // 1-5 scale
  userRole: 'researcher' | 'educator' | 'parent' | 'clinician'
  showTechnicalDetails: boolean
  showStatistics: boolean
  showComparisons: boolean
  explanationStyle: 'detailed' | 'concise' | 'visual'
  vocabularyLevel: 'basic' | 'intermediate' | 'advanced'
}

interface AdaptiveContent {
  title: string
  summary: string
  keyPoints: string[]
  technicalDetails?: string
  statistics?: string[]
  comparisons?: string[]
  visualCues?: string[]
  recommendations?: string[]
}

interface AdaptiveExplanationSystemProps {
  analysisData: any
  onConfigChange?: (config: ExplanationConfig) => void
  initialConfig?: Partial<ExplanationConfig>
}

const AdaptiveExplanationSystem: React.FC<AdaptiveExplanationSystemProps> = ({
  analysisData,
  onConfigChange,
  initialConfig = {},
}) => {
  const [config, setConfig] = useState<ExplanationConfig>({
    complexity: 3,
    userRole: 'educator',
    showTechnicalDetails: false,
    showStatistics: true,
    showComparisons: true,
    explanationStyle: 'detailed',
    vocabularyLevel: 'intermediate',
    ...initialConfig,
  })

  const [adaptiveContent, setAdaptiveContent] = useState<AdaptiveContent | null>(null)
  const [autoAdapt, setAutoAdapt] = useState(true)

  // Auto-adapt based on user role
  useEffect(() => {
    if (autoAdapt) {
      const roleDefaults = getRoleDefaults(config.userRole)
      setConfig(prev => ({ ...prev, ...roleDefaults }))
    }
  }, [config.userRole, autoAdapt])

  // Generate adaptive content based on configuration
  useEffect(() => {
    if (analysisData) {
      const content = generateAdaptiveContent(analysisData, config)
      setAdaptiveContent(content)
      onConfigChange?.(config)
    }
  }, [analysisData, config, onConfigChange])

  const getRoleDefaults = (role: string): Partial<ExplanationConfig> => {
    switch (role) {
      case 'researcher':
        return {
          complexity: 5,
          showTechnicalDetails: true,
          showStatistics: true,
          explanationStyle: 'detailed',
          vocabularyLevel: 'advanced',
        }
      case 'educator':
        return {
          complexity: 3,
          showTechnicalDetails: false,
          showStatistics: true,
          explanationStyle: 'detailed',
          vocabularyLevel: 'intermediate',
        }
      case 'parent':
        return {
          complexity: 2,
          showTechnicalDetails: false,
          showStatistics: false,
          explanationStyle: 'visual',
          vocabularyLevel: 'basic',
        }
      case 'clinician':
        return {
          complexity: 4,
          showTechnicalDetails: true,
          showStatistics: true,
          explanationStyle: 'concise',
          vocabularyLevel: 'advanced',
        }
      default:
        return {}
    }
  }

  const generateAdaptiveContent = (data: any, cfg: ExplanationConfig): AdaptiveContent => {
    const vocabularyMap = {
      basic: {
        'anomaly': 'unusual pattern',
        'saliency': 'important areas',
        'confidence': 'certainty',
        'reconstruction': 'comparison',
        'embedding': 'features',
      },
      intermediate: {
        'anomaly': 'anomaly',
        'saliency': 'saliency',
        'confidence': 'confidence',
        'reconstruction': 'reconstruction',
        'embedding': 'embedding',
      },
      advanced: {
        'anomaly': 'statistical anomaly',
        'saliency': 'saliency attribution',
        'confidence': 'confidence interval',
        'reconstruction': 'autoencoder reconstruction',
        'embedding': 'feature embedding',
      },
    }

    const vocab = vocabularyMap[cfg.vocabularyLevel]

    // Generate title based on complexity and role
    let title = 'Analysis Results'
    if (cfg.complexity >= 4) {
      title = `Comprehensive ${vocab.anomaly} Analysis`
    } else if (cfg.complexity >= 3) {
      title = `Drawing Analysis: ${data.is_anomaly ? 'Unusual Pattern Detected' : 'Typical Pattern'}`
    } else {
      title = data.is_anomaly ? 'Something Interesting Found' : 'Drawing Looks Normal'
    }

    // Generate summary based on role and complexity
    let summary = ''
    if (cfg.userRole === 'parent') {
      summary = data.is_anomaly 
        ? `Your child's drawing shows some ${vocab.anomaly} patterns that are different from what we typically see in children this age. This doesn't necessarily mean there's a problem - it just means the drawing is unique in some way.`
        : `Your child's drawing shows typical patterns for their age group. The computer analysis suggests this is within the normal range of development.`
    } else if (cfg.userRole === 'educator') {
      summary = data.is_anomaly
        ? `This drawing exhibits ${vocab.anomaly} characteristics with a ${vocab.confidence} level of ${(data.confidence * 100).toFixed(0)}%. The analysis highlights specific visual features that deviate from age-expected patterns.`
        : `This drawing demonstrates age-appropriate developmental characteristics. The analysis shows typical patterns consistent with the child's age group.`
    } else if (cfg.userRole === 'researcher') {
      summary = `${vocab.reconstruction} analysis yielded an ${vocab.anomaly} score of ${data.anomaly_score.toFixed(3)} (threshold: ${data.threshold.toFixed(3)}). ${vocab.confidence} metrics indicate ${data.confidence > 0.8 ? 'high' : data.confidence > 0.6 ? 'moderate' : 'low'} reliability.`
    } else { // clinician
      summary = data.is_anomaly
        ? `Clinical assessment indicates ${vocab.anomaly} patterns requiring further evaluation. ${vocab.confidence} level: ${(data.confidence * 100).toFixed(0)}%. Consider comprehensive developmental screening.`
        : `Assessment indicates typical developmental patterns. No immediate concerns identified through automated analysis.`
    }

    // Generate key points based on complexity
    const keyPoints: string[] = []
    
    if (cfg.complexity >= 2) {
      keyPoints.push(
        data.is_anomaly 
          ? `${vocab.anomaly.charAt(0).toUpperCase() + vocab.anomaly.slice(1)} detected in drawing patterns`
          : 'Drawing patterns appear typical for age group'
      )
    }

    if (cfg.complexity >= 3) {
      keyPoints.push(`Analysis ${vocab.confidence}: ${data.confidence > 0.8 ? 'High' : data.confidence > 0.6 ? 'Medium' : 'Low'}`)
      keyPoints.push(`${vocab.saliency.charAt(0).toUpperCase() + vocab.saliency.slice(1)} map highlights key regions`)
    }

    if (cfg.complexity >= 4) {
      keyPoints.push(`Age group model: ${data.age_group} years`)
      keyPoints.push(`${vocab.reconstruction.charAt(0).toUpperCase() + vocab.reconstruction.slice(1)} method used for analysis`)
    }

    if (cfg.complexity >= 5) {
      keyPoints.push(`Feature ${vocab.embedding} dimensionality: ${data.embedding_dimension || 'N/A'}`)
      keyPoints.push(`Model architecture: Vision Transformer (ViT)`)
    }

    // Generate technical details
    const technicalDetails = cfg.showTechnicalDetails ? 
      `Vision Transformer ${vocab.embedding} processed through age-stratified autoencoder. ${vocab.reconstruction} loss: ${data.anomaly_score?.toFixed(6)}. Threshold (95th percentile): ${data.threshold?.toFixed(6)}. Statistical significance: p < 0.05.` 
      : undefined

    // Generate statistics
    const statistics = cfg.showStatistics ? [
      `Score: ${data.anomaly_score?.toFixed(3)} (threshold: ${data.threshold?.toFixed(3)})`,
      `${vocab.confidence.charAt(0).toUpperCase() + vocab.confidence.slice(1)}: ${(data.confidence * 100).toFixed(1)}%`,
      `Age group: ${data.age_group} years`,
      `Sample size: ${data.sample_count || 'N/A'} drawings`,
    ] : undefined

    // Generate comparisons
    const comparisons = cfg.showComparisons ? [
      data.is_anomaly 
        ? `This drawing differs from ${(100 - data.percentile * 100).toFixed(0)}% of drawings in this age group`
        : `This drawing is similar to ${(data.percentile * 100).toFixed(0)}% of drawings in this age group`,
      `Compared to age-matched peers: ${data.is_anomaly ? 'Atypical' : 'Typical'}`,
    ] : undefined

    // Generate visual cues
    const visualCues = cfg.explanationStyle === 'visual' ? [
      'Red/warm areas show high importance',
      'Blue/cool areas show low importance',
      'Brighter colors indicate stronger attention',
      'Click regions for detailed explanations',
    ] : undefined

    // Generate recommendations
    const recommendations: string[] = []
    if (cfg.userRole === 'educator' && data.is_anomaly) {
      recommendations.push('Consider additional developmental assessment')
      recommendations.push('Document observations for professional consultation')
      recommendations.push('Monitor progress over time')
    } else if (cfg.userRole === 'parent') {
      recommendations.push(data.is_anomaly ? 'Discuss with teacher or pediatrician' : 'Continue encouraging creative expression')
      recommendations.push('Remember this is just one assessment tool')
    } else if (cfg.userRole === 'clinician' && data.is_anomaly) {
      recommendations.push('Conduct comprehensive developmental screening')
      recommendations.push('Consider multidisciplinary assessment')
      recommendations.push('Review developmental history')
    }

    return {
      title,
      summary,
      keyPoints,
      technicalDetails,
      statistics,
      comparisons,
      visualCues,
      recommendations: recommendations.length > 0 ? recommendations : undefined,
    }
  }

  const handleConfigChange = (key: keyof ExplanationConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }

  const getComplexityLabel = (value: number) => {
    const labels = ['Very Simple', 'Simple', 'Moderate', 'Detailed', 'Expert']
    return labels[value - 1] || 'Moderate'
  }

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'researcher': return <Science />
      case 'educator': return <School />
      case 'parent': return <Lightbulb />
      case 'clinician': return <Psychology />
      default: return <School />
    }
  }

  return (
    <Box>
      {/* Configuration Panel */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">
              Explanation Settings
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={autoAdapt}
                  onChange={(e) => setAutoAdapt(e.target.checked)}
                />
              }
              label="Auto-adapt to role"
            />
          </Box>

          <Box display="flex" gap={2} flexWrap="wrap" alignItems="center">
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>User Role</InputLabel>
              <Select
                value={config.userRole}
                label="User Role"
                onChange={(e) => handleConfigChange('userRole', e.target.value)}
              >
                <MenuItem value="researcher">
                  <Box display="flex" alignItems="center" gap={1}>
                    <Science fontSize="small" />
                    Researcher
                  </Box>
                </MenuItem>
                <MenuItem value="educator">
                  <Box display="flex" alignItems="center" gap={1}>
                    <School fontSize="small" />
                    Educator
                  </Box>
                </MenuItem>
                <MenuItem value="parent">
                  <Box display="flex" alignItems="center" gap={1}>
                    <Lightbulb fontSize="small" />
                    Parent
                  </Box>
                </MenuItem>
                <MenuItem value="clinician">
                  <Box display="flex" alignItems="center" gap={1}>
                    <Psychology fontSize="small" />
                    Clinician
                  </Box>
                </MenuItem>
              </Select>
            </FormControl>

            <Box sx={{ minWidth: 200 }}>
              <Typography variant="body2" gutterBottom>
                Complexity: {getComplexityLabel(config.complexity)}
              </Typography>
              <Slider
                value={config.complexity}
                onChange={(_, value) => handleConfigChange('complexity', value)}
                min={1}
                max={5}
                step={1}
                marks
                disabled={autoAdapt}
              />
            </Box>

            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Style</InputLabel>
              <Select
                value={config.explanationStyle}
                label="Style"
                onChange={(e) => handleConfigChange('explanationStyle', e.target.value)}
                disabled={autoAdapt}
              >
                <MenuItem value="detailed">Detailed</MenuItem>
                <MenuItem value="concise">Concise</MenuItem>
                <MenuItem value="visual">Visual</MenuItem>
              </Select>
            </FormControl>

            <Button
              size="small"
              startIcon={<TuneRounded />}
              onClick={() => {
                // Could open advanced settings dialog
              }}
            >
              Advanced
            </Button>
          </Box>

          <Box display="flex" gap={1} mt={2} flexWrap="wrap">
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={config.showTechnicalDetails}
                  onChange={(e) => handleConfigChange('showTechnicalDetails', e.target.checked)}
                  disabled={autoAdapt}
                />
              }
              label="Technical Details"
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={config.showStatistics}
                  onChange={(e) => handleConfigChange('showStatistics', e.target.checked)}
                  disabled={autoAdapt}
                />
              }
              label="Statistics"
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={config.showComparisons}
                  onChange={(e) => handleConfigChange('showComparisons', e.target.checked)}
                  disabled={autoAdapt}
                />
              }
              label="Comparisons"
            />
          </Box>
        </CardContent>
      </Card>

      {/* Adaptive Content Display */}
      {adaptiveContent && (
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" gap={1} mb={2}>
              {getRoleIcon(config.userRole)}
              <Typography variant="h5">
                {adaptiveContent.title}
              </Typography>
              <Chip
                label={getComplexityLabel(config.complexity)}
                size="small"
                color="primary"
                variant="outlined"
              />
            </Box>

            <Typography variant="body1" paragraph>
              {adaptiveContent.summary}
            </Typography>

            {/* Key Points */}
            <Paper sx={{ p: 2, mb: 2, backgroundColor: 'action.hover' }}>
              <Typography variant="subtitle1" gutterBottom>
                Key Points:
              </Typography>
              <List dense>
                {adaptiveContent.keyPoints.map((point, index) => (
                  <ListItem key={index} sx={{ py: 0 }}>
                    <ListItemIcon sx={{ minWidth: 32 }}>
                      <AutoFixHigh fontSize="small" color="primary" />
                    </ListItemIcon>
                    <ListItemText primary={point} />
                  </ListItem>
                ))}
              </List>
            </Paper>

            {/* Visual Cues */}
            {adaptiveContent.visualCues && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <Visibility />
                    <Typography variant="subtitle1">
                      How to Read the Visual Analysis
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <List dense>
                    {adaptiveContent.visualCues.map((cue, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <Lightbulb color="secondary" />
                        </ListItemIcon>
                        <ListItemText primary={cue} />
                      </ListItem>
                    ))}
                  </List>
                </AccordionDetails>
              </Accordion>
            )}

            {/* Statistics */}
            {adaptiveContent.statistics && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="subtitle1">
                    Statistical Information
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <List dense>
                    {adaptiveContent.statistics.map((stat, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={stat} />
                      </ListItem>
                    ))}
                  </List>
                </AccordionDetails>
              </Accordion>
            )}

            {/* Comparisons */}
            {adaptiveContent.comparisons && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="subtitle1">
                    Comparative Analysis
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <List dense>
                    {adaptiveContent.comparisons.map((comparison, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={comparison} />
                      </ListItem>
                    ))}
                  </List>
                </AccordionDetails>
              </Accordion>
            )}

            {/* Technical Details */}
            {adaptiveContent.technicalDetails && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <Science />
                    <Typography variant="subtitle1">
                      Technical Details
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Paper sx={{ p: 2, backgroundColor: 'grey.50' }}>
                    <Typography variant="body2" fontFamily="monospace">
                      {adaptiveContent.technicalDetails}
                    </Typography>
                  </Paper>
                </AccordionDetails>
              </Accordion>
            )}

            {/* Recommendations */}
            {adaptiveContent.recommendations && (
              <Alert severity="info" sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Recommendations:
                </Typography>
                <List dense>
                  {adaptiveContent.recommendations.map((rec, index) => (
                    <ListItem key={index} sx={{ py: 0 }}>
                      <ListItemIcon sx={{ minWidth: 32 }}>
                        <Lightbulb fontSize="small" />
                      </ListItemIcon>
                      <ListItemText
                        primary={rec}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  )
}

export default AdaptiveExplanationSystem