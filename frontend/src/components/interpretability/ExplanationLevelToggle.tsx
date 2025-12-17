import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  Paper,
  Chip,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material'
import {
  Science,
  School,
  ExpandMore,
  Lightbulb,
  Warning,
  CheckCircle,
  Info,
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'

interface SimplifiedExplanation {
  summary: string
  key_findings: string[]
  visual_indicators: Array<{
    indicator: string
    meaning: string
  }>
  confidence_level: string
  age_appropriate_context: string
  recommendations: string[]
}

interface ExplanationLevelToggleProps {
  analysisId: number
  technicalExplanation?: string
  userRole?: 'researcher' | 'educator' | 'parent' | 'clinician'
  onExplanationChange?: (level: 'technical' | 'simplified', explanation: string) => void
}

const ExplanationLevelToggle: React.FC<ExplanationLevelToggleProps> = ({
  analysisId,
  technicalExplanation,
  userRole = 'educator',
  onExplanationChange,
}) => {
  const [explanationLevel, setExplanationLevel] = useState<'technical' | 'simplified'>('simplified')

  // Fetch simplified explanation
  const { data: simplifiedData, isLoading, error } = useQuery<SimplifiedExplanation>({
    queryKey: ['simplified-explanation', analysisId, userRole],
    queryFn: async () => {
      const response = await axios.get(`/api/interpretability/${analysisId}/simplified`, {
        params: { user_role: userRole }
      })
      return response.data
    },
    enabled: !!analysisId,
  })

  const handleLevelChange = (
    _: React.MouseEvent<HTMLElement>,
    newLevel: 'technical' | 'simplified' | null,
  ) => {
    if (newLevel !== null) {
      setExplanationLevel(newLevel)
      const explanation = newLevel === 'technical' 
        ? technicalExplanation || 'Technical explanation not available'
        : simplifiedData?.summary || 'Simplified explanation loading...'
      onExplanationChange?.(newLevel, explanation)
    }
  }

  const getConfidenceColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'high':
        return 'success'
      case 'medium':
        return 'warning'
      case 'low':
        return 'error'
      default:
        return 'default'
    }
  }

  const getConfidenceIcon = (level: string) => {
    switch (level.toLowerCase()) {
      case 'high':
        return <CheckCircle />
      case 'medium':
        return <Warning />
      case 'low':
        return <Warning />
      default:
        return <Info />
    }
  }

  const getRoleDescription = (role: string) => {
    switch (role) {
      case 'researcher':
        return 'Research-focused explanations with statistical context'
      case 'educator':
        return 'Educational explanations for classroom use'
      case 'parent':
        return 'Parent-friendly explanations with practical guidance'
      case 'clinician':
        return 'Clinical explanations for professional assessment'
      default:
        return 'General explanations'
    }
  }

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">Explanation Level</Typography>
          <Chip 
            label={getRoleDescription(userRole)} 
            size="small" 
            variant="outlined"
            color="primary"
          />
        </Box>

        <ToggleButtonGroup
          value={explanationLevel}
          exclusive
          onChange={handleLevelChange}
          aria-label="explanation level"
          fullWidth
          sx={{ mb: 3 }}
        >
          <ToggleButton value="simplified" aria-label="simplified explanation">
            <School sx={{ mr: 1 }} />
            Simplified
          </ToggleButton>
          <ToggleButton value="technical" aria-label="technical explanation">
            <Science sx={{ mr: 1 }} />
            Technical
          </ToggleButton>
        </ToggleButtonGroup>

        {explanationLevel === 'simplified' && (
          <>
            {isLoading && (
              <Box display="flex" justifyContent="center" p={2}>
                <CircularProgress />
              </Box>
            )}

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                Failed to load simplified explanation. Please try again.
              </Alert>
            )}

            {simplifiedData && (
              <Box>
                {/* Confidence indicator */}
                <Box display="flex" alignItems="center" gap={1} mb={2}>
                  <Chip
                    icon={getConfidenceIcon(simplifiedData.confidence_level)}
                    label={`${simplifiedData.confidence_level} Confidence`}
                    color={getConfidenceColor(simplifiedData.confidence_level) as any}
                    variant="outlined"
                  />
                  <Typography variant="body2" color="text.secondary">
                    {simplifiedData.age_appropriate_context}
                  </Typography>
                </Box>

                {/* Main summary */}
                <Paper sx={{ p: 2, mb: 2, backgroundColor: 'action.hover' }}>
                  <Typography variant="body1" paragraph>
                    {simplifiedData.summary}
                  </Typography>
                </Paper>

                {/* Key findings */}
                <Accordion defaultExpanded>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="subtitle1" fontWeight="bold">
                      Key Findings
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <List dense>
                      {simplifiedData.key_findings.map((finding, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            <Lightbulb color="primary" />
                          </ListItemIcon>
                          <ListItemText primary={finding} />
                        </ListItem>
                      ))}
                    </List>
                  </AccordionDetails>
                </Accordion>

                {/* Visual indicators guide */}
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="subtitle1" fontWeight="bold">
                      How to Read the Visual Analysis
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <List dense>
                      {simplifiedData.visual_indicators.map((indicator, index) => (
                        <ListItem key={index}>
                          <ListItemText
                            primary={indicator.indicator}
                            secondary={indicator.meaning}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </AccordionDetails>
                </Accordion>

                {/* Recommendations */}
                {simplifiedData.recommendations.length > 0 && (
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Typography variant="subtitle1" fontWeight="bold">
                        Recommendations
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List dense>
                        {simplifiedData.recommendations.map((recommendation, index) => (
                          <ListItem key={index}>
                            <ListItemIcon>
                              <CheckCircle color="success" />
                            </ListItemIcon>
                            <ListItemText primary={recommendation} />
                          </ListItem>
                        ))}
                      </List>
                    </AccordionDetails>
                  </Accordion>
                )}

                {/* Confidence level explanation */}
                <Alert 
                  severity={getConfidenceColor(simplifiedData.confidence_level) as any} 
                  sx={{ mt: 2 }}
                >
                  <Typography variant="body2">
                    <strong>{simplifiedData.confidence_level} Confidence:</strong>{' '}
                    {simplifiedData.confidence_level === 'High' && 
                      'The analysis is based on strong patterns and reliable data. The findings are likely accurate.'}
                    {simplifiedData.confidence_level === 'Medium' && 
                      'The analysis shows clear patterns but may benefit from additional context or data.'}
                    {simplifiedData.confidence_level === 'Low' && 
                      'The analysis has limited confidence. Consider additional assessment or consultation.'}
                  </Typography>
                </Alert>
              </Box>
            )}
          </>
        )}

        {explanationLevel === 'technical' && (
          <Box>
            <Alert severity="info" sx={{ mb: 2 }}>
              <Typography variant="body2">
                Technical explanation includes detailed model outputs, statistical measures, and algorithmic details.
              </Typography>
            </Alert>

            <Paper sx={{ p: 2, backgroundColor: 'grey.50' }}>
              <Typography variant="body2" fontFamily="monospace" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                {technicalExplanation || 'Technical explanation not available for this analysis.'}
              </Typography>
            </Paper>

            <Divider sx={{ my: 2 }} />

            <Typography variant="subtitle2" gutterBottom>
              Technical Details:
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText
                  primary="Vision Model"
                  secondary="Vision Transformer (ViT) - Feature extraction from drawing images"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Anomaly Detection"
                  secondary="Autoencoder reconstruction loss - Measures deviation from learned patterns"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Interpretability Method"
                  secondary="Attention visualization + Gradient-based saliency mapping"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Age Group Modeling"
                  secondary="Age-stratified autoencoders trained on developmental patterns"
                />
              </ListItem>
            </List>
          </Box>
        )}
      </CardContent>
    </Card>
  )
}

export default ExplanationLevelToggle