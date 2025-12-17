import React from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Alert,
  Grid,
  Tooltip,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  CircularProgress,
  useTheme,
} from '@mui/material'
import {
  CheckCircle,
  Warning,
  Error,
  Info,
  DataUsage,
  Psychology,
  Security,
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'

interface ConfidenceMetrics {
  overall_confidence: number
  explanation_reliability: number
  model_certainty: number
  data_sufficiency: string
  warnings: string[]
  technical_details: {
    base_model_confidence: number
    training_data_quality: number
    score_extremity: number
    age_group_sample_count: number
    analysis_method: string
    vision_model: string
  }
}

interface ConfidenceIndicatorProps {
  analysisId: number
  showTechnicalDetails?: boolean
  compact?: boolean
}

const ConfidenceIndicator: React.FC<ConfidenceIndicatorProps> = ({
  analysisId,
  showTechnicalDetails = false,
  compact = false,
}) => {
  const theme = useTheme()

  // Fetch confidence metrics
  const { data: confidenceData, isLoading, error } = useQuery<ConfidenceMetrics>({
    queryKey: ['confidence-metrics', analysisId],
    queryFn: async () => {
      const response = await axios.get(`/api/interpretability/${analysisId}/confidence`)
      return response.data
    },
    enabled: !!analysisId,
  })

  const getConfidenceColor = (value: number) => {
    if (value >= 0.8) return theme.palette.success.main
    if (value >= 0.6) return theme.palette.warning.main
    return theme.palette.error.main
  }

  const getConfidenceLabel = (value: number) => {
    if (value >= 0.8) return 'High'
    if (value >= 0.6) return 'Medium'
    return 'Low'
  }

  const getConfidenceIcon = (value: number) => {
    if (value >= 0.8) return <CheckCircle />
    if (value >= 0.6) return <Warning />
    return <Error />
  }

  const getDataSufficiencyColor = (sufficiency: string) => {
    switch (sufficiency.toLowerCase()) {
      case 'sufficient':
        return 'success'
      case 'limited':
        return 'warning'
      case 'insufficient':
        return 'error'
      default:
        return 'default'
    }
  }

  const getDataSufficiencyIcon = (sufficiency: string) => {
    switch (sufficiency.toLowerCase()) {
      case 'sufficient':
        return <CheckCircle />
      case 'limited':
        return <Warning />
      case 'insufficient':
        return <Error />
      default:
        return <Info />
    }
  }

  const ConfidenceMeter: React.FC<{ 
    label: string
    value: number
    icon: React.ReactNode
    description: string
  }> = ({ label, value, icon, description }) => (
    <Tooltip title={description} placement="top">
      <Box>
        <Box display="flex" alignItems="center" gap={1} mb={1}>
          {icon}
          <Typography variant="body2" fontWeight="medium">
            {label}
          </Typography>
          <Chip
            label={getConfidenceLabel(value)}
            size="small"
            sx={{
              backgroundColor: getConfidenceColor(value),
              color: 'white',
              fontWeight: 'bold',
            }}
          />
        </Box>
        <LinearProgress
          variant="determinate"
          value={value * 100}
          sx={{
            height: 8,
            borderRadius: 4,
            backgroundColor: 'grey.200',
            '& .MuiLinearProgress-bar': {
              backgroundColor: getConfidenceColor(value),
              borderRadius: 4,
            },
          }}
        />
        <Typography variant="caption" color="text.secondary">
          {(value * 100).toFixed(1)}%
        </Typography>
      </Box>
    </Tooltip>
  )

  if (isLoading) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" gap={2}>
            <CircularProgress size={20} />
            <Typography>Loading confidence metrics...</Typography>
          </Box>
        </CardContent>
      </Card>
    )
  }

  if (error || !confidenceData) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">
            Failed to load confidence metrics
          </Alert>
        </CardContent>
      </Card>
    )
  }

  if (compact) {
    return (
      <Box display="flex" alignItems="center" gap={2}>
        <Tooltip title={`Overall confidence: ${(confidenceData.overall_confidence * 100).toFixed(1)}%`}>
          <Chip
            icon={getConfidenceIcon(confidenceData.overall_confidence)}
            label={`${getConfidenceLabel(confidenceData.overall_confidence)} Confidence`}
            sx={{
              backgroundColor: getConfidenceColor(confidenceData.overall_confidence),
              color: 'white',
            }}
          />
        </Tooltip>
        {confidenceData.warnings.length > 0 && (
          <Tooltip title={confidenceData.warnings.join('; ')}>
            <Warning color="warning" />
          </Tooltip>
        )}
      </Box>
    )
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Confidence & Reliability Assessment
        </Typography>

        {/* Overall confidence summary */}
        <Paper sx={{ p: 2, mb: 3, backgroundColor: 'action.hover' }}>
          <Box display="flex" alignItems="center" gap={2} mb={1}>
            {getConfidenceIcon(confidenceData.overall_confidence)}
            <Typography variant="h6">
              {getConfidenceLabel(confidenceData.overall_confidence)} Confidence
            </Typography>
            <Typography variant="h6" color="text.secondary">
              ({(confidenceData.overall_confidence * 100).toFixed(1)}%)
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary">
            Overall assessment of analysis reliability and trustworthiness
          </Typography>
        </Paper>

        {/* Detailed confidence metrics */}
        <Grid container spacing={3} mb={3}>
          <Grid item xs={12} md={4}>
            <ConfidenceMeter
              label="Model Certainty"
              value={confidenceData.model_certainty}
              icon={<Psychology color="primary" />}
              description="How certain the AI model is about its prediction based on training data quality and age group representation"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <ConfidenceMeter
              label="Explanation Reliability"
              value={confidenceData.explanation_reliability}
              icon={<Security color="primary" />}
              description="How reliable the visual explanations and saliency maps are for this specific analysis"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <Box>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <DataUsage color="primary" />
                <Typography variant="body2" fontWeight="medium">
                  Data Sufficiency
                </Typography>
                <Chip
                  icon={getDataSufficiencyIcon(confidenceData.data_sufficiency)}
                  label={confidenceData.data_sufficiency}
                  size="small"
                  color={getDataSufficiencyColor(confidenceData.data_sufficiency) as any}
                  variant="outlined"
                />
              </Box>
              <Typography variant="caption" color="text.secondary">
                Quality and quantity of training data for this age group
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Warnings */}
        {confidenceData.warnings.length > 0 && (
          <Alert severity="warning" sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Important Considerations:
            </Typography>
            <List dense>
              {confidenceData.warnings.map((warning, index) => (
                <ListItem key={index} sx={{ py: 0 }}>
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    <Warning fontSize="small" />
                  </ListItemIcon>
                  <ListItemText
                    primary={warning}
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                </ListItem>
              ))}
            </List>
          </Alert>
        )}

        {/* Technical details */}
        {showTechnicalDetails && (
          <Paper sx={{ p: 2, backgroundColor: 'grey.50' }}>
            <Typography variant="subtitle2" gutterBottom>
              Technical Confidence Breakdown
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2">
                  <strong>Base Model Confidence:</strong> {(confidenceData.technical_details.base_model_confidence * 100).toFixed(1)}%
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  <strong>Training Data Quality:</strong> {(confidenceData.technical_details.training_data_quality * 100).toFixed(1)}%
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  <strong>Score Extremity:</strong> {(confidenceData.technical_details.score_extremity * 100).toFixed(1)}%
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  <strong>Age Group Samples:</strong> {confidenceData.technical_details.age_group_sample_count}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  <strong>Analysis Method:</strong> {confidenceData.technical_details.analysis_method}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  <strong>Vision Model:</strong> {confidenceData.technical_details.vision_model}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        )}

        {/* Interpretation guidance */}
        <Box mt={3}>
          <Typography variant="subtitle2" gutterBottom>
            How to Interpret Confidence Levels:
          </Typography>
          <List dense>
            <ListItem>
              <ListItemIcon>
                <CheckCircle color="success" />
              </ListItemIcon>
              <ListItemText
                primary="High Confidence (80%+)"
                secondary="Strong evidence supports the analysis. Results are likely reliable for decision-making."
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <Warning color="warning" />
              </ListItemIcon>
              <ListItemText
                primary="Medium Confidence (60-79%)"
                secondary="Moderate evidence supports the analysis. Consider additional context or assessment."
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <Error color="error" />
              </ListItemIcon>
              <ListItemText
                primary="Low Confidence (<60%)"
                secondary="Limited evidence. Use results cautiously and seek additional professional input."
              />
            </ListItem>
          </List>
        </Box>
      </CardContent>
    </Card>
  )
}

export default ConfidenceIndicator