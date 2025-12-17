import React, { useState, useRef } from 'react'
import {
  Box,
  Tooltip,
  IconButton,
  Popover,
  Paper,
  Typography,
  Button,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Alert,
  Fade,
} from '@mui/material'
import {
  Help,
  Info,
  Lightbulb,
  Psychology,
  School,
  Science,
  Close,
  Launch,
} from '@mui/icons-material'

interface HelpContent {
  title: string
  description: string
  examples?: string[]
  tips?: string[]
  technicalDetails?: string
  roleSpecific?: {
    researcher?: string
    educator?: string
    parent?: string
    clinician?: string
  }
}

interface ContextualHelpProps {
  topic: string
  userRole?: 'researcher' | 'educator' | 'parent' | 'clinician'
  placement?: 'top' | 'bottom' | 'left' | 'right'
  size?: 'small' | 'medium' | 'large'
  showTechnical?: boolean
  children?: React.ReactNode
}

const ContextualHelpSystem: React.FC<ContextualHelpProps> = ({
  topic,
  userRole = 'educator',
  placement = 'top',
  size = 'medium',
  showTechnical = false,
  children,
}) => {
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const helpRef = useRef<HTMLButtonElement>(null)

  const helpContent: Record<string, HelpContent> = {
    'saliency-maps': {
      title: 'Saliency Maps',
      description: 'Visual representations showing which parts of the drawing the AI model focused on when making its decision.',
      examples: [
        'Red/warm areas indicate high importance',
        'Blue/cool areas indicate low importance',
        'Brightness shows the strength of attention',
      ],
      tips: [
        'Look for patterns in highlighted regions',
        'Compare with your own visual assessment',
        'Consider the overall distribution of attention',
      ],
      technicalDetails: 'Generated using gradient-based attribution methods on Vision Transformer attention weights, normalized across spatial dimensions.',
      roleSpecific: {
        educator: 'Use saliency maps to understand what visual features might indicate developmental patterns in student drawings.',
        researcher: 'Saliency maps provide quantitative evidence for feature importance in developmental assessment models.',
        parent: 'These colorful overlays show what the computer "noticed" most in your child\'s drawing.',
        clinician: 'Saliency analysis supports clinical observation by highlighting potentially significant visual elements.',
      },
    },
    'confidence-scores': {
      title: 'Confidence Scores',
      description: 'Numerical indicators of how certain the AI model is about its analysis and predictions.',
      examples: [
        'High (80%+): Strong evidence supports the analysis',
        'Medium (60-79%): Moderate evidence, consider additional context',
        'Low (<60%): Limited evidence, use results cautiously',
      ],
      tips: [
        'Higher confidence suggests more reliable results',
        'Low confidence may indicate edge cases or insufficient training data',
        'Always consider confidence when interpreting results',
      ],
      technicalDetails: 'Calculated from model uncertainty, training data quality, and prediction entropy across ensemble models.',
      roleSpecific: {
        educator: 'Use confidence levels to decide when to seek additional professional input for student assessments.',
        researcher: 'Confidence metrics help evaluate model reliability and identify areas needing more training data.',
        parent: 'Confidence tells you how sure the computer is - like a teacher being more or less certain about an answer.',
        clinician: 'Confidence scores inform clinical decision-making and the need for additional assessment tools.',
      },
    },
    'anomaly-detection': {
      title: 'Anomaly Detection',
      description: 'The process of identifying drawings that deviate significantly from typical patterns for a child\'s age group.',
      examples: [
        'Unusual proportions or spatial relationships',
        'Unexpected complexity or simplicity for age',
        'Atypical use of space or detail',
      ],
      tips: [
        'Anomalies are not necessarily concerning - they indicate difference',
        'Consider cultural and individual factors',
        'Use as one piece of information, not a diagnosis',
      ],
      technicalDetails: 'Uses autoencoder reconstruction loss to measure deviation from learned age-appropriate patterns in embedding space.',
      roleSpecific: {
        educator: 'Anomaly detection helps identify students who might benefit from additional developmental support or enrichment.',
        researcher: 'Provides quantitative measures of developmental variation for research studies and population analysis.',
        parent: 'Shows if your child\'s drawing is typical for their age - differences aren\'t necessarily problems.',
        clinician: 'Supports screening for developmental concerns but should be combined with comprehensive clinical assessment.',
      },
    },
    'age-groups': {
      title: 'Age Group Models',
      description: 'Separate AI models trained on drawings from children in specific age ranges to account for developmental differences.',
      examples: [
        'Ages 2-3: Basic shapes and scribbles',
        'Ages 4-5: Recognizable objects emerge',
        'Ages 6-8: More detailed and proportional drawings',
      ],
      tips: [
        'Each age group has different expectations',
        'Models account for natural developmental progression',
        'Cross-age comparisons should be interpreted carefully',
      ],
      technicalDetails: 'Age-stratified autoencoders trained on developmental datasets with minimum sample thresholds for statistical validity.',
      roleSpecific: {
        educator: 'Age-appropriate models help set realistic expectations for student developmental milestones.',
        researcher: 'Enables developmental trajectory analysis and cross-sectional age comparisons in studies.',
        parent: 'The computer knows what\'s typical for your child\'s age and compares accordingly.',
        clinician: 'Age-stratified analysis supports developmental assessment and milestone tracking.',
      },
    },
    'interactive-regions': {
      title: 'Interactive Regions',
      description: 'Clickable areas on the drawing that provide detailed explanations about what the AI detected.',
      examples: [
        'Hover for quick explanations',
        'Click for detailed analysis',
        'Color intensity shows importance level',
      ],
      tips: [
        'Explore different regions to understand the full analysis',
        'Pay attention to both highlighted and non-highlighted areas',
        'Use zoom controls for better visibility of small regions',
      ],
      technicalDetails: 'Regions defined by clustering attention weights and gradient magnitudes, with explanations generated from feature attribution analysis.',
      roleSpecific: {
        educator: 'Interactive exploration helps you understand and explain AI findings to students and parents.',
        researcher: 'Provides detailed feature-level analysis for research documentation and publication.',
        parent: 'Click around the drawing to learn what the computer noticed in different parts.',
        clinician: 'Detailed region analysis supports clinical observation and documentation.',
      },
    },
    'vision-transformer': {
      title: 'Vision Transformer (ViT)',
      description: 'The AI model that analyzes drawings by breaking them into patches and understanding relationships between different parts.',
      examples: [
        'Processes images as sequences of patches',
        'Learns spatial relationships and patterns',
        'Provides attention-based explanations',
      ],
      tips: [
        'More sophisticated than traditional image analysis',
        'Can understand complex spatial relationships',
        'Attention patterns provide interpretable insights',
      ],
      technicalDetails: 'Transformer architecture adapted for computer vision, using patch embeddings and multi-head self-attention for spatial feature learning.',
      roleSpecific: {
        educator: 'A sophisticated AI that can understand complex patterns in drawings, similar to how humans analyze visual information.',
        researcher: 'State-of-the-art computer vision model providing both high accuracy and interpretable attention mechanisms.',
        parent: 'An advanced computer program that looks at drawings the way a trained expert might.',
        clinician: 'Advanced AI model that provides both quantitative analysis and qualitative insights for clinical assessment.',
      },
    },
  }

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget)
  }

  const handleClose = () => {
    setAnchorEl(null)
    setShowAdvanced(false)
  }

  const open = Boolean(anchorEl)
  const content = helpContent[topic]

  if (!content) {
    return (
      <Tooltip title={`Help not available for topic: ${topic}`}>
        <IconButton size={size} disabled>
          <Help />
        </IconButton>
      </Tooltip>
    )
  }

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'researcher':
        return <Science />
      case 'educator':
        return <School />
      case 'parent':
        return <Lightbulb />
      case 'clinician':
        return <Psychology />
      default:
        return <Info />
    }
  }

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'researcher':
        return 'primary'
      case 'educator':
        return 'secondary'
      case 'parent':
        return 'success'
      case 'clinician':
        return 'warning'
      default:
        return 'default'
    }
  }

  return (
    <>
      {children ? (
        <Box
          ref={helpRef}
          onClick={handleClick}
          sx={{ cursor: 'pointer', display: 'inline-flex', alignItems: 'center' }}
        >
          {children}
        </Box>
      ) : (
        <Tooltip title={`Get help about ${content.title}`}>
          <IconButton
            ref={helpRef}
            size={size}
            onClick={handleClick}
            sx={{ color: 'text.secondary' }}
          >
            <Help />
          </IconButton>
        </Tooltip>
      )}

      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: placement === 'top' ? 'top' : placement === 'bottom' ? 'bottom' : 'center',
          horizontal: placement === 'left' ? 'left' : placement === 'right' ? 'right' : 'center',
        }}
        transformOrigin={{
          vertical: placement === 'top' ? 'bottom' : placement === 'bottom' ? 'top' : 'center',
          horizontal: placement === 'left' ? 'right' : placement === 'right' ? 'left' : 'center',
        }}
        PaperProps={{
          sx: { maxWidth: 400, p: 0 }
        }}
      >
        <Paper sx={{ p: 2 }}>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
            <Box display="flex" alignItems="center" gap={1}>
              <Info color="primary" />
              <Typography variant="h6">{content.title}</Typography>
            </Box>
            <IconButton size="small" onClick={handleClose}>
              <Close />
            </IconButton>
          </Box>

          <Box mb={2}>
            <Chip
              icon={getRoleIcon(userRole)}
              label={`${userRole.charAt(0).toUpperCase() + userRole.slice(1)} View`}
              size="small"
              color={getRoleColor(userRole) as any}
              variant="outlined"
            />
          </Box>

          <Typography variant="body2" paragraph>
            {content.description}
          </Typography>

          {/* Role-specific explanation */}
          {content.roleSpecific?.[userRole] && (
            <Alert severity="info" sx={{ mb: 2 }}>
              <Typography variant="body2">
                <strong>For {userRole}s:</strong> {content.roleSpecific[userRole]}
              </Typography>
            </Alert>
          )}

          {/* Examples */}
          {content.examples && (
            <Box mb={2}>
              <Typography variant="subtitle2" gutterBottom>
                Examples:
              </Typography>
              <List dense>
                {content.examples.map((example, index) => (
                  <ListItem key={index} sx={{ py: 0 }}>
                    <ListItemIcon sx={{ minWidth: 24 }}>
                      <Lightbulb fontSize="small" color="primary" />
                    </ListItemIcon>
                    <ListItemText
                      primary={example}
                      primaryTypographyProps={{ variant: 'body2' }}
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}

          {/* Tips */}
          {content.tips && (
            <Box mb={2}>
              <Typography variant="subtitle2" gutterBottom>
                💡 Tips:
              </Typography>
              <List dense>
                {content.tips.map((tip, index) => (
                  <ListItem key={index} sx={{ py: 0 }}>
                    <ListItemIcon sx={{ minWidth: 24 }}>
                      <Lightbulb fontSize="small" color="secondary" />
                    </ListItemIcon>
                    <ListItemText
                      primary={tip}
                      primaryTypographyProps={{ variant: 'body2' }}
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}

          {/* Technical details (if enabled) */}
          {(showTechnical || showAdvanced) && content.technicalDetails && (
            <Fade in={showAdvanced} timeout={300}>
              <Box>
                <Divider sx={{ my: 1 }} />
                <Typography variant="subtitle2" gutterBottom>
                  Technical Details:
                </Typography>
                <Paper sx={{ p: 1, backgroundColor: 'grey.50' }}>
                  <Typography variant="body2" fontFamily="monospace">
                    {content.technicalDetails}
                  </Typography>
                </Paper>
              </Box>
            </Fade>
          )}

          {/* Actions */}
          <Box display="flex" justifyContent="space-between" alignItems="center" mt={2}>
            {content.technicalDetails && (
              <Button
                size="small"
                onClick={() => setShowAdvanced(!showAdvanced)}
                startIcon={<Science />}
              >
                {showAdvanced ? 'Hide' : 'Show'} Technical
              </Button>
            )}
            <Button
              size="small"
              onClick={handleClose}
              endIcon={<Launch />}
              sx={{ ml: 'auto' }}
            >
              Got it
            </Button>
          </Box>
        </Paper>
      </Popover>
    </>
  )
}

export default ContextualHelpSystem