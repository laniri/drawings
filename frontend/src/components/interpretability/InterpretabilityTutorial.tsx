import React, { useState, useEffect } from 'react'
import {
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Paper,
  Card,
  CardContent,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
  IconButton,
  Fade,
  Zoom,
} from '@mui/material'
import {
  Close,

  Psychology,
  School,
  Lightbulb,
  TouchApp,
  ZoomIn,

  CheckCircle,
  ArrowForward,
  ArrowBack,
} from '@mui/icons-material'

interface TutorialStep {
  title: string
  content: string
  visual?: React.ReactNode
  tips: string[]
  userRole?: string[]
}

interface InterpretabilityTutorialProps {
  open: boolean
  onClose: () => void
  userRole?: 'researcher' | 'educator' | 'parent' | 'clinician'
  onComplete?: () => void
}

const InterpretabilityTutorial: React.FC<InterpretabilityTutorialProps> = ({
  open,
  onClose,
  userRole = 'educator',
  onComplete,
}) => {
  const [activeStep, setActiveStep] = useState(0)
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set())

  const getTutorialSteps = (): TutorialStep[] => {
    const baseSteps: TutorialStep[] = [
      {
        title: 'Welcome to Interpretability Analysis',
        content: 'This tutorial will guide you through understanding how our AI analyzes children\'s drawings and explains its decisions.',
        visual: (
          <Box display="flex" justifyContent="center" p={2}>
            <Psychology sx={{ fontSize: 60, color: 'primary.main' }} />
          </Box>
        ),
        tips: [
          'Take your time to understand each concept',
          'You can revisit this tutorial anytime',
          'Ask questions if something is unclear',
        ],
      },
      {
        title: 'Understanding Saliency Maps',
        content: 'Saliency maps show which parts of a drawing the AI focused on when making its decision. Brighter areas indicate higher importance.',
        visual: (
          <Card sx={{ maxWidth: 300, mx: 'auto' }}>
            <CardContent>
              <Box
                sx={{
                  width: '100%',
                  height: 150,
                  background: 'linear-gradient(45deg, rgba(255,0,0,0.3) 0%, rgba(255,255,0,0.5) 50%, rgba(0,255,0,0.3) 100%)',
                  border: '2px solid',
                  borderColor: 'primary.main',
                  borderRadius: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  Example Saliency Map
                </Typography>
              </Box>
              <Box mt={1} display="flex" justifyContent="space-between">
                <Chip label="Low Importance" size="small" sx={{ backgroundColor: 'rgba(0,255,0,0.3)' }} />
                <Chip label="High Importance" size="small" sx={{ backgroundColor: 'rgba(255,0,0,0.3)' }} />
              </Box>
            </CardContent>
          </Card>
        ),
        tips: [
          'Red/warm colors = high importance',
          'Green/cool colors = low importance',
          'The AI "looks" at highlighted areas most',
        ],
      },
      {
        title: 'Interactive Regions',
        content: 'You can hover over and click on different regions to get detailed explanations about what the AI detected.',
        visual: (
          <Box display="flex" justifyContent="center" gap={2} p={2}>
            <Box display="flex" flexDirection="column" alignItems="center">
              <TouchApp sx={{ fontSize: 40, color: 'primary.main' }} />
              <Typography variant="caption">Hover</Typography>
            </Box>
            <Box display="flex" flexDirection="column" alignItems="center">
              <ZoomIn sx={{ fontSize: 40, color: 'secondary.main' }} />
              <Typography variant="caption">Click</Typography>
            </Box>
          </Box>
        ),
        tips: [
          'Hover for quick explanations',
          'Click for detailed analysis',
          'Use zoom controls for better visibility',
        ],
      },
      {
        title: 'Confidence Levels',
        content: 'The system provides confidence scores to help you understand how reliable the analysis is.',
        visual: (
          <Box display="flex" flexDirection="column" gap={1} p={2}>
            <Box display="flex" alignItems="center" gap={1}>
              <CheckCircle sx={{ color: 'success.main' }} />
              <Typography variant="body2">High Confidence (80%+)</Typography>
            </Box>
            <Box display="flex" alignItems="center" gap={1}>
              <CheckCircle sx={{ color: 'warning.main' }} />
              <Typography variant="body2">Medium Confidence (60-79%)</Typography>
            </Box>
            <Box display="flex" alignItems="center" gap={1}>
              <CheckCircle sx={{ color: 'error.main' }} />
              <Typography variant="body2">Low Confidence (&lt;60%)</Typography>
            </Box>
          </Box>
        ),
        tips: [
          'Higher confidence = more reliable results',
          'Low confidence suggests caution needed',
          'Consider additional assessment for low confidence',
        ],
      },
    ]

    // Add role-specific steps
    if (userRole === 'educator') {
      baseSteps.push({
        title: 'Educational Applications',
        content: 'As an educator, you can use these insights to understand developmental patterns and identify students who might benefit from additional support.',
        visual: (
          <Box display="flex" justifyContent="center" p={2}>
            <School sx={{ fontSize: 60, color: 'primary.main' }} />
          </Box>
        ),
        tips: [
          'Look for patterns across multiple drawings',
          'Use simplified explanations with parents',
          'Document observations for professional consultation',
        ],
        userRole: ['educator'],
      })
    } else if (userRole === 'researcher') {
      baseSteps.push({
        title: 'Research Applications',
        content: 'Use technical explanations and export features to document findings and share results with colleagues.',
        visual: (
          <Box display="flex" justifyContent="center" p={2}>
            <Psychology sx={{ fontSize: 60, color: 'primary.main' }} />
          </Box>
        ),
        tips: [
          'Enable technical explanations for detailed analysis',
          'Export results for publications',
          'Compare confidence levels across age groups',
        ],
        userRole: ['researcher'],
      })
    } else if (userRole === 'parent') {
      baseSteps.push({
        title: 'Understanding Your Child\'s Drawing',
        content: 'The system helps identify if your child\'s drawing shows typical developmental patterns for their age.',
        visual: (
          <Box display="flex" justifyContent="center" p={2}>
            <Lightbulb sx={{ fontSize: 60, color: 'primary.main' }} />
          </Box>
        ),
        tips: [
          'Focus on simplified explanations',
          'Discuss results with educators or professionals',
          'Remember this is just one assessment tool',
        ],
        userRole: ['parent'],
      })
    }

    baseSteps.push({
      title: 'Best Practices',
      content: 'Remember that AI analysis is a tool to support human judgment, not replace it. Always consider the full context.',
      visual: (
        <Alert severity="info" sx={{ maxWidth: 400, mx: 'auto' }}>
          <Typography variant="body2">
            Use AI insights as one piece of information alongside professional expertise and contextual knowledge.
          </Typography>
        </Alert>
      ),
      tips: [
        'Combine AI insights with professional judgment',
        'Consider the child\'s context and background',
        'Seek professional consultation when needed',
      ],
    })

    return baseSteps
  }

  const steps = getTutorialSteps()

  const handleNext = () => {
    setCompletedSteps(prev => new Set([...prev, activeStep]))
    if (activeStep < steps.length - 1) {
      setActiveStep(activeStep + 1)
    }
  }

  const handleBack = () => {
    if (activeStep > 0) {
      setActiveStep(activeStep - 1)
    }
  }

  const handleComplete = () => {
    setCompletedSteps(prev => new Set([...prev, activeStep]))
    onComplete?.()
    onClose()
  }

  const handleStepClick = (stepIndex: number) => {
    setActiveStep(stepIndex)
  }

  useEffect(() => {
    if (open) {
      setActiveStep(0)
      setCompletedSteps(new Set())
    }
  }, [open])

  const getRoleDescription = (role: string) => {
    switch (role) {
      case 'researcher':
        return 'Research-focused tutorial with technical details'
      case 'educator':
        return 'Educational tutorial for classroom applications'
      case 'parent':
        return 'Parent-friendly tutorial with practical guidance'
      case 'clinician':
        return 'Clinical tutorial for professional assessment'
      default:
        return 'General tutorial'
    }
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { minHeight: '70vh' }
      }}
    >
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h5" component="div">
              Interpretability Tutorial
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {getRoleDescription(userRole)}
            </Typography>
          </Box>
          <IconButton onClick={onClose}>
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent>
        <Box sx={{ width: '100%' }}>
          <Stepper activeStep={activeStep} orientation="vertical">
            {steps.map((step, index) => (
              <Step key={step.title} completed={completedSteps.has(index)}>
                <StepLabel
                  onClick={() => handleStepClick(index)}
                  sx={{ cursor: 'pointer' }}
                >
                  <Typography variant="h6">{step.title}</Typography>
                </StepLabel>
                <StepContent>
                  <Fade in={activeStep === index} timeout={300}>
                    <Box>
                      <Typography variant="body1" paragraph>
                        {step.content}
                      </Typography>

                      {step.visual && (
                        <Zoom in={activeStep === index} timeout={500}>
                          <Box mb={2}>
                            {step.visual}
                          </Box>
                        </Zoom>
                      )}

                      <Paper sx={{ p: 2, backgroundColor: 'action.hover', mb: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          💡 Key Tips:
                        </Typography>
                        <List dense>
                          {step.tips.map((tip, tipIndex) => (
                            <ListItem key={tipIndex} sx={{ py: 0 }}>
                              <ListItemIcon sx={{ minWidth: 32 }}>
                                <Lightbulb fontSize="small" color="primary" />
                              </ListItemIcon>
                              <ListItemText
                                primary={tip}
                                primaryTypographyProps={{ variant: 'body2' }}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Paper>

                      <Box sx={{ mb: 1 }}>
                        <Button
                          variant="contained"
                          onClick={index === steps.length - 1 ? handleComplete : handleNext}
                          sx={{ mt: 1, mr: 1 }}
                          endIcon={index === steps.length - 1 ? <CheckCircle /> : <ArrowForward />}
                        >
                          {index === steps.length - 1 ? 'Complete Tutorial' : 'Next'}
                        </Button>
                        <Button
                          disabled={index === 0}
                          onClick={handleBack}
                          sx={{ mt: 1, mr: 1 }}
                          startIcon={<ArrowBack />}
                        >
                          Back
                        </Button>
                      </Box>
                    </Box>
                  </Fade>
                </StepContent>
              </Step>
            ))}
          </Stepper>
        </Box>
      </DialogContent>

      <DialogActions>
        <Box display="flex" justifyContent="space-between" width="100%" alignItems="center">
          <Typography variant="body2" color="text.secondary">
            Step {activeStep + 1} of {steps.length}
          </Typography>
          <Box>
            <Button onClick={onClose} color="inherit">
              Skip Tutorial
            </Button>
          </Box>
        </Box>
      </DialogActions>
    </Dialog>
  )
}

export default InterpretabilityTutorial