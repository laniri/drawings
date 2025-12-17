import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Tabs,
  Tab,
  Grid,
  Chip,
  Alert,
  Paper,
  IconButton,
  Tooltip,
  Fab,
  Badge,
  Collapse,
} from '@mui/material'
import {
  School,
  Help,
  Lightbulb,
  Psychology,
  Science,
  ExpandMore,
  ExpandLess,
  QuestionMark,
  AutoStories,
  Visibility,
  Settings,
} from '@mui/icons-material'

import InterpretabilityTutorial from './InterpretabilityTutorial'
import ContextualHelpSystem from './ContextualHelpSystem'
import ExampleGallery from './ExampleGallery'
import AdaptiveExplanationSystem from './AdaptiveExplanationSystem'
import ExplanationLevelToggle from './ExplanationLevelToggle'
import ConfidenceIndicator from './ConfidenceIndicator'

interface InterpretabilityEducationHubProps {
  analysisId?: number
  analysisData?: any
  userRole?: 'researcher' | 'educator' | 'parent' | 'clinician'
  ageGroup?: string
  showTutorialOnMount?: boolean
  onUserRoleChange?: (role: string) => void
}

const InterpretabilityEducationHub: React.FC<InterpretabilityEducationHubProps> = ({
  analysisId,
  analysisData,
  userRole = 'educator',
  ageGroup,
  showTutorialOnMount = false,
  onUserRoleChange,
}) => {
  const [activeTab, setActiveTab] = useState(0)
  const [showTutorial, setShowTutorial] = useState(showTutorialOnMount)
  const [currentUserRole, setCurrentUserRole] = useState(userRole)
  const [isFirstVisit, setIsFirstVisit] = useState(false)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['overview']))

  // Check if this is the user's first visit
  useEffect(() => {
    const hasVisited = localStorage.getItem('interpretability_tutorial_completed')
    if (!hasVisited) {
      setIsFirstVisit(true)
      setShowTutorial(true)
    }
  }, [])

  // Handle tutorial completion
  const handleTutorialComplete = () => {
    localStorage.setItem('interpretability_tutorial_completed', 'true')
    setIsFirstVisit(false)
    setShowTutorial(false)
  }

  // Handle role change
  const handleRoleChange = (newRole: string) => {
    setCurrentUserRole(newRole as any)
    onUserRoleChange?.(newRole)
  }



  // Toggle section expansion
  const toggleSection = (section: string) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev)
      if (newSet.has(section)) {
        newSet.delete(section)
      } else {
        newSet.add(section)
      }
      return newSet
    })
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

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'researcher': return 'primary'
      case 'educator': return 'secondary'
      case 'parent': return 'success'
      case 'clinician': return 'warning'
      default: return 'default'
    }
  }

  const tabLabels = [
    { label: 'Overview', icon: <Visibility /> },
    { label: 'Examples', icon: <AutoStories /> },
    { label: 'Tutorial', icon: <School /> },
    { label: 'Settings', icon: <Settings /> },
  ]

  return (
    <Box>
      {/* Header with role indicator and help */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box display="flex" alignItems="center" gap={2}>
              <Typography variant="h5">
                Interpretability Guide
              </Typography>
              <Chip
                icon={getRoleIcon(currentUserRole)}
                label={`${currentUserRole.charAt(0).toUpperCase() + currentUserRole.slice(1)} View`}
                color={getRoleColor(currentUserRole) as any}
                variant="outlined"
              />
              {isFirstVisit && (
                <Badge badgeContent="New" color="error">
                  <Chip label="First Visit" size="small" />
                </Badge>
              )}
            </Box>
            
            <Box display="flex" gap={1}>
              <ContextualHelpSystem
                topic="interpretability-overview"
                userRole={currentUserRole}
                placement="bottom"
              >
                <Tooltip title="Get help with interpretability">
                  <IconButton>
                    <Help />
                  </IconButton>
                </Tooltip>
              </ContextualHelpSystem>
              
              <Button
                variant="outlined"
                startIcon={<School />}
                onClick={() => setShowTutorial(true)}
                size="small"
              >
                Tutorial
              </Button>
            </Box>
          </Box>

          {/* Quick tips for first-time users */}
          {isFirstVisit && (
            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="body2">
                Welcome! This guide will help you understand AI interpretability results. 
                We recommend starting with the tutorial to learn the basics.
              </Typography>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Main content tabs */}
      <Card>
        <CardContent>
          <Tabs
            value={activeTab}
            onChange={(_, newValue) => setActiveTab(newValue)}
            variant="fullWidth"
            sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
          >
            {tabLabels.map((tab, index) => (
              <Tab
                key={index}
                icon={tab.icon}
                label={tab.label}
                iconPosition="start"
              />
            ))}
          </Tabs>

          {/* Overview Tab */}
          {activeTab === 0 && (
            <Box>
              <Grid container spacing={3}>
                {/* Quick Start Section */}
                <Grid item xs={12}>
                  <Paper sx={{ p: 2, mb: 2 }}>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="h6">
                        Quick Start Guide
                      </Typography>
                      <IconButton onClick={() => toggleSection('quickstart')}>
                        {expandedSections.has('quickstart') ? <ExpandLess /> : <ExpandMore />}
                      </IconButton>
                    </Box>
                    
                    <Collapse in={expandedSections.has('quickstart')}>
                      <Box mt={2}>
                        <Grid container spacing={2}>
                          <Grid item xs={12} md={4}>
                            <Card variant="outlined">
                              <CardContent>
                                <Box display="flex" alignItems="center" gap={1} mb={1}>
                                  <Visibility color="primary" />
                                  <Typography variant="subtitle2">
                                    1. View Saliency Maps
                                  </Typography>
                                </Box>
                                <Typography variant="body2" color="text.secondary">
                                  Colored overlays show which parts of the drawing the AI focused on.
                                </Typography>
                                <ContextualHelpSystem
                                  topic="saliency-maps"
                                  userRole={currentUserRole}
                                  size="small"
                                />
                              </CardContent>
                            </Card>
                          </Grid>
                          
                          <Grid item xs={12} md={4}>
                            <Card variant="outlined">
                              <CardContent>
                                <Box display="flex" alignItems="center" gap={1} mb={1}>
                                  <Psychology color="primary" />
                                  <Typography variant="subtitle2">
                                    2. Check Confidence
                                  </Typography>
                                </Box>
                                <Typography variant="body2" color="text.secondary">
                                  Confidence scores tell you how reliable the analysis is.
                                </Typography>
                                <ContextualHelpSystem
                                  topic="confidence-scores"
                                  userRole={currentUserRole}
                                  size="small"
                                />
                              </CardContent>
                            </Card>
                          </Grid>
                          
                          <Grid item xs={12} md={4}>
                            <Card variant="outlined">
                              <CardContent>
                                <Box display="flex" alignItems="center" gap={1} mb={1}>
                                  <AutoStories color="primary" />
                                  <Typography variant="subtitle2">
                                    3. Explore Examples
                                  </Typography>
                                </Box>
                                <Typography variant="body2" color="text.secondary">
                                  See examples of typical and unusual patterns for comparison.
                                </Typography>
                                <Button
                                  size="small"
                                  onClick={() => setActiveTab(1)}
                                  sx={{ mt: 1 }}
                                >
                                  View Examples
                                </Button>
                              </CardContent>
                            </Card>
                          </Grid>
                        </Grid>
                      </Box>
                    </Collapse>
                  </Paper>
                </Grid>

                {/* Current Analysis Section */}
                {analysisId && analysisData && (
                  <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                        <Typography variant="h6">
                          Current Analysis
                        </Typography>
                        <IconButton onClick={() => toggleSection('analysis')}>
                          {expandedSections.has('analysis') ? <ExpandLess /> : <ExpandMore />}
                        </IconButton>
                      </Box>
                      
                      <Collapse in={expandedSections.has('analysis')}>
                        <Grid container spacing={2}>
                          <Grid item xs={12} md={6}>
                            <ExplanationLevelToggle
                              analysisId={analysisId}
                              userRole={currentUserRole}
                            />
                          </Grid>
                          <Grid item xs={12} md={6}>
                            <ConfidenceIndicator
                              analysisId={analysisId}
                              compact={false}
                            />
                          </Grid>
                          <Grid item xs={12}>
                            <AdaptiveExplanationSystem
                              analysisData={analysisData}
                              initialConfig={{ userRole: currentUserRole }}
                              onConfigChange={(config) => {
                                if (config.userRole !== currentUserRole) {
                                  handleRoleChange(config.userRole)
                                }
                              }}
                            />
                          </Grid>
                        </Grid>
                      </Collapse>
                    </Paper>
                  </Grid>
                )}

                {/* Key Concepts Section */}
                <Grid item xs={12}>
                  <Paper sx={{ p: 2 }}>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                      <Typography variant="h6">
                        Key Concepts
                      </Typography>
                      <IconButton onClick={() => toggleSection('concepts')}>
                        {expandedSections.has('concepts') ? <ExpandLess /> : <ExpandMore />}
                      </IconButton>
                    </Box>
                    
                    <Collapse in={expandedSections.has('concepts')}>
                      <Grid container spacing={2}>
                        {[
                          { topic: 'saliency-maps', title: 'Saliency Maps', description: 'Visual importance indicators' },
                          { topic: 'confidence-scores', title: 'Confidence Scores', description: 'Reliability measures' },
                          { topic: 'anomaly-detection', title: 'Anomaly Detection', description: 'Pattern deviation analysis' },
                          { topic: 'age-groups', title: 'Age Group Models', description: 'Developmental expectations' },
                        ].map((concept) => (
                          <Grid item xs={12} sm={6} md={3} key={concept.topic}>
                            <Card variant="outlined" sx={{ height: '100%' }}>
                              <CardContent>
                                <Typography variant="subtitle2" gutterBottom>
                                  {concept.title}
                                </Typography>
                                <Typography variant="body2" color="text.secondary" paragraph>
                                  {concept.description}
                                </Typography>
                                <ContextualHelpSystem
                                  topic={concept.topic}
                                  userRole={currentUserRole}
                                  size="small"
                                >
                                  <Button size="small" startIcon={<Help />}>
                                    Learn More
                                  </Button>
                                </ContextualHelpSystem>
                              </CardContent>
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    </Collapse>
                  </Paper>
                </Grid>
              </Grid>
            </Box>
          )}

          {/* Examples Tab */}
          {activeTab === 1 && (
            <Box>
              <ExampleGallery
                ageGroup={ageGroup}
                userRole={currentUserRole}
                filterByType="all"
              />
            </Box>
          )}

          {/* Tutorial Tab */}
          {activeTab === 2 && (
            <Box>
              <Paper sx={{ p: 3, textAlign: 'center' }}>
                <School sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
                <Typography variant="h5" gutterBottom>
                  Interactive Tutorial
                </Typography>
                <Typography variant="body1" color="text.secondary" paragraph>
                  Learn how to interpret AI analysis results with our step-by-step guide.
                </Typography>
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<School />}
                  onClick={() => setShowTutorial(true)}
                >
                  Start Tutorial
                </Button>
              </Paper>
            </Box>
          )}

          {/* Settings Tab */}
          {activeTab === 3 && (
            <Box>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      User Role Settings
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Choose your role to get explanations tailored to your needs.
                    </Typography>
                    
                    <Grid container spacing={1}>
                      {[
                        { role: 'educator', label: 'Educator', description: 'For teachers and educational professionals' },
                        { role: 'researcher', label: 'Researcher', description: 'For academic and research purposes' },
                        { role: 'parent', label: 'Parent', description: 'For parents and caregivers' },
                        { role: 'clinician', label: 'Clinician', description: 'For healthcare professionals' },
                      ].map((option) => (
                        <Grid item xs={12} key={option.role}>
                          <Card
                            variant={currentUserRole === option.role ? 'elevation' : 'outlined'}
                            sx={{
                              cursor: 'pointer',
                              border: currentUserRole === option.role ? 2 : 1,
                              borderColor: currentUserRole === option.role ? 'primary.main' : 'divider',
                            }}
                            onClick={() => handleRoleChange(option.role)}
                          >
                            <CardContent sx={{ py: 1 }}>
                              <Box display="flex" alignItems="center" gap={2}>
                                {getRoleIcon(option.role)}
                                <Box>
                                  <Typography variant="subtitle2">
                                    {option.label}
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary">
                                    {option.description}
                                  </Typography>
                                </Box>
                              </Box>
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Tutorial & Help
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Access learning resources and reset tutorial progress.
                    </Typography>
                    
                    <Box display="flex" flexDirection="column" gap={2}>
                      <Button
                        variant="outlined"
                        startIcon={<School />}
                        onClick={() => setShowTutorial(true)}
                        fullWidth
                      >
                        Restart Tutorial
                      </Button>
                      
                      <Button
                        variant="outlined"
                        startIcon={<QuestionMark />}
                        onClick={() => {
                          localStorage.removeItem('interpretability_tutorial_completed')
                          setIsFirstVisit(true)
                        }}
                        fullWidth
                      >
                        Reset Tutorial Progress
                      </Button>
                      
                      <Button
                        variant="outlined"
                        startIcon={<AutoStories />}
                        onClick={() => setActiveTab(1)}
                        fullWidth
                      >
                        Browse Examples
                      </Button>
                    </Box>
                  </Paper>
                </Grid>
              </Grid>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Tutorial Dialog */}
      <InterpretabilityTutorial
        open={showTutorial}
        onClose={() => setShowTutorial(false)}
        userRole={currentUserRole}
        onComplete={handleTutorialComplete}
      />

      {/* Floating Help Button */}
      <Fab
        color="primary"
        sx={{
          position: 'fixed',
          bottom: 16,
          right: 16,
          zIndex: 1000,
        }}
        onClick={() => setShowTutorial(true)}
      >
        <Help />
      </Fab>
    </Box>
  )
}

export default InterpretabilityEducationHub