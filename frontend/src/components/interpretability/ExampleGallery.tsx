import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  CardMedia,
  Typography,
  Grid,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Tabs,
  Tab,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
  Skeleton,
  Tooltip,
  Badge,
} from '@mui/material'
import {
  Close,
  Visibility,
  School,
  Psychology,
  Warning,
  CheckCircle,
  Lightbulb,
  Compare,
  FilterList,
  Info,
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'

interface ExamplePattern {
  pattern_id: string
  pattern_name: string
  description: string
  age_group: string
  example_type: 'normal' | 'anomalous' | 'borderline'
  confidence_level: 'high' | 'medium' | 'low'
  visual_features: string[]
  interpretation_notes: string[]
  educational_context: string
  image_url: string
  saliency_url: string
  metadata: {
    drawing_count: number
    prevalence: number
    developmental_significance: string
  }
}

interface ExampleGalleryProps {
  ageGroup?: string
  userRole?: 'researcher' | 'educator' | 'parent' | 'clinician'
  filterByType?: 'normal' | 'anomalous' | 'borderline' | 'all'
  onExampleSelect?: (example: ExamplePattern) => void
}

const ExampleGallery: React.FC<ExampleGalleryProps> = ({
  ageGroup,
  userRole = 'educator',
  filterByType = 'all',
  onExampleSelect,
}) => {
  const [selectedExample, setSelectedExample] = useState<ExamplePattern | null>(null)
  const [activeTab, setActiveTab] = useState(0)
  const [typeFilter, setTypeFilter] = useState<'all' | 'normal' | 'anomalous' | 'borderline'>(filterByType)

  // Fetch example patterns
  const { data: examples, isLoading, error } = useQuery<ExamplePattern[]>({
    queryKey: ['example-patterns', ageGroup, userRole],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (ageGroup) params.append('age_group', ageGroup)
      params.append('user_role', userRole)
      
      const response = await axios.get(`/api/interpretability/examples?${params}`)
      return response.data
    },
  })

  const filteredExamples = examples?.filter(example => 
    typeFilter === 'all' || example.example_type === typeFilter
  ) || []

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'normal':
        return 'success'
      case 'anomalous':
        return 'error'
      case 'borderline':
        return 'warning'
      default:
        return 'default'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'normal':
        return <CheckCircle />
      case 'anomalous':
        return <Warning />
      case 'borderline':
        return <Info />
      default:
        return <Info />
    }
  }

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
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

  const handleExampleClick = (example: ExamplePattern) => {
    setSelectedExample(example)
    onExampleSelect?.(example)
  }

  const handleCloseDialog = () => {
    setSelectedExample(null)
  }

  const getRoleSpecificGuidance = (example: ExamplePattern) => {
    switch (userRole) {
      case 'educator':
        return `Educational Context: ${example.educational_context}. Use this example to help students and parents understand developmental expectations.`
      case 'researcher':
        return `Research Significance: This pattern appears in ${example.metadata.prevalence}% of drawings in this age group. Developmental significance: ${example.metadata.developmental_significance}.`
      case 'parent':
        return `What this means: ${example.educational_context}. This type of drawing is ${example.example_type === 'normal' ? 'typical' : example.example_type === 'anomalous' ? 'unusual' : 'sometimes seen'} for children this age.`
      case 'clinician':
        return `Clinical Notes: ${example.metadata.developmental_significance}. Consider this pattern in the context of comprehensive developmental assessment.`
      default:
        return example.educational_context
    }
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Example Gallery
          </Typography>
          <Grid container spacing={2}>
            {[1, 2, 3, 4].map((i) => (
              <Grid item xs={12} sm={6} md={3} key={i}>
                <Card>
                  <Skeleton variant="rectangular" height={200} />
                  <CardContent>
                    <Skeleton variant="text" />
                    <Skeleton variant="text" width="60%" />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    )
  }

  if (error || !examples) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">
            Failed to load example patterns. Please try again later.
          </Alert>
        </CardContent>
      </Card>
    )
  }

  return (
    <>
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">
              Interpretation Examples
              {ageGroup && (
                <Chip label={`Age ${ageGroup}`} size="small" sx={{ ml: 1 }} />
              )}
            </Typography>
            <Box display="flex" gap={1}>
              <Button
                size="small"
                variant={typeFilter === 'all' ? 'contained' : 'outlined'}
                onClick={() => setTypeFilter('all')}
                startIcon={<FilterList />}
              >
                All ({examples.length})
              </Button>
              <Button
                size="small"
                variant={typeFilter === 'normal' ? 'contained' : 'outlined'}
                color="success"
                onClick={() => setTypeFilter('normal')}
              >
                Normal ({examples.filter(e => e.example_type === 'normal').length})
              </Button>
              <Button
                size="small"
                variant={typeFilter === 'anomalous' ? 'contained' : 'outlined'}
                color="error"
                onClick={() => setTypeFilter('anomalous')}
              >
                Anomalous ({examples.filter(e => e.example_type === 'anomalous').length})
              </Button>
            </Box>
          </Box>

          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="body2">
              These examples show common patterns and their interpretations. Click on any example to learn more about what the AI detected and why.
            </Typography>
          </Alert>

          <Grid container spacing={2}>
            {filteredExamples.map((example) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={example.pattern_id}>
                <Card
                  sx={{
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: 4,
                    },
                  }}
                  onClick={() => handleExampleClick(example)}
                >
                  <Box position="relative">
                    <CardMedia
                      component="img"
                      height="160"
                      image={example.image_url}
                      alt={example.pattern_name}
                    />
                    <Box
                      position="absolute"
                      top={8}
                      right={8}
                      display="flex"
                      gap={0.5}
                    >
                      <Chip
                        icon={getTypeIcon(example.example_type)}
                        label={example.example_type}
                        size="small"
                        color={getTypeColor(example.example_type) as any}
                        sx={{ backgroundColor: 'rgba(255,255,255,0.9)' }}
                      />
                    </Box>
                    <Box
                      position="absolute"
                      bottom={8}
                      left={8}
                    >
                      <Badge
                        badgeContent={example.metadata.drawing_count}
                        color="primary"
                        max={999}
                      >
                        <Chip
                          label={`${(example.metadata.prevalence * 100).toFixed(0)}%`}
                          size="small"
                          sx={{ backgroundColor: 'rgba(255,255,255,0.9)' }}
                        />
                      </Badge>
                    </Box>
                  </Box>
                  <CardContent sx={{ pb: 1 }}>
                    <Typography variant="subtitle2" gutterBottom noWrap>
                      {example.pattern_name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                    }}>
                      {example.description}
                    </Typography>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mt={1}>
                      <Chip
                        label={`${example.confidence_level} confidence`}
                        size="small"
                        color={getConfidenceColor(example.confidence_level) as any}
                        variant="outlined"
                      />
                      <Tooltip title="Click to view details">
                        <IconButton size="small">
                          <Visibility />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          {filteredExamples.length === 0 && (
            <Box textAlign="center" py={4}>
              <Typography variant="body1" color="text.secondary">
                No examples found for the selected filter.
              </Typography>
              <Button
                onClick={() => setTypeFilter('all')}
                sx={{ mt: 1 }}
              >
                Show All Examples
              </Button>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Detailed example dialog */}
      <Dialog
        open={!!selectedExample}
        onClose={handleCloseDialog}
        maxWidth="lg"
        fullWidth
        PaperProps={{
          sx: { minHeight: '70vh' }
        }}
      >
        {selectedExample && (
          <>
            <DialogTitle>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Box>
                  <Typography variant="h5" component="div">
                    {selectedExample.pattern_name}
                  </Typography>
                  <Box display="flex" gap={1} mt={1}>
                    <Chip
                      icon={getTypeIcon(selectedExample.example_type)}
                      label={selectedExample.example_type}
                      color={getTypeColor(selectedExample.example_type) as any}
                    />
                    <Chip
                      label={`Age ${selectedExample.age_group}`}
                      variant="outlined"
                    />
                    <Chip
                      label={`${selectedExample.confidence_level} confidence`}
                      color={getConfidenceColor(selectedExample.confidence_level) as any}
                      variant="outlined"
                    />
                  </Box>
                </Box>
                <IconButton onClick={handleCloseDialog}>
                  <Close />
                </IconButton>
              </Box>
            </DialogTitle>

            <DialogContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Original Drawing
                  </Typography>
                  <Paper sx={{ p: 1, mb: 2 }}>
                    <img
                      src={selectedExample.image_url}
                      alt={selectedExample.pattern_name}
                      style={{ width: '100%', height: 'auto', maxHeight: 300, objectFit: 'contain' }}
                    />
                  </Paper>
                  
                  <Typography variant="h6" gutterBottom>
                    Saliency Analysis
                  </Typography>
                  <Paper sx={{ p: 1 }}>
                    <img
                      src={selectedExample.saliency_url}
                      alt="Saliency map"
                      style={{ width: '100%', height: 'auto', maxHeight: 300, objectFit: 'contain' }}
                    />
                  </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
                    <Tab label="Overview" />
                    <Tab label="Features" />
                    <Tab label="Guidance" />
                  </Tabs>

                  <Box mt={2}>
                    {activeTab === 0 && (
                      <Box>
                        <Typography variant="body1" paragraph>
                          {selectedExample.description}
                        </Typography>

                        <Alert severity="info" sx={{ mb: 2 }}>
                          <Typography variant="body2">
                            {getRoleSpecificGuidance(selectedExample)}
                          </Typography>
                        </Alert>

                        <Paper sx={{ p: 2, backgroundColor: 'action.hover' }}>
                          <Typography variant="subtitle2" gutterBottom>
                            Pattern Statistics:
                          </Typography>
                          <List dense>
                            <ListItem>
                              <ListItemText
                                primary="Prevalence"
                                secondary={`${(selectedExample.metadata.prevalence * 100).toFixed(1)}% of drawings in this age group`}
                              />
                            </ListItem>
                            <ListItem>
                              <ListItemText
                                primary="Sample Size"
                                secondary={`Based on ${selectedExample.metadata.drawing_count} drawings`}
                              />
                            </ListItem>
                            <ListItem>
                              <ListItemText
                                primary="Developmental Significance"
                                secondary={selectedExample.metadata.developmental_significance}
                              />
                            </ListItem>
                          </List>
                        </Paper>
                      </Box>
                    )}

                    {activeTab === 1 && (
                      <Box>
                        <Typography variant="h6" gutterBottom>
                          Visual Features Detected
                        </Typography>
                        <List>
                          {selectedExample.visual_features.map((feature, index) => (
                            <ListItem key={index}>
                              <ListItemIcon>
                                <Visibility color="primary" />
                              </ListItemIcon>
                              <ListItemText primary={feature} />
                            </ListItem>
                          ))}
                        </List>

                        <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                          Interpretation Notes
                        </Typography>
                        <List>
                          {selectedExample.interpretation_notes.map((note, index) => (
                            <ListItem key={index}>
                              <ListItemIcon>
                                <Psychology color="secondary" />
                              </ListItemIcon>
                              <ListItemText primary={note} />
                            </ListItem>
                          ))}
                        </List>
                      </Box>
                    )}

                    {activeTab === 2 && (
                      <Box>
                        <Typography variant="h6" gutterBottom>
                          {userRole === 'educator' ? 'Educational Guidance' :
                           userRole === 'researcher' ? 'Research Applications' :
                           userRole === 'parent' ? 'What This Means' :
                           'Clinical Considerations'}
                        </Typography>

                        <Alert 
                          severity={selectedExample.example_type === 'normal' ? 'success' : 
                                   selectedExample.example_type === 'anomalous' ? 'warning' : 'info'}
                          sx={{ mb: 2 }}
                        >
                          <Typography variant="body2">
                            {getRoleSpecificGuidance(selectedExample)}
                          </Typography>
                        </Alert>

                        <Paper sx={{ p: 2, backgroundColor: 'action.hover' }}>
                          <Typography variant="subtitle2" gutterBottom>
                            💡 Key Takeaways:
                          </Typography>
                          <List dense>
                            <ListItem>
                              <ListItemIcon>
                                <Lightbulb color="primary" />
                              </ListItemIcon>
                              <ListItemText
                                primary="Pattern Recognition"
                                secondary="Learn to identify similar patterns in other drawings"
                              />
                            </ListItem>
                            <ListItem>
                              <ListItemIcon>
                                <Compare color="primary" />
                              </ListItemIcon>
                              <ListItemText
                                primary="Comparative Analysis"
                                secondary="Use this example as a reference point for similar cases"
                              />
                            </ListItem>
                            <ListItem>
                              <ListItemIcon>
                                <School color="primary" />
                              </ListItemIcon>
                              <ListItemText
                                primary="Educational Value"
                                secondary="Understand the developmental context and implications"
                              />
                            </ListItem>
                          </List>
                        </Paper>
                      </Box>
                    )}
                  </Box>
                </Grid>
              </Grid>
            </DialogContent>

            <DialogActions>
              <Button onClick={handleCloseDialog}>
                Close
              </Button>
              <Button
                variant="contained"
                onClick={() => {
                  // Could implement functionality to use this example as a reference
                  handleCloseDialog()
                }}
              >
                Use as Reference
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </>
  )
}

export default ExampleGallery