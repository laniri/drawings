import React, { useState } from 'react'
import {
  Typography,
  Paper,
  Box,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Button,
  Chip,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Container,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material'
import {
  GitHub,
  Info,
  Assessment,
  Visibility,
  ExpandMore,
  Psychology,
  School,
  Security,
  Close,
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'

interface DemoSample {
  id: number
  title: string
  description: string
  age_group: string
  original_image: string
  saliency_map: string
  composite_image: string
  analysis_result: {
    anomaly_score: number
    is_anomaly: boolean
    confidence: number
    processing_time: number
    model_version: string
    age_group_model: string
  }
  interpretability: {
    explanation: string
    key_regions: Array<{
      region: string
      importance: number
      description: string
    }>
    technical_details: {
      saliency_method: string
      attention_regions: number
      confidence_threshold: number
    }
  }
  metadata: {
    created_at: string
    content_rating: string
    educational_value: string
  }
}

interface ProjectInfo {
  title: string
  overview: string
  description: string
  version: string
  features: string[]
  technical_stack: string[]
  research_context: string
}

const DemoPage: React.FC = () => {
  const [selectedSample, setSelectedSample] = useState<DemoSample | null>(null)
  const [detailsOpen, setDetailsOpen] = useState(false)

  // Fetch demo samples
  const { data: samplesData, isLoading: samplesLoading } = useQuery({
    queryKey: ['demo-samples'],
    queryFn: async () => {
      const response = await axios.get('/demo/samples')
      return response.data.data.samples as DemoSample[]
    },
  })

  // Fetch project info
  const { data: projectInfo } = useQuery({
    queryKey: ['demo-project-info'],
    queryFn: async () => {
      const response = await axios.get('/demo/project-info')
      return response.data.data as ProjectInfo
    },
  })

  // Fetch statistics
  const { data: statistics } = useQuery({
    queryKey: ['demo-statistics'],
    queryFn: async () => {
      const response = await axios.get('/demo/statistics')
      return response.data.data
    },
  })

  const handleViewDetails = (sample: DemoSample) => {
    setSelectedSample(sample)
    setDetailsOpen(true)
  }

  const handleCloseDetails = () => {
    setDetailsOpen(false)
    setSelectedSample(null)
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        {/* Header Section */}
        <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Typography variant="h2" component="h1" gutterBottom>
            Children's Drawing Anomaly Detection
          </Typography>
          <Typography variant="h5" color="text.secondary" paragraph>
            AI-powered analysis of children's drawings to identify developmental
            patterns
          </Typography>

          {projectInfo && (
            <Typography
              variant="body1"
              color="text.secondary"
              paragraph
              sx={{ maxWidth: 800, mx: 'auto' }}
            >
              {projectInfo.overview}
            </Typography>
          )}

          <Box
            sx={{
              mt: 3,
              display: 'flex',
              gap: 2,
              justifyContent: 'center',
              flexWrap: 'wrap',
            }}
          >
            {/* Links removed - GitHub link kept only at bottom */}
          </Box>
        </Box>

        {/* Medical Disclaimer */}
        <Alert
          severity="warning"
          sx={{
            mb: 4,
            textAlign: 'left',
            backgroundColor: '#ffebee',
            borderColor: '#f44336',
          }}
        >
          <Typography variant="h6" gutterBottom>
            ⚠️ IMPORTANT MEDICAL DISCLAIMER
          </Typography>
          <Typography variant="body2">
            This is a demonstration system only and is NOT intended for medical
            diagnosis. Results should never be used as a substitute for
            professional medical advice, diagnosis, or treatment. Always consult
            with qualified healthcare professionals for developmental concerns.
          </Typography>
        </Alert>

        {/* Demo Samples Section */}
        <Paper sx={{ p: 4, mb: 4 }}>
          <Typography variant="h4" gutterBottom>
            Interactive Demo Samples
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            Explore these pre-analyzed sample drawings to understand how our AI
            system works. Each sample includes the original drawing, AI analysis
            results, and interpretability visualizations.
          </Typography>

          {samplesLoading ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography>Loading demo samples...</Typography>
            </Box>
          ) : (
            <Grid container spacing={3}>
              {samplesData?.map((sample) => (
                <Grid item xs={12} md={6} lg={4} key={sample.id}>
                  <Card
                    sx={{
                      height: '100%',
                      cursor: 'pointer',
                      transition: 'transform 0.2s',
                      '&:hover': { transform: 'translateY(-4px)' },
                    }}
                  >
                    <CardMedia
                      component="img"
                      height="200"
                      image={
                        sample.original_image ||
                        'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjVmNWY1Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkRyYXdpbmcgTm90IEZvdW5kPC90ZXh0Pjwvc3ZnPg=='
                      }
                      alt={sample.title}
                      sx={{ objectFit: 'contain', bgcolor: 'grey.100' }}
                      onError={(e) => {
                        const target = e.target as HTMLImageElement
                        target.src =
                          'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjVmNWY1Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkRyYXdpbmcgTm90IEZvdW5kPC90ZXh0Pjwvc3ZnPg=='
                      }}
                    />
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        {sample.title}
                      </Typography>
                      <Typography
                        variant="body2"
                        color="text.secondary"
                        paragraph
                      >
                        {sample.description}
                      </Typography>

                      <Box
                        sx={{
                          display: 'flex',
                          gap: 1,
                          mb: 2,
                          flexWrap: 'wrap',
                        }}
                      >
                        <Chip
                          label={`Age ${sample.age_group}`}
                          size="small"
                          color="primary"
                        />
                        <Chip
                          label={
                            sample.analysis_result.is_anomaly
                              ? 'Anomaly'
                              : 'Normal'
                          }
                          size="small"
                          color={
                            sample.analysis_result.is_anomaly
                              ? 'error'
                              : 'success'
                          }
                        />
                        <Chip
                          label={`${(sample.analysis_result.confidence * 100).toFixed(0)}% confidence`}
                          size="small"
                          variant="outlined"
                        />
                      </Box>

                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<Visibility />}
                        fullWidth
                        onClick={() => handleViewDetails(sample)}
                      >
                        View Analysis Details
                      </Button>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </Paper>

        {/* System Statistics */}
        {statistics && (
          <Paper sx={{ p: 4, mb: 4 }}>
            <Typography variant="h4" gutterBottom>
              System Capabilities
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h3" color="primary">
                    {statistics.total_samples || '3'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Demo Samples
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h3" color="primary">
                    8
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Age Group Models
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h3" color="primary">
                    64
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Subject Categories
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h3" color="primary">
                    832
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Feature Dimensions
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        )}

        {/* Technical Information */}
        <Paper sx={{ p: 4, mb: 4 }}>
          <Typography variant="h4" gutterBottom>
            How It Works
          </Typography>

          <Grid container spacing={4}>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center', mb: 2 }}>
                <Psychology
                  sx={{ fontSize: 48, color: 'primary.main', mb: 1 }}
                />
                <Typography variant="h6" gutterBottom>
                  AI Vision Analysis
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Uses Vision Transformer (ViT) models to extract
                  768-dimensional visual features from drawings
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center', mb: 2 }}>
                <School sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
                <Typography variant="h6" gutterBottom>
                  Age-Aware Models
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Separate models trained for different age groups (2-3, 3-4,
                  4-5, 5-6, 6-7, 7-8, 8-9, 9-12 years)
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center', mb: 2 }}>
                <Assessment
                  sx={{ fontSize: 48, color: 'primary.main', mb: 1 }}
                />
                <Typography variant="h6" gutterBottom>
                  Interpretable Results
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Provides saliency maps and explanations showing which parts of
                  the drawing influenced the analysis
                </Typography>
              </Box>
            </Grid>
          </Grid>

          <Box sx={{ mt: 4 }}>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">Technical Stack</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <List dense>
                  <ListItem>
                    <ListItemIcon>
                      <Security fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Python 3.11+ with FastAPI web framework" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Security fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="PyTorch 2.2.2+ for deep learning models" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Security fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Vision Transformer (ViT) for feature extraction" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Security fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="React 18 with TypeScript frontend" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Security fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Material-UI component library" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Security fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="SQLAlchemy ORM with SQLite database" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Security fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="AWS deployment with Docker containers" />
                  </ListItem>
                </List>
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">Key Features</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <List dense>
                  <ListItem>
                    <ListItemIcon>
                      <Info fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Subject-aware modeling with 64 predefined categories" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Info fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Hybrid embeddings (832-dimensional vectors)" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Info fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Age-based modeling for different developmental stages" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Info fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Interactive interpretability with saliency maps" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Info fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Real-time analysis with comprehensive results" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Info fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Multi-format export capabilities (PNG, PDF, JSON, CSV)" />
                  </ListItem>
                </List>
              </AccordionDetails>
            </Accordion>
          </Box>
        </Paper>

        {/* Technical Links */}
        <Paper sx={{ p: 4, mb: 4 }}>
          <Typography variant="h4" gutterBottom>
            Learn More
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={6}>
              <Button
                variant="outlined"
                fullWidth
                startIcon={<GitHub />}
                href="https://github.com/laniri/drawings"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub Repository
              </Button>
            </Grid>
            <Grid item xs={12} sm={6} md={6}>
              <Button
                variant="outlined"
                fullWidth
                startIcon={<Assessment />}
                href="/documentation"
              >
                Technical Documentation
              </Button>
            </Grid>
          </Grid>
        </Paper>

        {/* Sample Detail Modal */}
        <Dialog
          open={detailsOpen}
          onClose={handleCloseDetails}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <Typography variant="h6">{selectedSample?.title}</Typography>
              <Button onClick={handleCloseDetails} size="small">
                <Close />
              </Button>
            </Box>
          </DialogTitle>
          <DialogContent>
            {selectedSample && (
              <Box>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Original Drawing
                    </Typography>
                    <Box
                      component="img"
                      src={
                        selectedSample.original_image ||
                        'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjVmNWY1Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk9yaWdpbmFsIERyYXdpbmc8L3RleHQ+PC9zdmc+'
                      }
                      alt="Original Drawing"
                      sx={{
                        width: '100%',
                        maxHeight: 300,
                        objectFit: 'contain',
                        border: '1px solid #ddd',
                        borderRadius: 1,
                      }}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Interpretability Map
                    </Typography>
                    <Box
                      component="img"
                      src={
                        selectedSample.saliency_map ||
                        'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjVmNWY1Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPlNhbGllbmN5IE1hcDwvdGV4dD48L3N2Zz4='
                      }
                      alt="Saliency Map"
                      sx={{
                        width: '100%',
                        maxHeight: 300,
                        objectFit: 'contain',
                        border: '1px solid #ddd',
                        borderRadius: 1,
                      }}
                    />
                  </Grid>
                </Grid>

                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Analysis Results
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6} sm={3}>
                      <Paper sx={{ p: 2, textAlign: 'center' }}>
                        <Typography variant="body2" color="text.secondary">
                          Age Group
                        </Typography>
                        <Typography variant="h6">
                          {selectedSample.age_group}
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Paper sx={{ p: 2, textAlign: 'center' }}>
                        <Typography variant="body2" color="text.secondary">
                          Anomaly Score
                        </Typography>
                        <Typography variant="h6">
                          {selectedSample.analysis_result.anomaly_score.toFixed(
                            3
                          )}
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Paper sx={{ p: 2, textAlign: 'center' }}>
                        <Typography variant="body2" color="text.secondary">
                          Confidence
                        </Typography>
                        <Typography variant="h6">
                          {(
                            selectedSample.analysis_result.confidence * 100
                          ).toFixed(0)}
                          %
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Paper sx={{ p: 2, textAlign: 'center' }}>
                        <Typography variant="body2" color="text.secondary">
                          Classification
                        </Typography>
                        <Typography
                          variant="h6"
                          color={
                            selectedSample.analysis_result.is_anomaly
                              ? 'error.main'
                              : 'success.main'
                          }
                        >
                          {selectedSample.analysis_result.is_anomaly
                            ? 'Anomaly'
                            : 'Normal'}
                        </Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                </Box>

                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    AI Interpretation
                  </Typography>
                  <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="body1">
                      {selectedSample.interpretability.explanation}
                    </Typography>
                  </Paper>
                </Box>

                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Key Regions Analyzed
                  </Typography>
                  <List>
                    {selectedSample.interpretability.key_regions.map(
                      (region, index) => (
                        <ListItem key={index} divider>
                          <ListItemText
                            primary={
                              <Box
                                sx={{
                                  display: 'flex',
                                  justifyContent: 'space-between',
                                  alignItems: 'center',
                                }}
                              >
                                <Typography variant="subtitle2">
                                  {region.region}
                                </Typography>
                                <Chip
                                  label={`${(region.importance * 100).toFixed(0)}% importance`}
                                  size="small"
                                  color="primary"
                                  variant="outlined"
                                />
                              </Box>
                            }
                            secondary={region.description}
                          />
                        </ListItem>
                      )
                    )}
                  </List>
                </Box>
              </Box>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseDetails}>Close</Button>
          </DialogActions>
        </Dialog>

        {/* Research Context */}
        <Paper
          sx={{
            p: 4,
            textAlign: 'center',
            bgcolor: 'primary.main',
            color: 'white',
          }}
        >
          <Typography variant="h4" gutterBottom>
            Research & Educational Purpose
          </Typography>
          <Typography variant="body1" paragraph>
            This demonstration showcases AI techniques for analyzing children's
            drawings in a research context. The system is designed for academic
            exploration and should not be used for professional assessments.
          </Typography>
          <Box
            sx={{
              display: 'flex',
              gap: 2,
              justifyContent: 'center',
              flexWrap: 'wrap',
            }}
          >
            <Button
              variant="contained"
              size="large"
              sx={{
                bgcolor: 'white',
                color: 'primary.main',
                '&:hover': { bgcolor: 'grey.100' },
              }}
              startIcon={<GitHub />}
              href="https://github.com/laniri/drawings"
              target="_blank"
              rel="noopener noreferrer"
            >
              Explore Source Code
            </Button>
          </Box>
        </Paper>
      </Box>
    </Container>
  )
}

export default DemoPage
