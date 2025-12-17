import React, { useState } from 'react'
import {
  Typography,
  Paper,
  Box,
  Grid,
  Card,
  CardContent,
  Button,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Checkbox,
  TextField,
  Alert,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  CircularProgress,
} from '@mui/material'
import {
  Description,
  PlayArrow,
  Stop,
  CheckCircle,
  Error,
  Warning,
  Visibility,
  Delete,
  Search,
  Assessment,
  Build,
  Schedule,
} from '@mui/icons-material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'

interface DocumentationStatus {
  is_generating: boolean
  current_task: string | null
  progress: number
  start_time: string | null
  last_update: string | null
  errors: string[]
  warnings: string[]
}

interface DocumentationMetrics {
  total_files: number
  last_generated: string | null
  generation_count: number
  average_duration: number
  success_rate: number
  file_breakdown: Record<string, number>
  validation_status: {
    is_valid: boolean
    total_files: number
    validated_files: number
    errors: number
    warnings: number
    categories: Record<string, any>
  }
}

interface DocumentationFile {
  path: string
  name: string
  title: string
  category: string
  size: number
  modified: string
  url: string
}

interface DocumentationCategory {
  name: string
  display_name: string
  description: string
}

interface GenerationRequest {
  categories?: string[]
  force: boolean
  validate: boolean
}

const DocumentationPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0)
  const [generateDialogOpen, setGenerateDialogOpen] = useState(false)
  const [selectedCategories, setSelectedCategories] = useState<string[]>([])
  const [forceGeneration, setForceGeneration] = useState(false)
  const [validateAfterGeneration, setValidateAfterGeneration] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('')
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false)
  const [previewCategory, setPreviewCategory] = useState('')
  const [batchDialogOpen, setBatchDialogOpen] = useState(false)
  const [batchRequests, setBatchRequests] = useState<any[]>([])
  const [scheduleDialogOpen, setScheduleDialogOpen] = useState(false)
  const [scheduleTime, setScheduleTime] = useState('')

  const queryClient = useQueryClient()



  // Fetch documentation status
  const { data: status, isLoading: statusLoading } = useQuery<DocumentationStatus>({
    queryKey: ['documentation-status'],
    queryFn: async () => {
      const response = await axios.get('/api/documentation/status')
      return response.data
    },
    refetchInterval: 5000, // Poll every 5 seconds
  })

  // Fetch documentation metrics
  const { data: metrics, isLoading: metricsLoading } = useQuery<DocumentationMetrics>({
    queryKey: ['documentation-metrics'],
    queryFn: async () => {
      const response = await axios.get('/api/documentation/metrics')
      return response.data
    },
    refetchInterval: 30000,
  })

  // Fetch documentation files
  const { data: filesData } = useQuery({
    queryKey: ['documentation-files', categoryFilter, searchTerm],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (categoryFilter) params.append('category', categoryFilter)
      if (searchTerm) params.append('search', searchTerm)
      
      const response = await axios.get(`/api/documentation/files?${params}`)
      return response.data
    },
  })

  // Fetch documentation categories
  const { data: categoriesData } = useQuery({
    queryKey: ['documentation-categories'],
    queryFn: async () => {
      const response = await axios.get('/api/documentation/categories')
      return response.data
    },
  })

  // Generate documentation mutation
  const generateMutation = useMutation({
    mutationFn: async (request: GenerationRequest) => {
      const response = await axios.post('/api/documentation/generate', request)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documentation-status'] })
      queryClient.invalidateQueries({ queryKey: ['documentation-metrics'] })
      queryClient.invalidateQueries({ queryKey: ['documentation-files'] })
      setGenerateDialogOpen(false)
    },
  })

  // Clear cache mutation
  const clearCacheMutation = useMutation({
    mutationFn: async () => {
      const response = await axios.delete('/api/documentation/cache')
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documentation-metrics'] })
    },
  })

  // Validate documentation mutation
  const validateMutation = useMutation({
    mutationFn: async (categories?: string[]) => {
      const response = await axios.post('/api/documentation/validate', { categories })
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documentation-metrics'] })
    },
  })

  // Preview documentation mutation
  const previewMutation = useMutation({
    mutationFn: async (category: string) => {
      const response = await axios.get(`/api/documentation/preview/${category}`)
      return response.data
    },
  })

  // Batch generate mutation
  const batchGenerateMutation = useMutation({
    mutationFn: async (batchData: any) => {
      const response = await axios.post('/api/documentation/batch/generate', batchData)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documentation-status'] })
      queryClient.invalidateQueries({ queryKey: ['documentation-metrics'] })
      setBatchDialogOpen(false)
    },
  })

  // Schedule generation mutation
  const scheduleMutation = useMutation({
    mutationFn: async (scheduleData: any) => {
      const response = await axios.post('/api/documentation/schedule', scheduleData)
      return response.data
    },
    onSuccess: () => {
      setScheduleDialogOpen(false)
    },
  })

  const handleGenerate = () => {
    const request: GenerationRequest = {
      categories: selectedCategories.length > 0 ? selectedCategories : undefined,
      force: forceGeneration,
      validate: validateAfterGeneration,
    }
    generateMutation.mutate(request)
  }

  const handleClearCache = () => {
    if (window.confirm('Are you sure you want to clear the documentation cache? This will force regeneration of all documentation.')) {
      clearCacheMutation.mutate()
    }
  }

  const handleValidate = () => {
    validateMutation.mutate(undefined)
  }

  const handlePreview = (category: string) => {
    setPreviewCategory(category)
    previewMutation.mutate(category)
    setPreviewDialogOpen(true)
  }

  const handleBatchGenerate = () => {
    const batchData = {
      batch_requests: batchRequests,
      schedule_delay: 0
    }
    batchGenerateMutation.mutate(batchData)
  }

  const handleScheduleGenerate = () => {
    const scheduleData = {
      schedule_time: scheduleTime,
      categories: selectedCategories,
      force: forceGeneration,
      validate: validateAfterGeneration
    }
    scheduleMutation.mutate(scheduleData)
  }

  const addBatchRequest = () => {
    setBatchRequests([...batchRequests, {
      name: `Batch ${batchRequests.length + 1}`,
      categories: [],
      force: false,
      validate: true,
      delay: 0
    }])
  }

  const updateBatchRequest = (index: number, updates: any) => {
    const updated = [...batchRequests]
    updated[index] = { ...updated[index], ...updates }
    setBatchRequests(updated)
  }

  const removeBatchRequest = (index: number) => {
    setBatchRequests(batchRequests.filter((_, i) => i !== index))
  }

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}m ${remainingSeconds.toFixed(0)}s`
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const getStatusColor = (isGenerating: boolean, hasErrors: boolean): 'info' | 'error' | 'success' => {
    if (isGenerating) return 'info'
    if (hasErrors) return 'error'
    return 'success'
  }

  const getValidationColor = (isValid: boolean, errors: number): 'error' | 'warning' | 'success' => {
    if (errors > 0) return 'error'
    if (!isValid) return 'warning'
    return 'success'
  }

  if (statusLoading || metricsLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    )
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          Documentation Management
        </Typography>
        <Box display="flex" gap={2}>
          <Button
            variant="outlined"
            startIcon={<Assessment />}
            onClick={handleValidate}
            disabled={status?.is_generating || validateMutation.isPending}
          >
            Validate
          </Button>
          <Button
            variant="outlined"
            startIcon={<Delete />}
            onClick={handleClearCache}
            disabled={status?.is_generating || clearCacheMutation.isPending}
          >
            Clear Cache
          </Button>
          <Button
            variant="outlined"
            startIcon={<Build />}
            onClick={() => setBatchDialogOpen(true)}
            disabled={status?.is_generating}
          >
            Batch Generate
          </Button>
          <Button
            variant="outlined"
            startIcon={<Schedule />}
            onClick={() => setScheduleDialogOpen(true)}
            disabled={status?.is_generating}
          >
            Schedule
          </Button>
          <Button
            variant="contained"
            startIcon={status?.is_generating ? <Stop /> : <PlayArrow />}
            onClick={() => setGenerateDialogOpen(true)}
            disabled={status?.is_generating}
            color={status?.is_generating ? 'secondary' : 'primary'}
          >
            {status?.is_generating ? 'Generating...' : 'Generate Docs'}
          </Button>
        </Box>
      </Box>

      {/* Status Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Generation Status</Typography>
            <Chip
              label={status?.is_generating ? 'Generating' : 'Idle'}
              color={getStatusColor(status?.is_generating || false, (status?.errors?.length || 0) > 0)}
              icon={status?.is_generating ? <CircularProgress size={16} /> : <CheckCircle />}
            />
          </Box>

          {status?.is_generating && (
            <Box mb={2}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="body2" color="text.secondary">
                  {status.current_task || 'Processing...'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {status.progress}%
                </Typography>
              </Box>
              <LinearProgress variant="determinate" value={status.progress} />
            </Box>
          )}

          {status?.errors && status.errors.length > 0 && (
            <Alert severity="error" sx={{ mb: 2 }}>
              <Typography variant="subtitle2">Errors:</Typography>
              {status.errors.map((error, index) => (
                <Typography key={index} variant="body2">• {error}</Typography>
              ))}
            </Alert>
          )}

          {status?.warnings && status.warnings.length > 0 && (
            <Alert severity="warning">
              <Typography variant="subtitle2">Warnings:</Typography>
              {status.warnings.map((warning, index) => (
                <Typography key={index} variant="body2">• {warning}</Typography>
              ))}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Metrics Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Total Files
                  </Typography>
                  <Typography variant="h4">
                    {metrics?.total_files || 0}
                  </Typography>
                </Box>
                <Description color="primary" sx={{ fontSize: 40 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Success Rate
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    {metrics?.success_rate?.toFixed(1) || 100}%
                  </Typography>
                </Box>
                <CheckCircle color="success" sx={{ fontSize: 40 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Avg Duration
                  </Typography>
                  <Typography variant="h4">
                    {metrics?.average_duration ? formatDuration(metrics.average_duration) : 'N/A'}
                  </Typography>
                </Box>
                <Schedule color="info" sx={{ fontSize: 40 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Validation
                  </Typography>
                  <Typography 
                    variant="h4" 
                    color={getValidationColor(
                      metrics?.validation_status?.is_valid || false,
                      metrics?.validation_status?.errors || 0
                    )}
                  >
                    {metrics?.validation_status?.errors || 0} errors
                  </Typography>
                </Box>
                <Assessment 
                  color={getValidationColor(
                    metrics?.validation_status?.is_valid || false,
                    metrics?.validation_status?.errors || 0
                  )} 
                  sx={{ fontSize: 40 }} 
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs for different views */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, newValue: number) => setActiveTab(newValue)}>
          <Tab label="Files" />
          <Tab label="Categories" />
          <Tab label="Validation" />
        </Tabs>

        {/* Files Tab */}
        {activeTab === 0 && (
          <Box p={3}>
            <Box display="flex" gap={2} mb={3}>
              <TextField
                label="Search files"
                variant="outlined"
                size="small"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: <Search />,
                }}
                sx={{ minWidth: 300 }}
              />
              <FormControl size="small" sx={{ minWidth: 200 }}>
                <InputLabel>Category</InputLabel>
                <Select
                  value={categoryFilter}
                  onChange={(e) => setCategoryFilter(e.target.value)}
                  label="Category"
                >
                  <MenuItem value="">All Categories</MenuItem>
                  {Object.keys(metrics?.file_breakdown || {}).map((category) => (
                    <MenuItem key={category} value={category}>
                      {category} ({metrics?.file_breakdown[category]})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>

            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Category</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell>Modified</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filesData?.files?.map((file: DocumentationFile) => (
                    <TableRow key={file.path}>
                      <TableCell>
                        <Box>
                          <Typography variant="body2" fontWeight="medium">
                            {file.title}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {file.path}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip label={file.category} size="small" />
                      </TableCell>
                      <TableCell>{formatFileSize(file.size)}</TableCell>
                      <TableCell>
                        {new Date(file.modified).toLocaleString()}
                      </TableCell>
                      <TableCell>
                        <Tooltip title="View file">
                          <IconButton
                            size="small"
                            onClick={() => window.open(file.url, '_blank')}
                          >
                            <Visibility />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}

        {/* Categories Tab */}
        {activeTab === 1 && (
          <Box p={3}>
            <Grid container spacing={2}>
              {categoriesData?.categories?.map((category: DocumentationCategory) => (
                <Grid item xs={12} md={6} key={category.name}>
                  <Card>
                    <CardContent>
                      <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
                        <Box>
                          <Typography variant="h6">{category.display_name}</Typography>
                          <Typography variant="body2" color="text.secondary">
                            {category.description}
                          </Typography>
                        </Box>
                        <Chip 
                          label={metrics?.file_breakdown[category.name] || 0} 
                          size="small" 
                        />
                      </Box>
                      <Box display="flex" gap={1}>
                        <Button
                          size="small"
                          variant="outlined"
                          startIcon={<Visibility />}
                          onClick={() => handlePreview(category.name)}
                          disabled={status?.is_generating}
                        >
                          Preview
                        </Button>
                        <Button
                          size="small"
                          variant="outlined"
                          startIcon={<Build />}
                          onClick={() => {
                            setSelectedCategories([category.name])
                            setGenerateDialogOpen(true)
                          }}
                          disabled={status?.is_generating}
                        >
                          Generate
                        </Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}

        {/* Validation Tab */}
        {activeTab === 2 && (
          <Box p={3}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Overall Status
                    </Typography>
                    <Box display="flex" alignItems="center" gap={2} mb={2}>
                      <Chip
                        label={metrics?.validation_status?.is_valid ? 'Valid' : 'Issues Found'}
                        color={getValidationColor(
                          metrics?.validation_status?.is_valid || false,
                          metrics?.validation_status?.errors || 0
                        )}
                        icon={metrics?.validation_status?.is_valid ? <CheckCircle /> : <Warning />}
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {metrics?.validation_status?.validated_files || 0} of {metrics?.validation_status?.total_files || 0} files validated
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Issues Summary
                    </Typography>
                    <Box display="flex" gap={2}>
                      <Chip
                        label={`${metrics?.validation_status?.errors || 0} Errors`}
                        color="error"
                        size="small"
                        icon={<Error />}
                      />
                      <Chip
                        label={`${metrics?.validation_status?.warnings || 0} Warnings`}
                        color="warning"
                        size="small"
                        icon={<Warning />}
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Category Breakdown
                    </Typography>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Category</TableCell>
                            <TableCell>Files</TableCell>
                            <TableCell>Validated</TableCell>
                            <TableCell>Errors</TableCell>
                            <TableCell>Warnings</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(metrics?.validation_status?.categories || {}).map(([category, data]: [string, any]) => (
                            <TableRow key={category}>
                              <TableCell>{category}</TableCell>
                              <TableCell>{data.files}</TableCell>
                              <TableCell>{data.validated}</TableCell>
                              <TableCell>
                                <Chip label={data.errors} color={data.errors > 0 ? 'error' : 'default'} size="small" />
                              </TableCell>
                              <TableCell>
                                <Chip label={data.warnings} color={data.warnings > 0 ? 'warning' : 'default'} size="small" />
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )}
      </Paper>

      {/* Generation Dialog */}
      <Dialog open={generateDialogOpen} onClose={() => setGenerateDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Generate Documentation</DialogTitle>
        <DialogContent>
          <Box mb={3}>
            <Typography variant="subtitle1" gutterBottom>
              Categories to Generate
            </Typography>
            <Typography variant="body2" color="text.secondary" mb={2}>
              Select specific categories or leave empty to generate all documentation.
            </Typography>
            <Grid container spacing={1}>
              {categoriesData?.categories?.map((category: DocumentationCategory) => (
                <Grid item xs={12} sm={6} key={category.name}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={selectedCategories.includes(category.name)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedCategories([...selectedCategories, category.name])
                          } else {
                            setSelectedCategories(selectedCategories.filter(c => c !== category.name))
                          }
                        }}
                      />
                    }
                    label={
                      <Box>
                        <Typography variant="body2">{category.display_name}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {category.description}
                        </Typography>
                      </Box>
                    }
                  />
                </Grid>
              ))}
            </Grid>
          </Box>

          <Box mb={2}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={forceGeneration}
                  onChange={(e) => setForceGeneration(e.target.checked)}
                />
              }
              label="Force regeneration (ignore cache)"
            />
          </Box>

          <Box mb={2}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={validateAfterGeneration}
                  onChange={(e) => setValidateAfterGeneration(e.target.checked)}
                />
              }
              label="Validate after generation"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setGenerateDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleGenerate}
            variant="contained"
            disabled={generateMutation.isPending}
            startIcon={generateMutation.isPending ? <CircularProgress size={16} /> : <PlayArrow />}
          >
            {generateMutation.isPending ? 'Starting...' : 'Generate'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Preview Dialog */}
      <Dialog open={previewDialogOpen} onClose={() => setPreviewDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Documentation Preview - {previewCategory}</DialogTitle>
        <DialogContent>
          {previewMutation.isPending ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : previewMutation.data ? (
            <Box>
              <Typography variant="h6" gutterBottom>
                Files to Generate
              </Typography>
              <List dense>
                {previewMutation.data.preview.files_to_generate.map((file: string) => (
                  <ListItem key={file}>
                    <ListItemIcon>
                      <Description />
                    </ListItemIcon>
                    <ListItemText primary={file} />
                  </ListItem>
                ))}
              </List>

              {previewMutation.data.preview.changes_detected.length > 0 && (
                <Box mt={2}>
                  <Typography variant="h6" gutterBottom>
                    Changes Detected
                  </Typography>
                  <List dense>
                    {previewMutation.data.preview.changes_detected.map((change: any, index: number) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <Warning color={change.type === 'modified' ? 'warning' : 'info'} />
                        </ListItemIcon>
                        <ListItemText 
                          primary={change.path}
                          secondary={`${change.type} - ${new Date(change.timestamp).toLocaleString()}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}

              {previewMutation.data.preview.dependencies.length > 0 && (
                <Box mt={2}>
                  <Typography variant="h6" gutterBottom>
                    Dependencies
                  </Typography>
                  <Box display="flex" gap={1} flexWrap="wrap">
                    {previewMutation.data.preview.dependencies.map((dep: string) => (
                      <Chip key={dep} label={dep} size="small" />
                    ))}
                  </Box>
                </Box>
              )}
            </Box>
          ) : (
            <Typography>No preview data available</Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewDialogOpen(false)}>
            Close
          </Button>
          <Button
            variant="contained"
            onClick={() => {
              setSelectedCategories([previewCategory])
              setPreviewDialogOpen(false)
              setGenerateDialogOpen(true)
            }}
            disabled={status?.is_generating}
          >
            Generate Now
          </Button>
        </DialogActions>
      </Dialog>

      {/* Batch Generation Dialog */}
      <Dialog open={batchDialogOpen} onClose={() => setBatchDialogOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle>Batch Documentation Generation</DialogTitle>
        <DialogContent>
          <Box mb={2}>
            <Button
              variant="outlined"
              onClick={addBatchRequest}
              startIcon={<PlayArrow />}
            >
              Add Batch Request
            </Button>
          </Box>

          {batchRequests.map((request, index) => (
            <Card key={index} sx={{ mb: 2 }}>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <TextField
                    label="Batch Name"
                    value={request.name}
                    onChange={(e) => updateBatchRequest(index, { name: e.target.value })}
                    size="small"
                  />
                  <IconButton
                    onClick={() => removeBatchRequest(index)}
                    color="error"
                  >
                    <Delete />
                  </IconButton>
                </Box>

                <Box mb={2}>
                  <Typography variant="subtitle2" gutterBottom>
                    Categories
                  </Typography>
                  <Grid container spacing={1}>
                    {categoriesData?.categories?.map((category: DocumentationCategory) => (
                      <Grid item xs={12} sm={6} key={category.name}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={request.categories.includes(category.name)}
                              onChange={(e) => {
                                const categories = e.target.checked
                                  ? [...request.categories, category.name]
                                  : request.categories.filter((c: string) => c !== category.name)
                                updateBatchRequest(index, { categories })
                              }}
                            />
                          }
                          label={category.display_name}
                        />
                      </Grid>
                    ))}
                  </Grid>
                </Box>

                <Box display="flex" gap={2}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={request.force}
                        onChange={(e) => updateBatchRequest(index, { force: e.target.checked })}
                      />
                    }
                    label="Force"
                  />
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={request.validate}
                        onChange={(e) => updateBatchRequest(index, { validate: e.target.checked })}
                      />
                    }
                    label="Validate"
                  />
                  <TextField
                    label="Delay (seconds)"
                    type="number"
                    value={request.delay}
                    onChange={(e) => updateBatchRequest(index, { delay: parseInt(e.target.value) || 0 })}
                    size="small"
                    sx={{ width: 150 }}
                  />
                </Box>
              </CardContent>
            </Card>
          ))}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBatchDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleBatchGenerate}
            variant="contained"
            disabled={batchRequests.length === 0 || batchGenerateMutation.isPending}
            startIcon={batchGenerateMutation.isPending ? <CircularProgress size={16} /> : <PlayArrow />}
          >
            {batchGenerateMutation.isPending ? 'Starting...' : 'Start Batch'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Schedule Dialog */}
      <Dialog open={scheduleDialogOpen} onClose={() => setScheduleDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Schedule Documentation Generation</DialogTitle>
        <DialogContent>
          <Box mb={3}>
            <TextField
              label="Schedule Time"
              type="datetime-local"
              value={scheduleTime}
              onChange={(e) => setScheduleTime(e.target.value)}
              fullWidth
              InputLabelProps={{
                shrink: true,
              }}
            />
          </Box>

          <Box mb={2}>
            <Typography variant="subtitle1" gutterBottom>
              Categories to Generate
            </Typography>
            <Grid container spacing={1}>
              {categoriesData?.categories?.map((category: DocumentationCategory) => (
                <Grid item xs={12} sm={6} key={category.name}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={selectedCategories.includes(category.name)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedCategories([...selectedCategories, category.name])
                          } else {
                            setSelectedCategories(selectedCategories.filter(c => c !== category.name))
                          }
                        }}
                      />
                    }
                    label={category.display_name}
                  />
                </Grid>
              ))}
            </Grid>
          </Box>

          <Box display="flex" gap={2}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={forceGeneration}
                  onChange={(e) => setForceGeneration(e.target.checked)}
                />
              }
              label="Force regeneration"
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={validateAfterGeneration}
                  onChange={(e) => setValidateAfterGeneration(e.target.checked)}
                />
              }
              label="Validate after generation"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setScheduleDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleScheduleGenerate}
            variant="contained"
            disabled={!scheduleTime || scheduleMutation.isPending}
            startIcon={scheduleMutation.isPending ? <CircularProgress size={16} /> : <Schedule />}
          >
            {scheduleMutation.isPending ? 'Scheduling...' : 'Schedule'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default DocumentationPage