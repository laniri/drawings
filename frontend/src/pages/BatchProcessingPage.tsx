import React, { useState, useCallback } from 'react'
import {
  Typography,
  Box,
  Button,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  IconButton,
  LinearProgress,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material'
import {
  CloudUpload,
  Delete,
  PlayArrow,
  Download,
  Image as ImageIcon,
  CheckCircle,
  Warning,
  Error,
} from '@mui/icons-material'
import { useDropzone } from 'react-dropzone'
import { useMutation, useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

interface BatchFile {
  id: string
  file: File
  preview: string
  status: 'pending' | 'uploading' | 'uploaded' | 'analyzing' | 'completed' | 'error'
  progress: number
  drawingId?: number
  analysisResult?: {
    anomaly_score: number
    is_anomaly: boolean
    confidence: number
  }
  error?: string
}

interface BatchJob {
  id: string
  total_files: number
  completed_files: number
  failed_files: number
  status: 'pending' | 'running' | 'completed' | 'failed'
  created_at: string
  completed_at?: string
}

const BatchProcessingPage: React.FC = () => {
  const [batchFiles, setBatchFiles] = useState<BatchFile[]>([])
  const [currentJob, setCurrentJob] = useState<BatchJob | null>(null)
  const [showResults, setShowResults] = useState(false)
  const navigate = useNavigate()

  // Fetch active batch jobs
  useQuery<BatchJob[]>({
    queryKey: ['batch-jobs'],
    queryFn: async () => {
      const response = await axios.get('/api/analysis/batch/jobs')
      return response.data
    },
    refetchInterval: 2000, // Poll every 2 seconds when batch is running
  })

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: BatchFile[] = acceptedFiles.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      preview: URL.createObjectURL(file),
      status: 'pending',
      progress: 0,
    }))
    setBatchFiles((prev) => [...prev, ...newFiles])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/bmp': ['.bmp'],
    },
    maxSize: 10 * 1024 * 1024, // 10MB per file
  })

  const batchUploadMutation = useMutation({
    mutationFn: async (files: BatchFile[]) => {
      const formData = new FormData()
      files.forEach((batchFile) => {
        formData.append('files', batchFile.file)
      })

      const response = await axios.post('/api/analysis/batch/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            )
            // Update progress for all files
            setBatchFiles((prev) =>
              prev.map((file) => ({
                ...file,
                status: 'uploading',
                progress,
              }))
            )
          }
        },
      })
      return response.data
    },
    onSuccess: (data) => {
      setCurrentJob(data.job)
      setBatchFiles((prev) =>
        prev.map((file) => ({
          ...file,
          status: 'uploaded',
          progress: 100,
        }))
      )
    },
    onError: () => {
      setBatchFiles((prev) =>
        prev.map((file) => ({
          ...file,
          status: 'error',
          error: 'Upload failed',
        }))
      )
    },
  })

  const batchAnalyzeMutation = useMutation({
    mutationFn: async (jobId: string) => {
      const response = await axios.post(`/api/analysis/batch/${jobId}/analyze`)
      return response.data
    },
    onSuccess: () => {
      // Start polling for results
      setCurrentJob((prev) => prev ? { ...prev, status: 'running' } : null)
    },
  })

  const removeFile = (id: string) => {
    setBatchFiles((prev) => {
      const file = prev.find((f) => f.id === id)
      if (file) {
        URL.revokeObjectURL(file.preview)
      }
      return prev.filter((f) => f.id !== id)
    })
  }

  const clearAll = () => {
    batchFiles.forEach((file) => URL.revokeObjectURL(file.preview))
    setBatchFiles([])
    setCurrentJob(null)
  }

  const startBatchProcessing = () => {
    if (batchFiles.length > 0) {
      batchUploadMutation.mutate(batchFiles)
    }
  }

  const startAnalysis = () => {
    if (currentJob) {
      batchAnalyzeMutation.mutate(currentJob.id)
    }
  }

  const getStatusIcon = (status: BatchFile['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle color="success" />
      case 'error':
        return <Error color="error" />
      case 'analyzing':
      case 'uploading':
        return <Warning color="warning" />
      default:
        return <ImageIcon />
    }
  }

  const getStatusColor = (status: BatchFile['status']) => {
    switch (status) {
      case 'completed':
        return 'success'
      case 'error':
        return 'error'
      case 'analyzing':
      case 'uploading':
        return 'warning'
      default:
        return 'default'
    }
  }

  const completedFiles = batchFiles.filter((f) => f.status === 'completed')
  const anomalousFiles = completedFiles.filter((f) => f.analysisResult?.is_anomaly)

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Batch Processing
      </Typography>

      <Grid container spacing={3}>
        {/* Upload Section */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Upload Multiple Drawings
            </Typography>

            <Box
              {...getRootProps()}
              sx={{
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'grey.300',
                borderRadius: 2,
                p: 4,
                textAlign: 'center',
                cursor: 'pointer',
                bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                mb: 3,
              }}
            >
              <input {...getInputProps()} />
              <CloudUpload sx={{ fontSize: 48, color: 'grey.400', mb: 2 }} />
              {isDragActive ? (
                <Typography variant="body1">Drop the images here...</Typography>
              ) : (
                <Box>
                  <Typography variant="body1" gutterBottom>
                    Drag and drop multiple images here, or click to select
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Supported formats: PNG, JPEG, BMP (max 10MB each)
                  </Typography>
                </Box>
              )}
            </Box>

            {batchFiles.length > 0 && (
              <Box>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="h6">
                    Files ({batchFiles.length})
                  </Typography>
                  <Box>
                    <Button
                      size="small"
                      onClick={clearAll}
                      disabled={batchUploadMutation.isPending}
                    >
                      Clear All
                    </Button>
                    <Button
                      variant="contained"
                      onClick={startBatchProcessing}
                      disabled={batchUploadMutation.isPending || batchFiles.length === 0}
                      startIcon={<PlayArrow />}
                      sx={{ ml: 1 }}
                    >
                      {batchUploadMutation.isPending ? 'Uploading...' : 'Start Processing'}
                    </Button>
                  </Box>
                </Box>

                <List sx={{ maxHeight: 400, overflow: 'auto' }}>
                  {batchFiles.map((batchFile) => (
                    <ListItem key={batchFile.id}>
                      <ListItemAvatar>
                        <Avatar>
                          {getStatusIcon(batchFile.status)}
                        </Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={batchFile.file.name}
                        secondary={
                          <Box>
                            <Typography variant="body2" component="span">
                              {Math.round(batchFile.file.size / 1024)} KB
                            </Typography>
                            <br />
                            <Chip
                              size="small"
                              label={batchFile.status}
                              color={getStatusColor(batchFile.status) as any}
                              variant="outlined"
                            />
                            {batchFile.progress > 0 && batchFile.progress < 100 && (
                              <LinearProgress
                                variant="determinate"
                                value={batchFile.progress}
                                sx={{ mt: 1 }}
                              />
                            )}
                          </Box>
                        }
                      />
                      <IconButton
                        edge="end"
                        onClick={() => removeFile(batchFile.id)}
                        disabled={batchUploadMutation.isPending}
                      >
                        <Delete />
                      </IconButton>
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Status Section */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Processing Status
            </Typography>

            {currentJob && (
              <Box>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="body2">
                    Job: {currentJob.id.slice(0, 8)}...
                  </Typography>
                  <Chip
                    size="small"
                    label={currentJob.status}
                    color={currentJob.status === 'completed' ? 'success' : 'primary'}
                  />
                </Box>

                <Typography variant="body2" gutterBottom>
                  Progress: {currentJob.completed_files} / {currentJob.total_files}
                </Typography>

                <LinearProgress
                  variant="determinate"
                  value={(currentJob.completed_files / currentJob.total_files) * 100}
                  sx={{ mb: 2 }}
                />

                {currentJob.status === 'pending' && (
                  <Button
                    variant="contained"
                    fullWidth
                    onClick={startAnalysis}
                    disabled={batchAnalyzeMutation.isPending}
                    startIcon={<PlayArrow />}
                  >
                    Start Analysis
                  </Button>
                )}

                {currentJob.status === 'completed' && (
                  <Button
                    variant="outlined"
                    fullWidth
                    onClick={() => setShowResults(true)}
                    startIcon={<Download />}
                  >
                    View Results
                  </Button>
                )}
              </Box>
            )}

            {!currentJob && batchFiles.length === 0 && (
              <Alert severity="info">
                Upload multiple drawings to start batch processing.
              </Alert>
            )}
          </Paper>

          {/* Summary */}
          {completedFiles.length > 0 && (
            <Paper sx={{ p: 3, mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                Summary
              </Typography>
              <Typography variant="body2" gutterBottom>
                Total Processed: {completedFiles.length}
              </Typography>
              <Typography variant="body2" gutterBottom>
                Anomalies Detected: {anomalousFiles.length}
              </Typography>
              <Typography variant="body2">
                Anomaly Rate: {completedFiles.length > 0 ? ((anomalousFiles.length / completedFiles.length) * 100).toFixed(1) : 0}%
              </Typography>
            </Paper>
          )}
        </Grid>
      </Grid>

      {/* Results Dialog */}
      <Dialog
        open={showResults}
        onClose={() => setShowResults(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>Batch Processing Results</DialogTitle>
        <DialogContent>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Filename</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Anomaly Score</TableCell>
                  <TableCell>Result</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {completedFiles.map((file) => (
                  <TableRow key={file.id}>
                    <TableCell>{file.file.name}</TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        label={file.status}
                        color={getStatusColor(file.status) as any}
                      />
                    </TableCell>
                    <TableCell>
                      {file.analysisResult?.anomaly_score.toFixed(3) || 'N/A'}
                    </TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        label={file.analysisResult?.is_anomaly ? 'Anomaly' : 'Normal'}
                        color={file.analysisResult?.is_anomaly ? 'warning' : 'success'}
                      />
                    </TableCell>
                    <TableCell>
                      {file.drawingId && (
                        <Button
                          size="small"
                          onClick={() => navigate(`/analysis/${file.drawingId}`)}
                        >
                          View Details
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowResults(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default BatchProcessingPage