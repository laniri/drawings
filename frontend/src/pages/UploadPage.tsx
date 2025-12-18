import React, { useState, useCallback } from 'react'
import {
  Typography,
  Paper,
  Box,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  LinearProgress,
  Grid,
  Card,
  CardContent,
  CardMedia,
} from '@mui/material'
import { useDropzone } from 'react-dropzone'
import { useForm, Controller } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { CloudUpload, Image as ImageIcon } from '@mui/icons-material'
import { useMutation } from '@tanstack/react-query'
import axios from 'axios'
import { useAppStore } from '../store/useAppStore'
import { useNavigate } from 'react-router-dom'
import SubjectCategorySelect from '../components/SubjectCategorySelect'

// Validation schema
const uploadSchema = z.object({
  age_years: z.number().min(2).max(18),
  subject: z.string().optional(),
  expert_label: z.enum(['', 'normal', 'concern', 'severe']).optional(),
  drawing_tool: z.string().optional(),
  prompt: z.string().optional(),
})

type UploadFormData = z.infer<typeof uploadSchema>

interface UploadedFile extends File {
  preview: string
}

const UploadPage: React.FC = () => {
  const navigate = useNavigate()
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [analysisStatus, setAnalysisStatus] = useState<string | null>(null)
  const [analysisId, setAnalysisId] = useState<number | null>(null)
  const { uploadProgress, setUploadProgress } = useAppStore()

  const {
    control,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<UploadFormData>({
    resolver: zodResolver(uploadSchema),
    defaultValues: {
      age_years: 5,
      subject: '',
      expert_label: '',
      drawing_tool: '',
      prompt: '',
    },
  })

  const uploadMutation = useMutation({
    mutationFn: async (data: { file: File; metadata: UploadFormData }) => {
      const formData = new FormData()
      formData.append('file', data.file)
      formData.append('age_years', data.metadata.age_years.toString())
      
      if (data.metadata.subject) {
        formData.append('subject', data.metadata.subject)
      }
      if (data.metadata.expert_label) {
        formData.append('expert_label', data.metadata.expert_label)
      }
      if (data.metadata.drawing_tool) {
        formData.append('drawing_tool', data.metadata.drawing_tool)
      }
      if (data.metadata.prompt) {
        formData.append('prompt', data.metadata.prompt)
      }

      const response = await axios.post('/api/drawings/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            )
            setUploadProgress(progress)
          }
        },
      })
      return response.data
    },
    onSuccess: async (uploadResult) => {
      setUploadError(null)
      setUploadedFile(null)
      reset()
      setUploadProgress(0)
      
      // Automatically trigger analysis for the uploaded drawing
      setAnalysisStatus('Analyzing drawing...')
      setAnalysisId(null)
      try {
        const analysisResponse = await axios.post(`/api/analysis/analyze/${uploadResult.id}`)
        const newAnalysisId = analysisResponse.data.analysis.id
        setAnalysisId(newAnalysisId)
        setAnalysisStatus('Analysis complete! Click to view results.')
        console.log(`Analysis completed for drawing ${uploadResult.id}, analysis ID: ${newAnalysisId}`)
        
        // Clear analysis status after a longer time to let user see the link
        setTimeout(() => {
          setAnalysisStatus(null)
          setAnalysisId(null)
        }, 15000)
      } catch (error) {
        console.warn('Failed to trigger automatic analysis:', error)
        setAnalysisStatus('Analysis failed, but upload was successful')
        setTimeout(() => setAnalysisStatus(null), 5000)
      }
    },
    onError: (error: any) => {
      setUploadError(
        error.response?.data?.detail || 'Upload failed. Please try again.'
      )
      setUploadProgress(0)
    },
  })

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      const fileWithPreview = Object.assign(file, {
        preview: URL.createObjectURL(file),
      }) as UploadedFile
      setUploadedFile(fileWithPreview)
      setUploadError(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/bmp': ['.bmp'],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
    onDropRejected: (fileRejections) => {
      const rejection = fileRejections[0]
      if (rejection.errors[0]?.code === 'file-too-large') {
        setUploadError('File is too large. Maximum size is 10MB.')
      } else if (rejection.errors[0]?.code === 'file-invalid-type') {
        setUploadError('Invalid file type. Please upload PNG, JPEG, or BMP images.')
      } else {
        setUploadError('File upload failed. Please try again.')
      }
    },
  })

  const onSubmit = (data: UploadFormData) => {
    if (!uploadedFile) {
      setUploadError('Please select a file to upload.')
      return
    }
    uploadMutation.mutate({ file: uploadedFile, metadata: data })
  }

  const removeFile = () => {
    if (uploadedFile) {
      URL.revokeObjectURL(uploadedFile.preview)
      setUploadedFile(null)
    }
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Upload Drawing
      </Typography>

      <Grid container spacing={3}>
        {/* File Upload Section */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Select Drawing
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
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  borderColor: 'primary.main',
                  bgcolor: 'action.hover',
                },
              }}
            >
              <input {...getInputProps()} />
              <CloudUpload sx={{ fontSize: 48, color: 'grey.400', mb: 2 }} />
              {isDragActive ? (
                <Typography variant="body1">Drop the image here...</Typography>
              ) : (
                <Box>
                  <Typography variant="body1" gutterBottom>
                    Drag and drop an image here, or click to select
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Supported formats: PNG, JPEG, BMP (max 10MB)
                  </Typography>
                </Box>
              )}
            </Box>

            {uploadedFile && (
              <Card sx={{ mt: 2 }}>
                <CardMedia
                  component="img"
                  height="200"
                  image={uploadedFile.preview}
                  alt="Preview"
                  sx={{ objectFit: 'contain' }}
                />
                <CardContent>
                  <Typography variant="body2">
                    {uploadedFile.name} ({Math.round(uploadedFile.size / 1024)} KB)
                  </Typography>
                  <Button
                    size="small"
                    color="error"
                    onClick={removeFile}
                    sx={{ mt: 1 }}
                  >
                    Remove
                  </Button>
                </CardContent>
              </Card>
            )}
          </Paper>
        </Grid>

        {/* Metadata Form Section */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Drawing Information
            </Typography>

            <Box component="form" onSubmit={handleSubmit(onSubmit)} sx={{ mt: 2 }}>
              <Controller
                name="age_years"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Child's Age (years)"
                    type="number"
                    fullWidth
                    margin="normal"
                    required
                    inputProps={{ min: 2, max: 18, step: 0.5 }}
                    error={!!errors.age_years}
                    helperText={errors.age_years?.message}
                    onChange={(e) => field.onChange(parseFloat(e.target.value))}
                  />
                )}
              />

              <SubjectCategorySelect
                control={control}
                name="subject"
                label="Drawing Subject (optional)"
                showSearch={true}
              />

              <Controller
                name="expert_label"
                control={control}
                render={({ field }) => (
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Expert Label (optional)</InputLabel>
                    <Select {...field} label="Expert Label (optional)">
                      <MenuItem value="">None</MenuItem>
                      <MenuItem value="normal">Normal</MenuItem>
                      <MenuItem value="concern">Concern</MenuItem>
                      <MenuItem value="severe">Severe</MenuItem>
                    </Select>
                  </FormControl>
                )}
              />

              <Controller
                name="drawing_tool"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Drawing Tool (optional)"
                    fullWidth
                    margin="normal"
                    placeholder="e.g., crayon, pencil, marker"
                  />
                )}
              />

              <Controller
                name="prompt"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Drawing Prompt (optional)"
                    fullWidth
                    margin="normal"
                    multiline
                    rows={3}
                    placeholder="Instructions given to the child"
                  />
                )}
              />

              {uploadError && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {uploadError}
                </Alert>
              )}

              {uploadMutation.isSuccess && !analysisStatus && (
                <Alert severity="success" sx={{ mt: 2 }}>
                  Drawing uploaded successfully!
                </Alert>
              )}

              {analysisStatus && (
                <Alert 
                  severity={analysisStatus.includes('complete') ? 'success' : analysisStatus.includes('failed') ? 'warning' : 'info'} 
                  sx={{ mt: 2 }}
                  action={
                    analysisId && analysisStatus.includes('complete') ? (
                      <Button 
                        color="inherit" 
                        size="small" 
                        onClick={() => navigate(`/analysis/${analysisId}`)}
                      >
                        View Results
                      </Button>
                    ) : null
                  }
                >
                  {analysisStatus}
                </Alert>
              )}

              {uploadProgress > 0 && uploadProgress < 100 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Uploading... {uploadProgress}%
                  </Typography>
                  <LinearProgress variant="determinate" value={uploadProgress} />
                </Box>
              )}

              <Button
                type="submit"
                variant="contained"
                fullWidth
                size="large"
                disabled={uploadMutation.isPending || !uploadedFile}
                sx={{ mt: 3 }}
                startIcon={<ImageIcon />}
              >
                {uploadMutation.isPending ? 'Uploading...' : 'Upload Drawing'}
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}

export default UploadPage