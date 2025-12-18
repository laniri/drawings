import { useState, useEffect } from 'react'
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Chip,
  IconButton,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  Alert,
  Avatar,
  Fab,
  Collapse,
  Card,
  CardContent,
  CardActions
} from '@mui/material'
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Note as NoteIcon,
  Help as QuestionIcon,
  Warning as ConcernIcon,
  Visibility as ObservationIcon,
  Science as HypothesisIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Save as SaveIcon,
  Comment as CommentIcon
} from '@mui/icons-material'

interface Annotation {
  id: string
  region_id: string
  text: string
  type: 'note' | 'question' | 'concern' | 'observation' | 'hypothesis'
  user_id?: string
  timestamp: string
  position?: { x: number; y: number }
}

interface AnnotationToolsProps {
  analysisId: number
  regions: Array<{
    region_id: string
    bounding_box: number[]
    spatial_location: string
    importance_score: number
  }>
  onAnnotationAdd?: (annotation: Annotation) => void
  onAnnotationUpdate?: (annotation: Annotation) => void
  onAnnotationDelete?: (annotationId: string) => void
  readOnly?: boolean
}

const annotationTypes = {
  note: { icon: NoteIcon, label: 'Note', color: '#2196f3', description: 'General observation or comment' },
  question: { icon: QuestionIcon, label: 'Question', color: '#ff9800', description: 'Question or uncertainty' },
  concern: { icon: ConcernIcon, label: 'Concern', color: '#f44336', description: 'Area of concern or flag' },
  observation: { icon: ObservationIcon, label: 'Observation', color: '#4caf50', description: 'Clinical observation' },
  hypothesis: { icon: HypothesisIcon, label: 'Hypothesis', color: '#9c27b0', description: 'Hypothesis or theory' }
}

export default function AnnotationTools({
  analysisId,
  regions,
  onAnnotationAdd,
  onAnnotationUpdate,
  onAnnotationDelete,
  readOnly = false
}: AnnotationToolsProps) {
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [annotationDialogOpen, setAnnotationDialogOpen] = useState(false)
  const [editingAnnotation, setEditingAnnotation] = useState<Annotation | null>(null)
  const [newAnnotation, setNewAnnotation] = useState({
    text: '',
    type: 'note' as keyof typeof annotationTypes,
    region_id: ''
  })
  const [expanded, setExpanded] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load existing annotations
  useEffect(() => {
    loadAnnotations()
  }, [analysisId])

  const loadAnnotations = async () => {
    try {
      setLoading(true)
      // For now, we'll use mock data since the backend stores annotations in a simplified way
      // In a full implementation, you'd have a dedicated annotations endpoint
      const mockAnnotations: Annotation[] = [
        {
          id: '1',
          region_id: 'region_1',
          text: 'This area shows unusual line quality that may indicate motor control issues.',
          type: 'observation',
          timestamp: new Date().toISOString(),
          user_id: 'clinician_1'
        },
        {
          id: '2',
          region_id: 'region_2',
          text: 'Need to compare with previous drawings to assess progression.',
          type: 'note',
          timestamp: new Date(Date.now() - 86400000).toISOString(),
          user_id: 'researcher_1'
        }
      ]
      setAnnotations(mockAnnotations)
    } catch (err) {
      console.error('Error loading annotations:', err)
      setError('Failed to load annotations')
    } finally {
      setLoading(false)
    }
  }

  const handleAddAnnotation = (regionId?: string) => {
    setNewAnnotation({
      text: '',
      type: 'note',
      region_id: regionId || ''
    })
    setEditingAnnotation(null)
    setAnnotationDialogOpen(true)
  }

  const handleEditAnnotation = (annotation: Annotation) => {
    setEditingAnnotation(annotation)
    setNewAnnotation({
      text: annotation.text,
      type: annotation.type,
      region_id: annotation.region_id
    })
    setAnnotationDialogOpen(true)
  }

  const handleSaveAnnotation = async () => {
    try {
      setLoading(true)
      setError(null)

      const annotationData = {
        region_id: newAnnotation.region_id,
        annotation_text: newAnnotation.text,
        annotation_type: newAnnotation.type
      }

      const response = await fetch(`/api/interpretability/${analysisId}/annotate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(annotationData)
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to save annotation')
      }

      const result = await response.json()

      // Create the annotation object
      const annotation: Annotation = {
        id: result.annotation_id || Date.now().toString(),
        region_id: newAnnotation.region_id,
        text: newAnnotation.text,
        type: newAnnotation.type,
        timestamp: new Date().toISOString(),
        user_id: 'current_user'
      }

      if (editingAnnotation) {
        // Update existing annotation
        setAnnotations(prev => prev.map(a => 
          a.id === editingAnnotation.id ? annotation : a
        ))
        if (onAnnotationUpdate) {
          onAnnotationUpdate(annotation)
        }
      } else {
        // Add new annotation
        setAnnotations(prev => [...prev, annotation])
        if (onAnnotationAdd) {
          onAnnotationAdd(annotation)
        }
      }

      setAnnotationDialogOpen(false)
      setNewAnnotation({ text: '', type: 'note', region_id: '' })
      setEditingAnnotation(null)

    } catch (err) {
      console.error('Error saving annotation:', err)
      setError(err instanceof Error ? err.message : 'Failed to save annotation')
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteAnnotation = async (annotationId: string) => {
    try {
      // In a full implementation, you'd call a delete endpoint
      setAnnotations(prev => prev.filter(a => a.id !== annotationId))
      
      if (onAnnotationDelete) {
        onAnnotationDelete(annotationId)
      }
    } catch (err) {
      console.error('Error deleting annotation:', err)
      setError('Failed to delete annotation')
    }
  }

  const getRegionName = (regionId: string) => {
    const region = regions.find(r => r.region_id === regionId)
    return region ? `${region.spatial_location} (${(region.importance_score * 100).toFixed(0)}% importance)` : regionId
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getAnnotationsByRegion = (regionId: string) => {
    return annotations.filter(a => a.region_id === regionId)
  }

  const renderAnnotationItem = (annotation: Annotation) => {
    const typeConfig = annotationTypes[annotation.type]
    const IconComponent = typeConfig.icon

    return (
      <Card key={annotation.id} sx={{ mb: 1, border: `1px solid ${typeConfig.color}` }}>
        <CardContent sx={{ pb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 1 }}>
            <Avatar
              sx={{
                width: 24,
                height: 24,
                backgroundColor: typeConfig.color,
                mr: 1,
                mt: 0.5
              }}
            >
              <IconComponent sx={{ fontSize: 14 }} />
            </Avatar>
            <Box sx={{ flex: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                <Chip
                  label={typeConfig.label}
                  size="small"
                  sx={{
                    backgroundColor: typeConfig.color,
                    color: 'white',
                    fontSize: '0.7rem'
                  }}
                />
                <Typography variant="caption" color="text.secondary">
                  {formatTimestamp(annotation.timestamp)}
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ mb: 1 }}>
                {annotation.text}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Region: {getRegionName(annotation.region_id)}
              </Typography>
            </Box>
          </Box>
        </CardContent>
        {!readOnly && (
          <CardActions sx={{ pt: 0, pb: 1, px: 2 }}>
            <Button
              size="small"
              startIcon={<EditIcon />}
              onClick={() => handleEditAnnotation(annotation)}
            >
              Edit
            </Button>
            <Button
              size="small"
              color="error"
              startIcon={<DeleteIcon />}
              onClick={() => handleDeleteAnnotation(annotation.id)}
            >
              Delete
            </Button>
          </CardActions>
        )}
      </Card>
    )
  }

  return (
    <Box>
      {/* Header */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <CommentIcon sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6">
              Annotations ({annotations.length})
            </Typography>
            <IconButton
              size="small"
              onClick={() => setExpanded(!expanded)}
              sx={{ ml: 1 }}
            >
              {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Box>
          
          {!readOnly && (
            <Button
              variant="contained"
              size="small"
              startIcon={<AddIcon />}
              onClick={() => handleAddAnnotation()}
            >
              Add Note
            </Button>
          )}
        </Box>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </Paper>

      {/* Annotations List */}
      <Collapse in={expanded}>
        <Box>
          {annotations.length > 0 ? (
            <Box>
              {/* Group by region */}
              {regions.map(region => {
                const regionAnnotations = getAnnotationsByRegion(region.region_id)
                if (regionAnnotations.length === 0) return null

                return (
                  <Paper key={region.region_id} sx={{ p: 2, mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      {getRegionName(region.region_id)}
                    </Typography>
                    {regionAnnotations.map(renderAnnotationItem)}
                    {!readOnly && (
                      <Button
                        size="small"
                        startIcon={<AddIcon />}
                        onClick={() => handleAddAnnotation(region.region_id)}
                        sx={{ mt: 1 }}
                      >
                        Add Note to This Region
                      </Button>
                    )}
                  </Paper>
                )
              })}

              {/* Unassigned annotations */}
              {annotations.filter(a => !regions.some(r => r.region_id === a.region_id)).map(renderAnnotationItem)}
            </Box>
          ) : (
            <Paper sx={{ p: 3, textAlign: 'center' }}>
              <CommentIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="body1" color="text.secondary" gutterBottom>
                No annotations yet
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Add notes, questions, or observations to help with interpretation
              </Typography>
              {!readOnly && (
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={() => handleAddAnnotation()}
                >
                  Add First Annotation
                </Button>
              )}
            </Paper>
          )}
        </Box>
      </Collapse>

      {/* Add/Edit Annotation Dialog */}
      <Dialog
        open={annotationDialogOpen}
        onClose={() => setAnnotationDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {editingAnnotation ? 'Edit Annotation' : 'Add Annotation'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            {/* Region Selection */}
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Region</InputLabel>
              <Select
                value={newAnnotation.region_id}
                onChange={(e) => setNewAnnotation(prev => ({ ...prev, region_id: e.target.value }))}
                label="Region"
              >
                {regions.map(region => (
                  <MenuItem key={region.region_id} value={region.region_id}>
                    {getRegionName(region.region_id)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Annotation Type */}
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Type</InputLabel>
              <Select
                value={newAnnotation.type}
                onChange={(e) => setNewAnnotation(prev => ({ ...prev, type: e.target.value as keyof typeof annotationTypes }))}
                label="Type"
              >
                {Object.entries(annotationTypes).map(([key, config]) => {
                  const IconComponent = config.icon
                  return (
                    <MenuItem key={key} value={key}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <IconComponent sx={{ mr: 1, color: config.color }} />
                        <Box>
                          <Typography variant="body2">{config.label}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {config.description}
                          </Typography>
                        </Box>
                      </Box>
                    </MenuItem>
                  )
                })}
              </Select>
            </FormControl>

            {/* Annotation Text */}
            <TextField
              fullWidth
              multiline
              rows={4}
              label="Annotation Text"
              value={newAnnotation.text}
              onChange={(e) => setNewAnnotation(prev => ({ ...prev, text: e.target.value }))}
              placeholder="Enter your note, observation, or question..."
              helperText="Provide detailed context that will help with interpretation and future reference"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAnnotationDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSaveAnnotation}
            disabled={!newAnnotation.text.trim() || !newAnnotation.region_id || loading}
            startIcon={loading ? undefined : <SaveIcon />}
          >
            {loading ? 'Saving...' : (editingAnnotation ? 'Update' : 'Save')}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Quick Add FAB (when collapsed) */}
      {!expanded && !readOnly && (
        <Fab
          color="primary"
          size="small"
          sx={{ position: 'fixed', bottom: 16, right: 16 }}
          onClick={() => handleAddAnnotation()}
        >
          <AddIcon />
        </Fab>
      )}
    </Box>
  )
}