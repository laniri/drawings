import React, { useState, useRef, useCallback } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Tooltip,
  Zoom,
  Paper,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Chip,
  Grid,
} from '@mui/material'
import {
  ZoomIn,
  ZoomOut,
  Close,
  Visibility,
  VisibilityOff,
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'

interface InteractiveRegion {
  region_id: string
  bounding_box: [number, number, number, number] // [x1, y1, x2, y2]
  importance_score: number
  spatial_location: string
  hover_explanation: string
  click_explanation: string
}

interface AttentionPatch {
  patch_id: string
  coordinates: [number, number, number, number] // [x, y, width, height]
  attention_weight: number
  layer_index: number
  head_index: number
}

interface InteractiveInterpretabilityData {
  saliency_regions: InteractiveRegion[]
  attention_patches: AttentionPatch[]
  region_explanations: Record<string, string>
  confidence_scores: Record<string, number>
  interaction_metadata: {
    total_regions: number
    total_patches: number
    image_dimensions: [number, number]
    patch_size: number
    analysis_method: string
  }
}

interface InteractiveInterpretabilityViewerProps {
  analysisId: number
  drawingImageUrl: string
  saliencyMapUrl: string
  onRegionClick?: (regionId: string, explanation: string) => void
}

const InteractiveInterpretabilityViewer: React.FC<InteractiveInterpretabilityViewerProps> = ({
  analysisId,
  drawingImageUrl,
  saliencyMapUrl,
  onRegionClick,
}) => {

  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null)
  const [selectedRegion, setSelectedRegion] = useState<InteractiveRegion | null>(null)
  const [showAttentionPatches, setShowAttentionPatches] = useState(false)
  const [zoomLevel, setZoomLevel] = useState(1)
  const [imagePosition, setImagePosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const containerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)

  // Fetch interactive interpretability data
  const { data: interactiveData, isLoading, error } = useQuery<InteractiveInterpretabilityData>({
    queryKey: ['interactive-interpretability', analysisId],
    queryFn: async () => {
      const response = await axios.get(`/api/interpretability/${analysisId}/interactive`)
      return response.data
    },
    enabled: !!analysisId,
  })

  const handleRegionHover = useCallback((regionId: string | null) => {
    setHoveredRegion(regionId)
  }, [])

  const handleRegionClick = useCallback((region: InteractiveRegion) => {
    setSelectedRegion(region)
    onRegionClick?.(region.region_id, region.click_explanation)
  }, [onRegionClick])

  const handleZoomIn = useCallback(() => {
    setZoomLevel(prev => Math.min(prev * 1.5, 5))
  }, [])

  const handleZoomOut = useCallback(() => {
    setZoomLevel(prev => Math.max(prev / 1.5, 0.5))
  }, [])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (zoomLevel > 1) {
      setIsDragging(true)
      setDragStart({ x: e.clientX - imagePosition.x, y: e.clientY - imagePosition.y })
    }
  }, [zoomLevel, imagePosition])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging && zoomLevel > 1) {
      setImagePosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      })
    }
  }, [isDragging, dragStart, zoomLevel])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  const getRegionStyle = (region: InteractiveRegion) => {
    const [x1, y1, x2, y2] = region.bounding_box
    const isHovered = hoveredRegion === region.region_id
    const opacity = isHovered ? 0.8 : 0.4
    const borderWidth = isHovered ? 3 : 2
    
    // Color based on importance score
    const hue = (1 - region.importance_score) * 120 // Red (0) to Green (120)
    const color = `hsl(${hue}, 70%, 50%)`

    return {
      position: 'absolute' as const,
      left: `${x1}px`,
      top: `${y1}px`,
      width: `${x2 - x1}px`,
      height: `${y2 - y1}px`,
      border: `${borderWidth}px solid ${color}`,
      backgroundColor: `${color}${Math.round(opacity * 255).toString(16).padStart(2, '0')}`,
      cursor: 'pointer',
      transition: 'all 0.2s ease-in-out',
      borderRadius: '4px',
      zIndex: isHovered ? 10 : 5,
    }
  }

  const getPatchStyle = (patch: AttentionPatch) => {
    const [x, y, width, height] = patch.coordinates
    const opacity = patch.attention_weight * 0.6
    
    return {
      position: 'absolute' as const,
      left: `${x}px`,
      top: `${y}px`,
      width: `${width}px`,
      height: `${height}px`,
      backgroundColor: `rgba(255, 165, 0, ${opacity})`, // Orange with variable opacity
      border: '1px solid rgba(255, 165, 0, 0.8)',
      pointerEvents: 'none' as const,
      borderRadius: '2px',
    }
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent>
          <Typography>Loading interactive interpretability data...</Typography>
        </CardContent>
      </Card>
    )
  }

  if (error || !interactiveData) {
    return (
      <Card>
        <CardContent>
          <Typography color="error">
            Failed to load interactive interpretability data
          </Typography>
        </CardContent>
      </Card>
    )
  }

  return (
    <Box>
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Interactive Saliency Analysis</Typography>
            <Box display="flex" gap={1}>
              <IconButton onClick={() => setShowAttentionPatches(!showAttentionPatches)}>
                {showAttentionPatches ? <VisibilityOff /> : <Visibility />}
              </IconButton>
              <IconButton onClick={handleZoomOut} disabled={zoomLevel <= 0.5}>
                <ZoomOut />
              </IconButton>
              <Chip label={`${Math.round(zoomLevel * 100)}%`} size="small" />
              <IconButton onClick={handleZoomIn} disabled={zoomLevel >= 5}>
                <ZoomIn />
              </IconButton>
            </Box>
          </Box>

          <Box
            ref={containerRef}
            sx={{
              position: 'relative',
              overflow: 'hidden',
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 1,
              cursor: zoomLevel > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default',
              height: 400,
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {/* Base image */}
            <Box
              sx={{
                position: 'relative',
                transform: `scale(${zoomLevel}) translate(${imagePosition.x / zoomLevel}px, ${imagePosition.y / zoomLevel}px)`,
                transformOrigin: 'center center',
                transition: isDragging ? 'none' : 'transform 0.2s ease-out',
              }}
            >
              <img
                ref={imageRef}
                src={drawingImageUrl}
                alt="Drawing with interactive regions"
                style={{
                  maxWidth: '100%',
                  maxHeight: '100%',
                  display: 'block',
                }}
              />

              {/* Saliency overlay */}
              <img
                src={saliencyMapUrl}
                alt="Saliency map overlay"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  opacity: 0.4,
                  mixBlendMode: 'multiply',
                  pointerEvents: 'none',
                }}
              />

              {/* Interactive regions */}
              {interactiveData.saliency_regions.map((region) => (
                <Tooltip
                  key={region.region_id}
                  title={
                    <Box>
                      <Typography variant="body2" fontWeight="bold">
                        {region.spatial_location}
                      </Typography>
                      <Typography variant="body2">
                        Importance: {(region.importance_score * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        {region.hover_explanation}
                      </Typography>
                    </Box>
                  }
                  TransitionComponent={Zoom}
                  placement="top"
                  arrow
                >
                  <Box
                    style={getRegionStyle(region)}
                    onMouseEnter={() => handleRegionHover(region.region_id)}
                    onMouseLeave={() => handleRegionHover(null)}
                    onClick={(e) => {
                      e.stopPropagation()
                      handleRegionClick(region)
                    }}
                  />
                </Tooltip>
              ))}

              {/* Attention patches (when enabled) */}
              {showAttentionPatches &&
                interactiveData.attention_patches.map((patch) => (
                  <Tooltip
                    key={patch.patch_id}
                    title={
                      <Box>
                        <Typography variant="body2">
                          Attention: {(patch.attention_weight * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="body2">
                          Layer {patch.layer_index}, Head {patch.head_index}
                        </Typography>
                      </Box>
                    }
                    placement="top"
                  >
                    <Box style={getPatchStyle(patch)} />
                  </Tooltip>
                ))}
            </Box>
          </Box>

          {/* Region information panel */}
          {hoveredRegion && (
            <Paper sx={{ mt: 2, p: 2, backgroundColor: 'action.hover' }}>
              <Typography variant="body2" color="text.secondary">
                Hover over regions to see explanations • Click for detailed analysis
              </Typography>
            </Paper>
          )}

          {/* Analysis metadata */}
          <Box mt={2}>
            <Grid container spacing={2}>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Regions: {interactiveData.interaction_metadata.total_regions}
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Patches: {interactiveData.interaction_metadata.total_patches}
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Method: {interactiveData.interaction_metadata.analysis_method}
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Patch Size: {interactiveData.interaction_metadata.patch_size}px
                </Typography>
              </Grid>
            </Grid>
          </Box>
        </CardContent>
      </Card>

      {/* Detailed region explanation dialog */}
      <Dialog
        open={!!selectedRegion}
        onClose={() => setSelectedRegion(null)}
        maxWidth="md"
        fullWidth
      >
        {selectedRegion && (
          <>
            <DialogTitle>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="h6">
                  Region Analysis: {selectedRegion.spatial_location}
                </Typography>
                <IconButton onClick={() => setSelectedRegion(null)}>
                  <Close />
                </IconButton>
              </Box>
            </DialogTitle>
            <DialogContent>
              <Box mb={2}>
                <Chip
                  label={`Importance: ${(selectedRegion.importance_score * 100).toFixed(1)}%`}
                  color={selectedRegion.importance_score > 0.7 ? 'error' : selectedRegion.importance_score > 0.4 ? 'warning' : 'success'}
                  variant="outlined"
                />
                <Chip
                  label={`Confidence: ${((interactiveData.confidence_scores[selectedRegion.region_id] || 0) * 100).toFixed(1)}%`}
                  sx={{ ml: 1 }}
                  variant="outlined"
                />
              </Box>
              <Typography variant="body1" paragraph>
                {selectedRegion.click_explanation}
              </Typography>
              {interactiveData.region_explanations[selectedRegion.region_id] && (
                <Paper sx={{ p: 2, backgroundColor: 'action.hover' }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Technical Details:
                  </Typography>
                  <Typography variant="body2">
                    {interactiveData.region_explanations[selectedRegion.region_id]}
                  </Typography>
                </Paper>
              )}
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setSelectedRegion(null)}>Close</Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  )
}

export default InteractiveInterpretabilityViewer