import React, { useState } from 'react'
import {
  Box,
  Button,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  FormLabel,
  FormGroup,
  FormControlLabel,
  Checkbox,
  RadioGroup,
  Radio,
  TextField,
  Alert,
  CircularProgress,
  Chip,
  Typography,
  Paper,
  Grid,
  IconButton,
  Tooltip
} from '@mui/material'
import {
  FileDownload as DownloadIcon,
  PictureAsPdf as PdfIcon,
  Image as ImageIcon,
  TableChart as CsvIcon,
  Code as JsonIcon,
  Web as HtmlIcon,
  Settings as SettingsIcon,
  Share as ShareIcon,
  History as HistoryIcon,
  CheckCircle as CheckIcon
} from '@mui/icons-material'

interface ExportOptions {
  format: 'pdf' | 'png' | 'csv' | 'json' | 'html'
  include_annotations: boolean
  include_comparisons: boolean
  simplified_version: boolean
  export_options: {
    resolution?: string
    page_size?: string
    include_metadata?: boolean
    include_saliency?: boolean
    include_confidence?: boolean
  }
}

interface ExportResult {
  export_id: string
  file_path: string
  file_url: string
  format: string
  file_size: number
  created_at: string
  expires_at?: string
}

interface ExportHistory {
  export_id: string
  format: string
  file_size: number
  created_at: string
  status: 'completed' | 'failed' | 'expired'
}

interface ExportToolbarProps {
  analysisId: number
  drawingFilename: string
  onExportComplete?: (result: ExportResult) => void
  disabled?: boolean
}

const formatIcons = {
  pdf: PdfIcon,
  png: ImageIcon,
  csv: CsvIcon,
  json: JsonIcon,
  html: HtmlIcon
}

const formatLabels = {
  pdf: 'PDF Report',
  png: 'PNG Image',
  csv: 'CSV Data',
  json: 'JSON Data',
  html: 'HTML Report'
}

const formatDescriptions = {
  pdf: 'Comprehensive report with images and analysis',
  png: 'High-quality image with saliency overlays',
  csv: 'Structured data for spreadsheet analysis',
  json: 'Raw data for programmatic access',
  html: 'Interactive web report'
}

export default function ExportToolbar({
  analysisId,
  drawingFilename,
  onExportComplete,
  disabled = false
}: ExportToolbarProps) {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const [exportDialogOpen, setExportDialogOpen] = useState(false)
  const [historyDialogOpen, setHistoryDialogOpen] = useState(false)
  const [selectedFormat, setSelectedFormat] = useState<ExportOptions['format']>('pdf')
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    format: 'pdf',
    include_annotations: true,
    include_comparisons: true,
    simplified_version: false,
    export_options: {
      resolution: 'high',
      page_size: 'A4',
      include_metadata: true,
      include_saliency: true,
      include_confidence: true
    }
  })
  const [exporting, setExporting] = useState(false)
  const [exportError, setExportError] = useState<string | null>(null)
  const [exportSuccess, setExportSuccess] = useState<ExportResult | null>(null)
  const [exportHistory, setExportHistory] = useState<ExportHistory[]>([])

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget)
  }

  const handleMenuClose = () => {
    setAnchorEl(null)
  }

  const handleFormatSelect = (format: ExportOptions['format']) => {
    setSelectedFormat(format)
    setExportOptions(prev => ({ ...prev, format }))
    setExportDialogOpen(true)
    handleMenuClose()
  }

  const handleQuickExport = async (format: ExportOptions['format']) => {
    const quickOptions: ExportOptions = {
      format,
      include_annotations: true,
      include_comparisons: true,
      simplified_version: false,
      export_options: {}
    }
    
    await performExport(quickOptions)
    handleMenuClose()
  }

  const handleCustomExport = async () => {
    await performExport(exportOptions)
    setExportDialogOpen(false)
  }

  const performExport = async (options: ExportOptions) => {
    try {
      setExporting(true)
      setExportError(null)
      setExportSuccess(null)

      const response = await fetch(`/api/v1/interpretability/${analysisId}/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(options)
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Export failed')
      }

      const result: ExportResult = await response.json()
      setExportSuccess(result)
      
      if (onExportComplete) {
        onExportComplete(result)
      }

      // Add to history
      setExportHistory(prev => [{
        export_id: result.export_id,
        format: result.format,
        file_size: result.file_size,
        created_at: result.created_at,
        status: 'completed'
      }, ...prev])

      // Auto-download the file
      const link = document.createElement('a')
      link.href = result.file_url
      link.download = `${drawingFilename}_analysis.${result.format}`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)

    } catch (err) {
      console.error('Export failed:', err)
      setExportError(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setExporting(false)
    }
  }

  const handleOptionChange = (key: keyof ExportOptions, value: any) => {
    setExportOptions(prev => ({
      ...prev,
      [key]: value
    }))
  }

  const handleExportOptionChange = (key: string, value: any) => {
    setExportOptions(prev => ({
      ...prev,
      export_options: {
        ...prev.export_options,
        [key]: value
      }
    }))
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  return (
    <Box>
      {/* Main Export Button */}
      <Button
        variant="contained"
        startIcon={<DownloadIcon />}
        onClick={handleMenuOpen}
        disabled={disabled || exporting}
        sx={{ mr: 1 }}
      >
        {exporting ? 'Exporting...' : 'Export'}
      </Button>

      {/* Export History Button */}
      <Tooltip title="View export history">
        <IconButton
          onClick={() => setHistoryDialogOpen(true)}
          disabled={disabled}
          size="small"
        >
          <HistoryIcon />
        </IconButton>
      </Tooltip>

      {/* Export Format Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
        PaperProps={{
          sx: { minWidth: 280 }
        }}
      >
        <MenuItem disabled>
          <Typography variant="subtitle2" color="text.secondary">
            Quick Export
          </Typography>
        </MenuItem>
        
        {Object.entries(formatLabels).map(([format, label]) => {
          const IconComponent = formatIcons[format as keyof typeof formatIcons]
          return (
            <MenuItem
              key={format}
              onClick={() => handleQuickExport(format as ExportOptions['format'])}
            >
              <ListItemIcon>
                <IconComponent fontSize="small" />
              </ListItemIcon>
              <ListItemText
                primary={label}
                secondary={formatDescriptions[format as keyof typeof formatDescriptions]}
              />
            </MenuItem>
          )
        })}

        <Divider />

        <MenuItem onClick={() => setExportDialogOpen(true)}>
          <ListItemIcon>
            <SettingsIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="Custom Export"
            secondary="Configure export options"
          />
        </MenuItem>
      </Menu>

      {/* Custom Export Dialog */}
      <Dialog
        open={exportDialogOpen}
        onClose={() => setExportDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Export Configuration
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            {/* Format Selection */}
            <FormControl component="fieldset" sx={{ mb: 3 }}>
              <FormLabel component="legend">Export Format</FormLabel>
              <RadioGroup
                value={selectedFormat}
                onChange={(e) => {
                  const format = e.target.value as ExportOptions['format']
                  setSelectedFormat(format)
                  handleOptionChange('format', format)
                }}
                row
              >
                {Object.entries(formatLabels).map(([format, label]) => {
                  const IconComponent = formatIcons[format as keyof typeof formatIcons]
                  return (
                    <FormControlLabel
                      key={format}
                      value={format}
                      control={<Radio />}
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <IconComponent sx={{ mr: 1, fontSize: 20 }} />
                          {label}
                        </Box>
                      }
                    />
                  )
                })}
              </RadioGroup>
            </FormControl>

            {/* Content Options */}
            <FormControl component="fieldset" sx={{ mb: 3 }}>
              <FormLabel component="legend">Content Options</FormLabel>
              <FormGroup>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={exportOptions.include_annotations}
                      onChange={(e) => handleOptionChange('include_annotations', e.target.checked)}
                    />
                  }
                  label="Include user annotations and notes"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={exportOptions.include_comparisons}
                      onChange={(e) => handleOptionChange('include_comparisons', e.target.checked)}
                    />
                  }
                  label="Include comparison examples"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={exportOptions.simplified_version}
                      onChange={(e) => handleOptionChange('simplified_version', e.target.checked)}
                    />
                  }
                  label="Use simplified explanations"
                />
              </FormGroup>
            </FormControl>

            {/* Format-specific Options */}
            {(selectedFormat === 'pdf' || selectedFormat === 'html') && (
              <FormControl component="fieldset" sx={{ mb: 3 }}>
                <FormLabel component="legend">Report Options</FormLabel>
                <FormGroup>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={exportOptions.export_options.include_metadata}
                        onChange={(e) => handleExportOptionChange('include_metadata', e.target.checked)}
                      />
                    }
                    label="Include technical metadata"
                  />
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={exportOptions.export_options.include_saliency}
                        onChange={(e) => handleExportOptionChange('include_saliency', e.target.checked)}
                      />
                    }
                    label="Include saliency maps"
                  />
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={exportOptions.export_options.include_confidence}
                        onChange={(e) => handleExportOptionChange('include_confidence', e.target.checked)}
                      />
                    }
                    label="Include confidence metrics"
                  />
                </FormGroup>
              </FormControl>
            )}

            {selectedFormat === 'png' && (
              <FormControl sx={{ mb: 3, minWidth: 200 }}>
                <FormLabel>Image Resolution</FormLabel>
                <RadioGroup
                  value={exportOptions.export_options.resolution || 'high'}
                  onChange={(e) => handleExportOptionChange('resolution', e.target.value)}
                >
                  <FormControlLabel value="standard" control={<Radio />} label="Standard (72 DPI)" />
                  <FormControlLabel value="high" control={<Radio />} label="High (150 DPI)" />
                  <FormControlLabel value="print" control={<Radio />} label="Print Quality (300 DPI)" />
                </RadioGroup>
              </FormControl>
            )}

            {selectedFormat === 'pdf' && (
              <FormControl sx={{ mb: 3, minWidth: 200 }}>
                <FormLabel>Page Size</FormLabel>
                <RadioGroup
                  value={exportOptions.export_options.page_size || 'A4'}
                  onChange={(e) => handleExportOptionChange('page_size', e.target.value)}
                >
                  <FormControlLabel value="A4" control={<Radio />} label="A4" />
                  <FormControlLabel value="Letter" control={<Radio />} label="Letter" />
                  <FormControlLabel value="Legal" control={<Radio />} label="Legal" />
                </RadioGroup>
              </FormControl>
            )}

            {/* Error Display */}
            {exportError && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {exportError}
              </Alert>
            )}

            {/* Success Display */}
            {exportSuccess && (
              <Alert severity="success" sx={{ mb: 2 }} icon={<CheckIcon />}>
                <Typography variant="body2">
                  Export completed successfully! File size: {formatFileSize(exportSuccess.file_size)}
                </Typography>
              </Alert>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleCustomExport}
            disabled={exporting}
            startIcon={exporting ? <CircularProgress size={16} /> : <DownloadIcon />}
          >
            {exporting ? 'Exporting...' : 'Export'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Export History Dialog */}
      <Dialog
        open={historyDialogOpen}
        onClose={() => setHistoryDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Export History
        </DialogTitle>
        <DialogContent>
          {exportHistory.length > 0 ? (
            <Box sx={{ mt: 2 }}>
              {exportHistory.map((item) => {
                const IconComponent = formatIcons[item.format as keyof typeof formatIcons]
                return (
                  <Paper key={item.export_id} sx={{ p: 2, mb: 2 }}>
                    <Grid container spacing={2} alignItems="center">
                      <Grid item>
                        <IconComponent color="primary" />
                      </Grid>
                      <Grid item xs>
                        <Typography variant="subtitle2">
                          {formatLabels[item.format as keyof typeof formatLabels]}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {formatDate(item.created_at)} • {formatFileSize(item.file_size)}
                        </Typography>
                      </Grid>
                      <Grid item>
                        <Chip
                          label={item.status}
                          color={item.status === 'completed' ? 'success' : 'error'}
                          size="small"
                        />
                      </Grid>
                    </Grid>
                  </Paper>
                )
              })}
            </Box>
          ) : (
            <Alert severity="info">
              No export history available
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHistoryDialogOpen(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}