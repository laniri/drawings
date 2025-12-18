import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import {
  ComparativeAnalysisPanel,
  ExportToolbar,
  AnnotationTools,
  HistoricalInterpretationTracker
} from '../components/interpretability'

// Mock fetch globally
Object.defineProperty(window, 'fetch', {
  value: vi.fn()
})

const createTestQueryClient = () => {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })
}

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = createTestQueryClient()
  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )
}

describe('Comparative Analysis and Export Features', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('ComparativeAnalysisPanel', () => {
    const mockCurrentAnalysis = {
      id: 1,
      drawing_id: 1,
      anomaly_score: 0.75,
      normalized_score: 75.0,
      is_anomaly: true,
      confidence: 0.85,
      age_group: '5-6',
      analysis_timestamp: '2024-01-15T10:30:00Z'
    }

    const mockCurrentDrawing = {
      id: 1,
      filename: 'test_drawing.png',
      age_years: 5.5,
      subject: 'house'
    }

    const mockComparisonData = {
      normal_examples: [
        {
          drawing_id: 2,
          filename: 'normal_example.png',
          age_years: 5.2,
          subject: 'house',
          anomaly_score: 0.15,
          normalized_score: 15.0,
          confidence: 0.9,
          analysis_timestamp: '2024-01-10T09:00:00Z'
        }
      ],
      anomalous_examples: [
        {
          drawing_id: 3,
          filename: 'anomalous_example.png',
          age_years: 5.8,
          subject: 'person',
          anomaly_score: 0.85,
          normalized_score: 85.0,
          confidence: 0.8,
          analysis_timestamp: '2024-01-12T14:20:00Z'
        }
      ],
      explanation_context: 'These examples show typical patterns for age group 5-6',
      age_group: '5-6',
      total_available: 150
    }

    it('renders comparative analysis panel with current analysis summary', async () => {
      const mockFetch = vi.fn()
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockComparisonData)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ analyses: [] })
        })

      window.fetch = mockFetch

      render(
        <TestWrapper>
          <ComparativeAnalysisPanel
            currentAnalysis={mockCurrentAnalysis}
            currentDrawing={mockCurrentDrawing}
          />
        </TestWrapper>
      )

      expect(screen.getByText('Comparative Analysis')).toBeInTheDocument()
      expect(screen.getByText('Current Analysis Summary')).toBeInTheDocument()
      expect(screen.getByText('test_drawing.png')).toBeInTheDocument()
      expect(screen.getByText('Age: 5.5 years')).toBeInTheDocument()
      expect(screen.getByText('Score: 75.0/100')).toBeInTheDocument()
    })

    it('loads and displays comparison examples', async () => {
      const mockFetch = vi.fn()
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockComparisonData)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ analyses: [] })
        })

      window.fetch = mockFetch

      render(
        <TestWrapper>
          <ComparativeAnalysisPanel
            currentAnalysis={mockCurrentAnalysis}
            currentDrawing={mockCurrentDrawing}
          />
        </TestWrapper>
      )

      await waitFor(() => {
        expect(screen.getByText('Normal Examples (Age 5-6)')).toBeInTheDocument()
        expect(screen.getByText('Anomalous Examples (Age 5-6)')).toBeInTheDocument()
      })

      expect(screen.getByText('normal_example.png')).toBeInTheDocument()
      expect(screen.getByText('anomalous_example.png')).toBeInTheDocument()
    })

    it('switches between different tab views', async () => {
      const mockFetch = vi.fn()
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockComparisonData)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ analyses: [] })
        })

      window.fetch = mockFetch

      render(
        <TestWrapper>
          <ComparativeAnalysisPanel
            currentAnalysis={mockCurrentAnalysis}
            currentDrawing={mockCurrentDrawing}
          />
        </TestWrapper>
      )

      // Click on Analysis History tab
      const historyTab = screen.getByText('Analysis History')
      fireEvent.click(historyTab)

      expect(screen.getByText('Track changes in analysis results over time')).toBeInTheDocument()

      // Click on Age Group Context tab
      const contextTab = screen.getByText('Age Group Context')
      fireEvent.click(contextTab)

      await waitFor(() => {
        expect(screen.getByText('Age Group Context')).toBeInTheDocument()
      })
    })
  })

  describe('ExportToolbar', () => {
    it('renders export toolbar with download button', () => {
      render(
        <TestWrapper>
          <ExportToolbar
            analysisId={1}
            drawingFilename="test_drawing.png"
          />
        </TestWrapper>
      )

      expect(screen.getByText('Export')).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /export/i })).toBeInTheDocument()
    })

    it('opens export menu when clicked', () => {
      render(
        <TestWrapper>
          <ExportToolbar
            analysisId={1}
            drawingFilename="test_drawing.png"
          />
        </TestWrapper>
      )

      const exportButton = screen.getByRole('button', { name: /export/i })
      fireEvent.click(exportButton)

      expect(screen.getByText('Quick Export')).toBeInTheDocument()
      expect(screen.getByText('PDF Report')).toBeInTheDocument()
      expect(screen.getByText('PNG Image')).toBeInTheDocument()
      expect(screen.getByText('CSV Data')).toBeInTheDocument()
    })

    it('handles export operation', async () => {
      const mockExportResult = {
        export_id: 'test-export-123',
        file_path: '/exports/test_export.pdf',
        file_url: '/static/exports/test_export.pdf',
        format: 'pdf',
        file_size: 1024000,
        created_at: '2024-01-15T10:30:00Z'
      }

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockExportResult)
      })

      window.fetch = mockFetch

      const onExportComplete = vi.fn()

      render(
        <TestWrapper>
          <ExportToolbar
            analysisId={1}
            drawingFilename="test_drawing.png"
            onExportComplete={onExportComplete}
          />
        </TestWrapper>
      )

      const exportButton = screen.getByRole('button', { name: /export/i })
      fireEvent.click(exportButton)

      const pdfOption = screen.getByText('PDF Report')
      fireEvent.click(pdfOption)

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          '/api/interpretability/1/export',
          expect.objectContaining({
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: expect.stringContaining('"format":"pdf"')
          })
        )
      })
    })
  })

  describe('AnnotationTools', () => {
    const mockRegions = [
      {
        region_id: 'region_1',
        bounding_box: [10, 20, 50, 60],
        spatial_location: 'top-left',
        importance_score: 0.8
      },
      {
        region_id: 'region_2',
        bounding_box: [100, 120, 150, 180],
        spatial_location: 'center',
        importance_score: 0.6
      }
    ]

    it('renders annotation tools with regions', () => {
      render(
        <TestWrapper>
          <AnnotationTools
            analysisId={1}
            regions={mockRegions}
          />
        </TestWrapper>
      )

      expect(screen.getByText('Annotations (0)')).toBeInTheDocument()
      expect(screen.getByText('Add Note')).toBeInTheDocument()
    })

    it('opens annotation dialog when add button is clicked', () => {
      render(
        <TestWrapper>
          <AnnotationTools
            analysisId={1}
            regions={mockRegions}
          />
        </TestWrapper>
      )

      const addButton = screen.getByText('Add Note')
      fireEvent.click(addButton)

      expect(screen.getByText('Add Annotation')).toBeInTheDocument()
      expect(screen.getByLabelText('Region')).toBeInTheDocument()
      expect(screen.getByLabelText('Type')).toBeInTheDocument()
      expect(screen.getByLabelText('Annotation Text')).toBeInTheDocument()
    })

    it('handles annotation creation', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          annotation_id: 'test-annotation-123',
          analysis_id: 1,
          region_id: 'region_1'
        })
      })

      window.fetch = mockFetch

      const onAnnotationAdd = vi.fn()

      render(
        <TestWrapper>
          <AnnotationTools
            analysisId={1}
            regions={mockRegions}
            onAnnotationAdd={onAnnotationAdd}
          />
        </TestWrapper>
      )

      const addButton = screen.getByText('Add Note')
      fireEvent.click(addButton)

      // Fill out the form
      const textField = screen.getByLabelText('Annotation Text')
      fireEvent.change(textField, { target: { value: 'Test annotation text' } })

      const saveButton = screen.getByText('Save')
      fireEvent.click(saveButton)

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          '/api/interpretability/1/annotate',
          expect.objectContaining({
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            }
          })
        )
      })
    })
  })

  describe('HistoricalInterpretationTracker', () => {
    const mockCurrentAnalysis = {
      id: 1,
      normalized_score: 75.0,
      is_anomaly: true,
      analysis_timestamp: '2024-01-15T10:30:00Z'
    }

    const mockHistoricalData = {
      analyses: [
        {
          id: 1,
          anomaly_score: 0.75,
          normalized_score: 75.0,
          is_anomaly: true,
          confidence: 0.85,
          age_group: '5-6',
          analysis_timestamp: '2024-01-15T10:30:00Z'
        },
        {
          id: 2,
          anomaly_score: 0.65,
          normalized_score: 65.0,
          is_anomaly: true,
          confidence: 0.8,
          age_group: '5-6',
          analysis_timestamp: '2024-01-10T09:00:00Z'
        }
      ]
    }

    it('renders historical interpretation tracker', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockHistoricalData)
      })

      window.fetch = mockFetch

      render(
        <TestWrapper>
          <HistoricalInterpretationTracker
            drawingId={1}
            currentAnalysis={mockCurrentAnalysis}
          />
        </TestWrapper>
      )

      expect(screen.getByText('Historical Analysis Tracking')).toBeInTheDocument()
      
      await waitFor(() => {
        expect(screen.getByText('Longitudinal Pattern Analysis')).toBeInTheDocument()
      })
    })

    it('switches between different view modes', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockHistoricalData)
      })

      window.fetch = mockFetch

      render(
        <TestWrapper>
          <HistoricalInterpretationTracker
            drawingId={1}
            currentAnalysis={mockCurrentAnalysis}
          />
        </TestWrapper>
      )

      await waitFor(() => {
        expect(screen.getByText('Timeline')).toBeInTheDocument()
      })

      // Switch to Chart view
      const chartOption = screen.getByDisplayValue('timeline')
      fireEvent.mouseDown(chartOption)
      
      const chartMenuItem = screen.getByText('Chart')
      fireEvent.click(chartMenuItem)

      expect(screen.getByText('Score Progression Over Time')).toBeInTheDocument()
    })
  })
})