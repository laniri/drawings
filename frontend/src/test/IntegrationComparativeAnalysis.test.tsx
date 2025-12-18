import React from 'react'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { vi, describe, it, expect } from 'vitest'
import {
  ComparativeAnalysisPanel,
  ExportToolbar,
  AnnotationTools,
  HistoricalInterpretationTracker
} from '../components/interpretability'

// Mock fetch globally
const mockFetch = vi.fn()
Object.defineProperty(window, 'fetch', {
  value: mockFetch,
  writable: true
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

describe('Comparative Analysis Integration', () => {
  it('renders all new components without crashing', () => {
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

    const mockRegions = [
      {
        region_id: 'region_1',
        bounding_box: [10, 20, 50, 60],
        spatial_location: 'top-left',
        importance_score: 0.8
      }
    ]

    // Mock successful API responses
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        normal_examples: [],
        anomalous_examples: [],
        explanation_context: 'Test context',
        age_group: '5-6',
        total_available: 0
      })
    })

    // Test ComparativeAnalysisPanel
    const { unmount: unmount1 } = render(
      <TestWrapper>
        <ComparativeAnalysisPanel
          currentAnalysis={mockCurrentAnalysis}
          currentDrawing={mockCurrentDrawing}
        />
      </TestWrapper>
    )
    
    expect(screen.getByText('Comparative Analysis')).toBeInTheDocument()
    unmount1()

    // Test ExportToolbar
    const { unmount: unmount2 } = render(
      <TestWrapper>
        <ExportToolbar
          analysisId={1}
          drawingFilename="test_drawing.png"
        />
      </TestWrapper>
    )
    
    expect(screen.getByText('Export')).toBeInTheDocument()
    unmount2()

    // Test AnnotationTools
    const { unmount: unmount3 } = render(
      <TestWrapper>
        <AnnotationTools
          analysisId={1}
          regions={mockRegions}
        />
      </TestWrapper>
    )
    
    expect(screen.getByText(/Annotations/)).toBeInTheDocument()
    unmount3()

    // Test HistoricalInterpretationTracker
    render(
      <TestWrapper>
        <HistoricalInterpretationTracker
          drawingId={1}
          currentAnalysis={mockCurrentAnalysis}
        />
      </TestWrapper>
    )
    
    expect(screen.getByText('Historical Analysis Tracking')).toBeInTheDocument()
  })
})