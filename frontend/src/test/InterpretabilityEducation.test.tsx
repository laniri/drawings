import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import { describe, it, expect, vi, beforeEach } from 'vitest'

import InterpretabilityTutorial from '../components/interpretability/InterpretabilityTutorial'
import ContextualHelpSystem from '../components/interpretability/ContextualHelpSystem'
import ExampleGallery from '../components/interpretability/ExampleGallery'
import AdaptiveExplanationSystem from '../components/interpretability/AdaptiveExplanationSystem'
import InterpretabilityEducationHub from '../components/interpretability/InterpretabilityEducationHub'

// Mock fetch globally
const mockFetch = vi.fn()
Object.defineProperty(globalThis, 'fetch', {
  value: mockFetch,
  writable: true
})

const theme = createTheme()
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
})

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider theme={theme}>
      {children}
    </ThemeProvider>
  </QueryClientProvider>
)

describe('Interpretability Education Components', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Setup default mock responses for all API calls
    mockFetch.mockImplementation((url) => {
      // Mock the example patterns API
      if (url.includes('/api/interpretability/examples')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve([
            {
              pattern_id: 'test-1',
              pattern_name: 'Test Pattern',
              description: 'Test description',
              age_group: '3-4',
              example_type: 'normal',
              confidence_level: 'high',
              visual_features: ['test feature'],
              interpretation_notes: ['test note'],
              educational_context: 'test context',
              image_url: '/test-image.jpg',
              saliency_url: '/test-saliency.jpg',
              metadata: {
                drawing_count: 100,
                prevalence: 0.5,
                developmental_significance: 'test significance'
              }
            }
          ])
        })
      }
      
      // Default mock for other API calls
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve([])
      })
    })
  })
  it('renders InterpretabilityTutorial without crashing', () => {
    render(
      <TestWrapper>
        <InterpretabilityTutorial
          open={true}
          onClose={() => {}}
          userRole="educator"
        />
      </TestWrapper>
    )
    
    expect(screen.getByText('Interpretability Tutorial')).toBeInTheDocument()
  })

  it('renders ContextualHelpSystem without crashing', () => {
    render(
      <TestWrapper>
        <ContextualHelpSystem
          topic="saliency-maps"
          userRole="educator"
        />
      </TestWrapper>
    )
    
    // Should render a help button
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('renders ExampleGallery without crashing', async () => {
    render(
      <TestWrapper>
        <ExampleGallery
          ageGroup="3-4"
          userRole="educator"
        />
      </TestWrapper>
    )
    
    // Wait for the component to load and show the title
    await waitFor(() => {
      expect(screen.getByText('Interpretation Examples')).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('renders AdaptiveExplanationSystem without crashing', () => {
    const mockAnalysisData = {
      anomaly_score: 0.75,
      normalized_score: 75,
      is_anomaly: true,
      confidence: 0.85,
      threshold: 0.65,
      age_group: '5-6'
    }

    render(
      <TestWrapper>
        <AdaptiveExplanationSystem
          analysisData={mockAnalysisData}
        />
      </TestWrapper>
    )
    
    expect(screen.getByText('Explanation Settings')).toBeInTheDocument()
  })

  it('renders InterpretabilityEducationHub without crashing', () => {
    render(
      <TestWrapper>
        <InterpretabilityEducationHub
          userRole="educator"
          ageGroup="5-6"
        />
      </TestWrapper>
    )
    
    expect(screen.getByText('Interpretability Guide')).toBeInTheDocument()
    expect(screen.getByText('Educator View')).toBeInTheDocument()
  })

  it('shows different content for different user roles', async () => {
    const { rerender } = render(
      <TestWrapper>
        <InterpretabilityEducationHub
          userRole="researcher"
          ageGroup="5-6"
        />
      </TestWrapper>
    )
    
    await waitFor(() => {
      expect(screen.getByText('Researcher View')).toBeInTheDocument()
    })

    // Clear the previous render and render with parent role
    rerender(
      <TestWrapper>
        <InterpretabilityEducationHub
          userRole="parent"
          ageGroup="5-6"
        />
      </TestWrapper>
    )
    
    await waitFor(() => {
      // The component shows "Parent View" in the chip label
      expect(screen.getByText('Parent View')).toBeInTheDocument()
    }, { timeout: 5000 })
  })
})