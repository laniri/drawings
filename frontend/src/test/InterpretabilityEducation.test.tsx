import React from 'react'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import { describe, it, expect } from 'vitest'

import InterpretabilityTutorial from '../components/interpretability/InterpretabilityTutorial'
import ContextualHelpSystem from '../components/interpretability/ContextualHelpSystem'
import ExampleGallery from '../components/interpretability/ExampleGallery'
import AdaptiveExplanationSystem from '../components/interpretability/AdaptiveExplanationSystem'
import InterpretabilityEducationHub from '../components/interpretability/InterpretabilityEducationHub'

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

  it('renders ExampleGallery without crashing', () => {
    render(
      <TestWrapper>
        <ExampleGallery
          ageGroup="3-4"
          userRole="educator"
        />
      </TestWrapper>
    )
    
    expect(screen.getByText('Interpretation Examples')).toBeInTheDocument()
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

  it('shows different content for different user roles', () => {
    const { rerender } = render(
      <TestWrapper>
        <InterpretabilityEducationHub
          userRole="researcher"
          ageGroup="5-6"
        />
      </TestWrapper>
    )
    
    expect(screen.getByText('Researcher View')).toBeInTheDocument()

    rerender(
      <TestWrapper>
        <InterpretabilityEducationHub
          userRole="parent"
          ageGroup="5-6"
        />
      </TestWrapper>
    )
    
    expect(screen.getByText('Parent View')).toBeInTheDocument()
  })
})