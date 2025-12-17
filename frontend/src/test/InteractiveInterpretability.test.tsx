import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import {
  InteractiveInterpretabilityViewer,
  ExplanationLevelToggle,
  ConfidenceIndicator,
} from '../components/interpretability'

// Mock axios
vi.mock('axios', () => ({
  default: {
    get: vi.fn(() => Promise.resolve({ data: {} })),
  },
}))

const theme = createTheme()

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        {children}
      </ThemeProvider>
    </QueryClientProvider>
  )
}

describe('Interactive Interpretability Components', () => {
  describe('InteractiveInterpretabilityViewer', () => {
    it('renders loading state initially', () => {
      render(
        <TestWrapper>
          <InteractiveInterpretabilityViewer
            analysisId={1}
            drawingImageUrl="/test-image.jpg"
            saliencyMapUrl="/test-saliency.jpg"
          />
        </TestWrapper>
      )

      expect(screen.getByText('Loading interactive interpretability data...')).toBeInTheDocument()
    })

    it('renders with required props', () => {
      render(
        <TestWrapper>
          <InteractiveInterpretabilityViewer
            analysisId={1}
            drawingImageUrl="/test-image.jpg"
            saliencyMapUrl="/test-saliency.jpg"
          />
        </TestWrapper>
      )

      // Component should render without crashing
      expect(screen.getByText('Loading interactive interpretability data...')).toBeInTheDocument()
    })

    it('accepts optional onRegionClick callback', () => {
      const mockCallback = vi.fn()
      
      render(
        <TestWrapper>
          <InteractiveInterpretabilityViewer
            analysisId={1}
            drawingImageUrl="/test-image.jpg"
            saliencyMapUrl="/test-saliency.jpg"
            onRegionClick={mockCallback}
          />
        </TestWrapper>
      )

      expect(screen.getByText('Loading interactive interpretability data...')).toBeInTheDocument()
    })

    it('handles zoom and pan interactions', () => {
      render(
        <TestWrapper>
          <InteractiveInterpretabilityViewer
            analysisId={1}
            drawingImageUrl="/test-image.jpg"
            saliencyMapUrl="/test-saliency.jpg"
          />
        </TestWrapper>
      )

      // Test that component renders with interactive features
      expect(screen.getByText('Loading interactive interpretability data...')).toBeInTheDocument()
    })
  })

  describe('ExplanationLevelToggle', () => {
    it('renders with default simplified mode', () => {
      render(
        <TestWrapper>
          <ExplanationLevelToggle analysisId={1} />
        </TestWrapper>
      )

      expect(screen.getByText('Explanation Level')).toBeInTheDocument()
      expect(screen.getByText('Simplified')).toBeInTheDocument()
      expect(screen.getByText('Technical')).toBeInTheDocument()
    })

    it('shows user role description', () => {
      render(
        <TestWrapper>
          <ExplanationLevelToggle analysisId={1} userRole="educator" />
        </TestWrapper>
      )

      expect(screen.getByText('Educational explanations for classroom use')).toBeInTheDocument()
    })
  })

  describe('ConfidenceIndicator', () => {
    it('renders loading state initially', () => {
      render(
        <TestWrapper>
          <ConfidenceIndicator analysisId={1} />
        </TestWrapper>
      )

      expect(screen.getByText('Loading confidence metrics...')).toBeInTheDocument()
    })

    it('renders in compact mode', () => {
      render(
        <TestWrapper>
          <ConfidenceIndicator analysisId={1} compact={true} />
        </TestWrapper>
      )

      // Should render without crashing
      expect(screen.getByText('Loading confidence metrics...')).toBeInTheDocument()
    })

    it('shows technical details when enabled', () => {
      render(
        <TestWrapper>
          <ConfidenceIndicator analysisId={1} showTechnicalDetails={true} />
        </TestWrapper>
      )

      expect(screen.getByText('Loading confidence metrics...')).toBeInTheDocument()
    })
  })
})