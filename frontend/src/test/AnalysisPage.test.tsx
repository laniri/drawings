import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import axios from 'axios'
import { render } from './utils'
import AnalysisPage from '../pages/AnalysisPage'

const mockedAxios = vi.mocked(axios)

// Mock useParams
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useParams: () => ({ id: '123' }),
    useNavigate: () => vi.fn(),
  }
})

const mockAnalysisData = {
  drawing: {
    id: 123,
    filename: 'test-drawing.png',
    age_years: 5.5,
    subject: 'house',
    expert_label: 'normal',
    upload_timestamp: '2023-01-01T00:00:00Z',
  },
  analysis: {
    id: 456,
    anomaly_score: 0.123,
    normalized_score: 0.456,
    is_anomaly: false,
    confidence: 0.89,
    age_group: '5-6',
    method_used: 'autoencoder',
    analysis_timestamp: '2023-01-01T01:00:00Z',
  },
  interpretability: {
    saliency_map_url: '/api/saliency/123.png',
    overlay_image_url: '/api/overlay/123.png',
    explanation_text: 'The drawing shows normal developmental patterns.',
    importance_regions: [
      { x: 10, y: 20, width: 30, height: 40, importance: 0.8 },
    ],
  },
}

describe('AnalysisPage Integration Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('displays analysis results correctly', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: mockAnalysisData })
    
    render(<AnalysisPage />)
    
    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 4, name: 'Analysis Results' })).toBeInTheDocument()
      expect(screen.getByText('test-drawing.png')).toBeInTheDocument()
      expect(screen.getByText('Age: 5.5 years')).toBeInTheDocument()
      expect(screen.getByText('Subject: house')).toBeInTheDocument()
    })
  })

  it('handles loading state', () => {
    mockedAxios.get.mockImplementationOnce(() => new Promise(() => {})) // Never resolves
    
    render(<AnalysisPage />)
    
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
  })

  it('handles error state', async () => {
    mockedAxios.get.mockRejectedValueOnce(new Error('Network error'))
    
    render(<AnalysisPage />)
    
    await waitFor(() => {
      expect(screen.getByText('Failed to load analysis results. Please try again.')).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /Run Analysis/ })).toBeInTheDocument()
    })
  })
})