import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import axios from 'axios'
import { render } from './utils'
import ConfigurationPage from '../pages/ConfigurationPage'

const mockedAxios = vi.mocked(axios)

const mockConfig = {
  vision_model: 'vit',
  threshold_percentile: 95.0,
  age_grouping_strategy: '1-year',
  anomaly_detection_method: 'autoencoder',
}

const mockAgeGroupModels = [
  {
    id: 1,
    age_min: 3,
    age_max: 4,
    model_type: 'autoencoder',
    vision_model: 'vit',
    sample_count: 150,
    threshold: 0.123,
    is_active: true,
    created_timestamp: '2023-01-01T00:00:00Z',
  },
  {
    id: 2,
    age_min: 5,
    age_max: 6,
    model_type: 'autoencoder',
    vision_model: 'vit',
    sample_count: 200,
    threshold: 0.156,
    is_active: true,
    created_timestamp: '2023-01-02T00:00:00Z',
  },
]

describe('ConfigurationPage Integration Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockedAxios.get.mockImplementation((url) => {
      if (url === '/api/config') {
        return Promise.resolve({ data: mockConfig })
      }
      if (url === '/api/models/age-groups') {
        return Promise.resolve({ data: mockAgeGroupModels })
      }
      return Promise.reject(new Error('Unknown URL'))
    })
  })

  it('renders configuration interface correctly', async () => {
    render(<ConfigurationPage />)
    
    await waitFor(() => {
      expect(screen.getByText('System Configuration')).toBeInTheDocument()
      expect(screen.getByText('General Settings')).toBeInTheDocument()
      expect(screen.getByText('Age Group Models')).toBeInTheDocument()
    })
  })

  it('displays age group models correctly', async () => {
    render(<ConfigurationPage />)
    
    await waitFor(() => {
      expect(screen.getByText('Ages 3-4')).toBeInTheDocument()
      expect(screen.getByText('Ages 5-6')).toBeInTheDocument()
      expect(screen.getByText('Samples: 150 • Threshold: 0.123')).toBeInTheDocument()
      expect(screen.getByText('Samples: 200 • Threshold: 0.156')).toBeInTheDocument()
    })
  })

  it('shows train new button', async () => {
    render(<ConfigurationPage />)
    
    await waitFor(() => {
      expect(screen.getByText('Train New')).toBeInTheDocument()
    })
  })

  it('shows save configuration button', async () => {
    render(<ConfigurationPage />)
    
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Save Configuration/ })).toBeInTheDocument()
    })
  })
})