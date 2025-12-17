import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen } from '@testing-library/react'
import { render } from './utils'
import UploadPage from '../pages/UploadPage'

describe('UploadPage Integration Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders upload interface correctly', () => {
    render(<UploadPage />)
    
    expect(screen.getByRole('heading', { name: 'Upload Drawing' })).toBeInTheDocument()
    expect(screen.getByText('Select Drawing')).toBeInTheDocument()
    expect(screen.getByText('Drawing Information')).toBeInTheDocument()
    expect(screen.getByLabelText(/Child's Age/)).toBeInTheDocument()
  })

  it('shows upload button', () => {
    render(<UploadPage />)
    
    const uploadButton = screen.getByRole('button', { name: /Upload Drawing/ })
    expect(uploadButton).toBeInTheDocument()
    expect(uploadButton).toBeDisabled() // Should be disabled without file
  })

  it('displays form fields correctly', () => {
    render(<UploadPage />)
    
    expect(screen.getByLabelText(/Child's Age/)).toBeInTheDocument()
    expect(screen.getByLabelText(/Drawing Subject/)).toBeInTheDocument()
    expect(screen.getAllByText('Expert Label (optional)').length).toBeGreaterThan(0)
    expect(screen.getByLabelText(/Drawing Tool/)).toBeInTheDocument()
    expect(screen.getByLabelText(/Drawing Prompt/)).toBeInTheDocument()
  })

  it('shows dropzone with correct text', () => {
    render(<UploadPage />)
    
    expect(screen.getByText(/Drag and drop an image here/)).toBeInTheDocument()
    expect(screen.getByText(/Supported formats: PNG, JPEG, BMP/)).toBeInTheDocument()
  })
})