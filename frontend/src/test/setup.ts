import '@testing-library/jest-dom'

// Mock axios for tests
import { vi } from 'vitest'

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}))

// Mock URL.createObjectURL and URL.revokeObjectURL
global.URL.createObjectURL = vi.fn(() => 'mocked-url')
global.URL.revokeObjectURL = vi.fn()

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  observe = vi.fn()
  unobserve = vi.fn()
  disconnect = vi.fn()
}

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  observe = vi.fn()
  unobserve = vi.fn()
  disconnect = vi.fn()
}