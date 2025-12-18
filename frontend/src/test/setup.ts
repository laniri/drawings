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
Object.defineProperty(window, 'URL', {
  value: {
    createObjectURL: vi.fn(() => 'mocked-url'),
    revokeObjectURL: vi.fn()
  }
})

// Mock ResizeObserver
Object.defineProperty(window, 'ResizeObserver', {
  value: class ResizeObserver {
    observe = vi.fn()
    unobserve = vi.fn()
    disconnect = vi.fn()
  }
})

// Mock IntersectionObserver
Object.defineProperty(window, 'IntersectionObserver', {
  value: class IntersectionObserver {
    observe = vi.fn()
    unobserve = vi.fn()
    disconnect = vi.fn()
  }
})