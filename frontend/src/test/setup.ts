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

// Mock URL constructor and methods
Object.defineProperty(globalThis, 'URL', {
  value: class URL {
    constructor(url: string, base?: string) {
      this.href = base ? `${base}/${url}` : url
      this.origin = base || 'http://localhost:3000'
      this.protocol = 'http:'
      this.host = 'localhost:3000'
      this.hostname = 'localhost'
      this.port = '3000'
      this.pathname = url.startsWith('/') ? url : `/${url}`
      this.search = ''
      this.hash = ''
    }

    href: string
    origin: string
    protocol: string
    host: string
    hostname: string
    port: string
    pathname: string
    search: string
    hash: string

    static createObjectURL = vi.fn(() => 'mocked-url')
    static revokeObjectURL = vi.fn()

    toString() {
      return this.href
    }
  },
})

// Also set it on window for browser compatibility
Object.defineProperty(window, 'URL', {
  value: globalThis.URL,
})

// Mock ResizeObserver
Object.defineProperty(window, 'ResizeObserver', {
  value: class ResizeObserver {
    observe = vi.fn()
    unobserve = vi.fn()
    disconnect = vi.fn()
  },
})

// Mock IntersectionObserver
Object.defineProperty(window, 'IntersectionObserver', {
  value: class IntersectionObserver {
    observe = vi.fn()
    unobserve = vi.fn()
    disconnect = vi.fn()
  },
})
