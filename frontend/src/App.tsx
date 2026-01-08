import { Routes, Route } from 'react-router-dom'
import { Container } from '@mui/material'

import Layout from './components/Layout/Layout'
import ProtectedRoute from './components/ProtectedRoute'
import UploadPage from './pages/UploadPage'
import AnalysisPage from './pages/AnalysisPage'
import DashboardPage from './pages/DashboardPage'
import ConfigurationPage from './pages/ConfigurationPage'
import BatchProcessingPage from './pages/BatchProcessingPage'
import DocumentationPage from './pages/DocumentationPage'
import DemoPage from './pages/DemoPage'

function App() {
  return (
    <Layout>
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Routes>
          {/* Public routes */}
          <Route path="/" element={<DemoPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/analysis/:id" element={<AnalysisPage />} />
          <Route path="/documentation" element={<DocumentationPage />} />
          
          {/* Protected routes - require authentication */}
          <Route 
            path="/dashboard" 
            element={
              <ProtectedRoute>
                <DashboardPage />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/batch" 
            element={
              <ProtectedRoute>
                <BatchProcessingPage />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/config" 
            element={
              <ProtectedRoute>
                <ConfigurationPage />
              </ProtectedRoute>
            } 
          />
        </Routes>
      </Container>
    </Layout>
  )
}

export default App
