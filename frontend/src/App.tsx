import { Routes, Route } from 'react-router-dom'
import { Container } from '@mui/material'

import Layout from './components/Layout/Layout'
import UploadPage from './pages/UploadPage'
import AnalysisPage from './pages/AnalysisPage'
import DashboardPage from './pages/DashboardPage'
import ConfigurationPage from './pages/ConfigurationPage'
import BatchProcessingPage from './pages/BatchProcessingPage'
import DocumentationPage from './pages/DocumentationPage'

function App() {
  return (
    <Layout>
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/analysis/:id" element={<AnalysisPage />} />
          <Route path="/batch" element={<BatchProcessingPage />} />
          <Route path="/config" element={<ConfigurationPage />} />
          <Route path="/documentation" element={<DocumentationPage />} />
        </Routes>
      </Container>
    </Layout>
  )
}

export default App