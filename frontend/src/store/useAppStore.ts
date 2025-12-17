import { create } from 'zustand'

interface Drawing {
  id: number
  filename: string
  age_years: number
  subject?: string
  upload_timestamp: string
}

interface AnalysisResult {
  id: number
  drawing_id: number
  anomaly_score: number
  is_anomaly: boolean
  confidence: number
  analysis_timestamp: string
}

interface SystemConfig {
  vision_model: string
  threshold_percentile: number
  age_grouping_strategy: string
  anomaly_detection_method: string
}

interface ModelStatus {
  model_type: string
  is_loaded: boolean
  last_updated: string
}

interface AppState {
  // Drawing management
  drawings: Drawing[]
  selectedDrawing: Drawing | null
  uploadProgress: number
  
  // Analysis state
  currentAnalysis: AnalysisResult | null
  analysisHistory: AnalysisResult[]
  isAnalyzing: boolean
  
  // Configuration
  systemConfig: SystemConfig
  modelStatus: ModelStatus
  
  // UI state
  sidebarOpen: boolean
  currentView: 'upload' | 'analysis' | 'dashboard' | 'config'
  
  // Actions
  setDrawings: (drawings: Drawing[]) => void
  setSelectedDrawing: (drawing: Drawing | null) => void
  setUploadProgress: (progress: number) => void
  setCurrentAnalysis: (analysis: AnalysisResult | null) => void
  setIsAnalyzing: (analyzing: boolean) => void
  setSystemConfig: (config: SystemConfig) => void
  setCurrentView: (view: 'upload' | 'analysis' | 'dashboard' | 'config') => void
}

export const useAppStore = create<AppState>((set) => ({
  // Initial state
  drawings: [],
  selectedDrawing: null,
  uploadProgress: 0,
  currentAnalysis: null,
  analysisHistory: [],
  isAnalyzing: false,
  systemConfig: {
    vision_model: 'vit',
    threshold_percentile: 95.0,
    age_grouping_strategy: '1-year',
    anomaly_detection_method: 'autoencoder',
  },
  modelStatus: {
    model_type: 'vit',
    is_loaded: false,
    last_updated: '',
  },
  sidebarOpen: true,
  currentView: 'dashboard',
  
  // Actions
  setDrawings: (drawings) => set({ drawings }),
  setSelectedDrawing: (drawing) => set({ selectedDrawing: drawing }),
  setUploadProgress: (progress) => set({ uploadProgress: progress }),
  setCurrentAnalysis: (analysis) => set({ currentAnalysis: analysis }),
  setIsAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),
  setSystemConfig: (config) => set({ systemConfig: config }),
  setCurrentView: (view) => set({ currentView: view }),
}))