import React, { useState } from 'react'
import {
  Typography,
  Box,
  Grid,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Alert,
  Card,
  CardContent,
  CardHeader,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  LinearProgress,
} from '@mui/material'
import {
  Save,
  Settings,
  ModelTraining,
  Delete,
  Add,
} from '@mui/icons-material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useForm, Controller } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import axios from 'axios'

// Configuration schema
const configSchema = z.object({
  threshold_percentile: z.number().min(50).max(99.9),
  age_grouping_strategy: z.enum(['1-year', '2-year', '3-year']),
  min_samples_per_group: z.number().min(10).optional(),
  max_age_group_span: z.number().min(1).max(16).optional(),
  vision_model: z.literal('vit'),
  anomaly_detection_method: z.literal('autoencoder'),
})

type ConfigFormData = z.infer<typeof configSchema>

interface SystemConfig {
  vision_model: string
  threshold_percentile: number
  age_grouping_strategy: string
  anomaly_detection_method: string
  min_samples_per_group?: number
  max_age_group_span?: number
}

interface AgeGroupModel {
  id: number
  age_min: number
  age_max: number
  model_type: string
  vision_model: string
  sample_count: number
  threshold: number
  is_active: boolean
  created_timestamp: string
}

const ConfigurationPage: React.FC = () => {
  const [trainDialogOpen, setTrainDialogOpen] = useState(false)
  const [selectedAgeGroup, setSelectedAgeGroup] = useState<string>('')
  const [justSubmitted, setJustSubmitted] = useState(false)
  const queryClient = useQueryClient()

  // Fetch current configuration
  const { data: config, isLoading: configLoading } = useQuery<SystemConfig>({
    queryKey: ['system-config'],
    queryFn: async () => {
      const response = await axios.get('/api/config/')
      return response.data
    },
  })

  // Fetch age group models
  const { data: ageGroupModels, isLoading: modelsLoading } = useQuery<AgeGroupModel[]>({
    queryKey: ['age-group-models'],
    queryFn: async () => {
      const response = await axios.get('/api/models/age-groups')
      return response.data.models
    },
  })

  const {
    control,
    handleSubmit,
    reset,
    formState: { isDirty },
  } = useForm<ConfigFormData>({
    resolver: zodResolver(configSchema),
    defaultValues: {
      threshold_percentile: 95.0,
      min_samples_per_group: 50,
      max_age_group_span: 4.0,
      anomaly_detection_method: 'autoencoder',
      age_grouping_strategy: '1-year',
      vision_model: 'vit',
    },
  })

  // Update threshold configuration
  const updateThresholdMutation = useMutation({
    mutationFn: async (percentile: number) => {
      const response = await axios.put('/api/config/threshold', {
        percentile: percentile
      })
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-config'] })
      // Invalidate dashboard stats since threshold changes affect anomaly classifications
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] })
    },
  })

  // Update age grouping configuration
  const updateAgeGroupingMutation = useMutation({
    mutationFn: async (data: { threshold_percentile?: number; age_grouping_strategy?: string; min_samples_per_group?: number; max_age_group_span?: number }) => {
      const response = await axios.put('/api/config/age-grouping', data)
      return response.data
    },
    onSuccess: (data, variables) => {
      // Update the query cache with the new values to prevent form reset
      queryClient.setQueryData(['system-config'], (oldData: SystemConfig | undefined) => {
        if (!oldData) return oldData
        return {
          ...oldData,
          threshold_percentile: variables.threshold_percentile ?? oldData.threshold_percentile,
          age_grouping_strategy: variables.age_grouping_strategy ?? oldData.age_grouping_strategy,
          min_samples_per_group: variables.min_samples_per_group ?? oldData.min_samples_per_group,
          max_age_group_span: variables.max_age_group_span ?? oldData.max_age_group_span,
        }
      })
      
      // Don't invalidate system-config to prevent form reset, but invalidate dashboard stats
      // since threshold changes affect anomaly classifications
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] })
      
      // Reset the flag after a short delay
      setTimeout(() => setJustSubmitted(false), 2000)
    },
  })

  // Update general configuration (fallback)
  const updateConfigMutation = useMutation({
    mutationFn: async (data: ConfigFormData) => {
      const response = await axios.put('/api/config/', data)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-config'] })
      // Invalidate dashboard stats since config changes might affect anomaly classifications
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] })
      reset()
    },
  })

  // Train new model
  const trainModelMutation = useMutation({
    mutationFn: async (ageGroup: string) => {
      const response = await axios.post('/api/models/train', {
        age_group: ageGroup,
      })
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['age-group-models'] })
      setTrainDialogOpen(false)
      setSelectedAgeGroup('')
    },
  })

  // Delete model
  const deleteModelMutation = useMutation({
    mutationFn: async (modelId: number) => {
      await axios.delete(`/api/models/${modelId}`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['age-group-models'] })
    },
  })

  const onSubmit = (data: ConfigFormData) => {
    // Use the age-grouping endpoint for all updates (workaround)
    const mappedStrategy = data.age_grouping_strategy === '1-year' ? 'yearly' : data.age_grouping_strategy
    
    setJustSubmitted(true)
    updateAgeGroupingMutation.mutate({
      threshold_percentile: data.threshold_percentile,
      age_grouping_strategy: mappedStrategy,
      min_samples_per_group: data.min_samples_per_group,
      max_age_group_span: data.max_age_group_span,
    })
  }

  const handleTrainModel = () => {
    if (selectedAgeGroup) {
      trainModelMutation.mutate(selectedAgeGroup)
    }
  }

  // Reset form when config data loads or updates (but not immediately after submission)
  React.useEffect(() => {
    if (config && !justSubmitted) {
      // Map API values to frontend values
      const mapAgeGroupingStrategy = (apiValue: string): '1-year' | '2-year' | '3-year' => {
        switch (apiValue) {
          case 'yearly': return '1-year'
          case '2-year': return '2-year'
          case '3-year': return '3-year'
          default: return '1-year'
        }
      }

      reset({
        threshold_percentile: config.threshold_percentile,
        min_samples_per_group: config.min_samples_per_group,
        max_age_group_span: config.max_age_group_span,
        age_grouping_strategy: mapAgeGroupingStrategy(config.age_grouping_strategy),
        vision_model: config.vision_model || 'vit',
        anomaly_detection_method: config.anomaly_detection_method || 'autoencoder',
      })
    }
  }, [config, reset, justSubmitted])

  if (configLoading || modelsLoading) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Configuration
        </Typography>
        <LinearProgress />
      </Box>
    )
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Configuration
      </Typography>

      <Grid container spacing={3}>
        {/* General Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="General Settings"
              avatar={<Settings />}
            />
            <CardContent>
              <Box component="form" onSubmit={handleSubmit(onSubmit)}>
                <Controller
                  name="vision_model"
                  control={control}
                  render={({ field }) => (
                    <FormControl fullWidth margin="normal">
                      <InputLabel>Vision Model</InputLabel>
                      <Select {...field} label="Vision Model" disabled>
                        <MenuItem value="vit">Vision Transformer (ViT)</MenuItem>
                      </Select>
                    </FormControl>
                  )}
                />

                <Controller
                  name="anomaly_detection_method"
                  control={control}
                  render={({ field }) => (
                    <FormControl fullWidth margin="normal">
                      <InputLabel>Anomaly Detection Method</InputLabel>
                      <Select {...field} label="Anomaly Detection Method" disabled>
                        <MenuItem value="autoencoder">Autoencoder</MenuItem>
                      </Select>
                    </FormControl>
                  )}
                />

                <Controller
                  name="age_grouping_strategy"
                  control={control}
                  render={({ field }) => (
                    <FormControl fullWidth margin="normal">
                      <InputLabel>Age Grouping Strategy</InputLabel>
                      <Select {...field} label="Age Grouping Strategy">
                        <MenuItem value="1-year">1-Year Intervals</MenuItem>
                        <MenuItem value="2-year">2-Year Intervals</MenuItem>
                        <MenuItem value="3-year">3-Year Intervals</MenuItem>
                      </Select>
                    </FormControl>
                  )}
                />

                <Box sx={{ mt: 3, mb: 2 }}>
                  <Typography gutterBottom>
                    Threshold Percentile: {control._formValues.threshold_percentile || 95}%
                  </Typography>
                  <Controller
                    name="threshold_percentile"
                    control={control}
                    render={({ field }) => (
                      <Slider
                        {...field}
                        min={50}
                        max={99.9}
                        step={0.1}
                        marks={[
                          { value: 80, label: '80%' },
                          { value: 90, label: '90%' },
                          { value: 95, label: '95%' },
                          { value: 99, label: '99%' },
                        ]}
                        valueLabelDisplay="auto"
                      />
                    )}
                  />
                </Box>

                {(updateThresholdMutation.error || updateAgeGroupingMutation.error || updateConfigMutation.error) && (
                  <Alert severity="error" sx={{ mt: 2 }}>
                    Failed to update configuration. Please try again.
                  </Alert>
                )}

                {(updateThresholdMutation.isSuccess || updateAgeGroupingMutation.isSuccess || updateConfigMutation.isSuccess) && (
                  <Alert severity="success" sx={{ mt: 2 }}>
                    Configuration updated successfully!
                  </Alert>
                )}

                <Button
                  type="submit"
                  variant="contained"
                  fullWidth
                  disabled={!isDirty || updateThresholdMutation.isPending || updateAgeGroupingMutation.isPending || updateConfigMutation.isPending}
                  startIcon={<Save />}
                  sx={{ mt: 3 }}
                >
                  {(updateThresholdMutation.isPending || updateAgeGroupingMutation.isPending || updateConfigMutation.isPending) ? 'Saving...' : 'Save Configuration'}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Age Group Models */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Age Group Models"
              avatar={<ModelTraining />}
              action={
                <Button
                  size="small"
                  startIcon={<Add />}
                  onClick={() => setTrainDialogOpen(true)}
                >
                  Train New
                </Button>
              }
            />
            <CardContent>
              <List>
                {ageGroupModels?.map((model) => (
                  <React.Fragment key={model.id}>
                    <ListItem>
                      <ListItemText
                        primary={`Ages ${model.age_min}-${model.age_max}`}
                        secondary={
                          <Box>
                            <Typography variant="body2" component="span">
                              Samples: {model.sample_count} • Threshold: {model.threshold.toFixed(3)}
                            </Typography>
                            <br />
                            <Typography variant="caption" color="text.secondary">
                              Created: {new Date(model.created_timestamp).toLocaleDateString()}
                            </Typography>
                            <Box sx={{ mt: 1 }}>
                              <Chip
                                size="small"
                                label={model.is_active ? 'Active' : 'Inactive'}
                                color={model.is_active ? 'success' : 'default'}
                                variant="outlined"
                              />
                            </Box>
                          </Box>
                        }
                      />
                      <ListItemSecondaryAction>
                        <IconButton
                          edge="end"
                          onClick={() => deleteModelMutation.mutate(model.id)}
                          disabled={deleteModelMutation.isPending}
                        >
                          <Delete />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                    <Divider />
                  </React.Fragment>
                )) || (
                  <ListItem>
                    <ListItemText
                      primary="No models trained yet"
                      secondary="Train your first age group model to get started"
                    />
                  </ListItem>
                )}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Train Model Dialog */}
      <Dialog open={trainDialogOpen} onClose={() => setTrainDialogOpen(false)}>
        <DialogTitle>Train New Age Group Model</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>Age Group</InputLabel>
            <Select
              value={selectedAgeGroup}
              onChange={(e) => setSelectedAgeGroup(e.target.value)}
              label="Age Group"
            >
              <MenuItem value="3-4">Ages 3-4</MenuItem>
              <MenuItem value="4-5">Ages 4-5</MenuItem>
              <MenuItem value="5-6">Ages 5-6</MenuItem>
              <MenuItem value="6-7">Ages 6-7</MenuItem>
              <MenuItem value="7-8">Ages 7-8</MenuItem>
              <MenuItem value="8-9">Ages 8-9</MenuItem>
              <MenuItem value="9-10">Ages 9-10</MenuItem>
              <MenuItem value="10-11">Ages 10-11</MenuItem>
              <MenuItem value="11-12">Ages 11-12</MenuItem>
              <MenuItem value="12-13">Ages 12-13</MenuItem>
              <MenuItem value="13-14">Ages 13-14</MenuItem>
              <MenuItem value="14-15">Ages 14-15</MenuItem>
              <MenuItem value="15-16">Ages 15-16</MenuItem>
              <MenuItem value="16-17">Ages 16-17</MenuItem>
              <MenuItem value="17-18">Ages 17-18</MenuItem>
            </Select>
          </FormControl>
          <Alert severity="info" sx={{ mt: 2 }}>
            Training a new model requires sufficient normal examples from the selected age group.
            The process may take several minutes to complete.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTrainDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleTrainModel}
            variant="contained"
            disabled={!selectedAgeGroup || trainModelMutation.isPending}
          >
            {trainModelMutation.isPending ? 'Training...' : 'Train Model'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default ConfigurationPage