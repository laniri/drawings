import React, { useState, useMemo } from 'react'
import {
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Box,
  Typography,
  Chip,
  Avatar,
  InputAdornment,
  ListSubheader,
} from '@mui/material'
import {
  Search,
  Person,
  Home,
  Nature,
  Pets,
  DirectionsCar,
  School,
  Category,
  Help,
} from '@mui/icons-material'
import { Control, Controller } from 'react-hook-form'

// Subject categories with icons and groupings
const SUBJECT_CATEGORIES = {
  unspecified: { label: 'Unspecified', icon: Help, group: 'Default' },
  
  // People and body parts
  person: { label: 'Person', icon: Person, group: 'People & Body' },
  face: { label: 'Face', icon: Person, group: 'People & Body' },
  hand: { label: 'Hand', icon: Person, group: 'People & Body' },
  family: { label: 'Family', icon: Person, group: 'People & Body' },
  
  // Animals
  bear: { label: 'Bear', icon: Pets, group: 'Animals' },
  bee: { label: 'Bee', icon: Pets, group: 'Animals' },
  bird: { label: 'Bird', icon: Pets, group: 'Animals' },
  camel: { label: 'Camel', icon: Pets, group: 'Animals' },
  cat: { label: 'Cat', icon: Pets, group: 'Animals' },
  cow: { label: 'Cow', icon: Pets, group: 'Animals' },
  dog: { label: 'Dog', icon: Pets, group: 'Animals' },
  elephant: { label: 'Elephant', icon: Pets, group: 'Animals' },
  fish: { label: 'Fish', icon: Pets, group: 'Animals' },
  frog: { label: 'Frog', icon: Pets, group: 'Animals' },
  horse: { label: 'Horse', icon: Pets, group: 'Animals' },
  octopus: { label: 'Octopus', icon: Pets, group: 'Animals' },
  rabbit: { label: 'Rabbit', icon: Pets, group: 'Animals' },
  sheep: { label: 'Sheep', icon: Pets, group: 'Animals' },
  snail: { label: 'Snail', icon: Pets, group: 'Animals' },
  spider: { label: 'Spider', icon: Pets, group: 'Animals' },
  tiger: { label: 'Tiger', icon: Pets, group: 'Animals' },
  whale: { label: 'Whale', icon: Pets, group: 'Animals' },
  
  // Objects & Household Items
  TV: { label: 'TV', icon: School, group: 'Objects & Household' },
  bed: { label: 'Bed', icon: Home, group: 'Objects & Household' },
  book: { label: 'Book', icon: School, group: 'Objects & Household' },
  bottle: { label: 'Bottle', icon: Category, group: 'Objects & Household' },
  bowl: { label: 'Bowl', icon: Category, group: 'Objects & Household' },
  chair: { label: 'Chair', icon: Home, group: 'Objects & Household' },
  clock: { label: 'Clock', icon: Category, group: 'Objects & Household' },
  couch: { label: 'Couch', icon: Home, group: 'Objects & Household' },
  cup: { label: 'Cup', icon: Category, group: 'Objects & Household' },
  hat: { label: 'Hat', icon: Category, group: 'Objects & Household' },
  house: { label: 'House', icon: Home, group: 'Objects & Household' },
  key: { label: 'Key', icon: Category, group: 'Objects & Household' },
  lamp: { label: 'Lamp', icon: Home, group: 'Objects & Household' },
  phone: { label: 'Phone', icon: Category, group: 'Objects & Household' },
  piano: { label: 'Piano', icon: School, group: 'Objects & Household' },
  scissors: { label: 'Scissors', icon: Category, group: 'Objects & Household' },
  watch: { label: 'Watch', icon: Category, group: 'Objects & Household' },
  
  // Transportation
  airplane: { label: 'Airplane', icon: DirectionsCar, group: 'Transportation' },
  bike: { label: 'Bike', icon: DirectionsCar, group: 'Transportation' },
  boat: { label: 'Boat', icon: DirectionsCar, group: 'Transportation' },
  car: { label: 'Car', icon: DirectionsCar, group: 'Transportation' },
  train: { label: 'Train', icon: DirectionsCar, group: 'Transportation' },
  
  // Nature & Food
  apple: { label: 'Apple', icon: Nature, group: 'Nature & Food' },
  cactus: { label: 'Cactus', icon: Nature, group: 'Nature & Food' },
  'ice cream': { label: 'Ice Cream', icon: Nature, group: 'Nature & Food' },
  mushroom: { label: 'Mushroom', icon: Nature, group: 'Nature & Food' },
  tree: { label: 'Tree', icon: Nature, group: 'Nature & Food' },
  
  // Abstract & Other
  abstract: { label: 'Abstract', icon: Category, group: 'Abstract & Other' },
  other: { label: 'Other', icon: Category, group: 'Abstract & Other' },
}

interface SubjectCategorySelectProps {
  control: Control<any>
  name: string
  label?: string
  required?: boolean
  error?: boolean
  helperText?: string
  showSearch?: boolean
}

const SubjectCategorySelect: React.FC<SubjectCategorySelectProps> = ({
  control,
  name,
  label = 'Drawing Subject',
  required = false,
  error = false,
  helperText,
  showSearch = true,
}) => {
  const [searchTerm, setSearchTerm] = useState('')

  // Group categories by their group
  const groupedCategories = useMemo(() => {
    const groups: Record<string, Array<{ key: string; category: typeof SUBJECT_CATEGORIES[keyof typeof SUBJECT_CATEGORIES] }>> = {}
    
    Object.entries(SUBJECT_CATEGORIES).forEach(([key, category]) => {
      if (!groups[category.group]) {
        groups[category.group] = []
      }
      groups[category.group].push({ key, category })
    })
    
    return groups
  }, [])

  // Filter categories based on search term
  const filteredCategories = useMemo(() => {
    if (!searchTerm) return groupedCategories
    
    const filtered: Record<string, Array<{ key: string; category: typeof SUBJECT_CATEGORIES[keyof typeof SUBJECT_CATEGORIES] }>> = {}
    
    Object.entries(groupedCategories).forEach(([groupName, categories]) => {
      const matchingCategories = categories.filter(({ category }) =>
        category.label.toLowerCase().includes(searchTerm.toLowerCase())
      )
      
      if (matchingCategories.length > 0) {
        filtered[groupName] = matchingCategories
      }
    })
    
    return filtered
  }, [groupedCategories, searchTerm])

  const renderCategoryIcon = (IconComponent: React.ComponentType<any>) => (
    <Avatar sx={{ width: 24, height: 24, bgcolor: 'primary.light', mr: 1 }}>
      <IconComponent sx={{ fontSize: 14 }} />
    </Avatar>
  )

  return (
    <Box>
      {showSearch && (
        <TextField
          fullWidth
          size="small"
          placeholder="Search subjects..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
          }}
          sx={{ mb: 2 }}
        />
      )}
      
      <Controller
        name={name}
        control={control}
        render={({ field }) => (
          <FormControl fullWidth margin="normal" error={error}>
            <InputLabel required={required}>{label}</InputLabel>
            <Select
              {...field}
              label={label}
              value={field.value || ''}
              renderValue={(selected) => {
                if (!selected) {
                  return <Typography color="text.secondary">Optional - leave blank if unsure</Typography>
                }
                
                const category = SUBJECT_CATEGORIES[selected as keyof typeof SUBJECT_CATEGORIES]
                if (!category) return selected
                
                return (
                  <Box display="flex" alignItems="center">
                    {renderCategoryIcon(category.icon)}
                    {category.label}
                  </Box>
                )
              }}
            >
              <MenuItem value="">
                <Box display="flex" alignItems="center">
                  {renderCategoryIcon(Help)}
                  <Box>
                    <Typography>Unspecified</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Use when subject is unclear or mixed
                    </Typography>
                  </Box>
                </Box>
              </MenuItem>
              
              {Object.entries(filteredCategories).map(([groupName, categories]) => [
                <ListSubheader key={groupName} sx={{ bgcolor: 'background.paper' }}>
                  {groupName}
                </ListSubheader>,
                ...categories.map(({ key, category }) => (
                  <MenuItem key={key} value={key}>
                    <Box display="flex" alignItems="center">
                      {renderCategoryIcon(category.icon)}
                      {category.label}
                    </Box>
                  </MenuItem>
                ))
              ]).flat()}
            </Select>
            
            {helperText && (
              <Typography variant="caption" color={error ? 'error' : 'text.secondary'} sx={{ mt: 0.5, ml: 1.5 }}>
                {helperText}
              </Typography>
            )}
            
            {!helperText && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, ml: 1.5 }}>
                Optional: Select the main subject of the drawing to improve analysis accuracy
              </Typography>
            )}
          </FormControl>
        )}
      />
      
      {/* Popular subjects as chips for quick selection */}
      <Box sx={{ mt: 1 }}>
        <Typography variant="caption" color="text.secondary" gutterBottom>
          Popular subjects:
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
          {['person', 'house', 'tree', 'car', 'family', 'cat', 'dog'].map((subjectKey) => {
            const category = SUBJECT_CATEGORIES[subjectKey as keyof typeof SUBJECT_CATEGORIES]
            if (!category) return null
            
            return (
              <Controller
                key={subjectKey}
                name={name}
                control={control}
                render={({ field }) => (
                  <Chip
                    size="small"
                    label={category.label}
                    variant={field.value === subjectKey ? 'filled' : 'outlined'}
                    color={field.value === subjectKey ? 'primary' : 'default'}
                    onClick={() => field.onChange(field.value === subjectKey ? '' : subjectKey)}
                    sx={{ cursor: 'pointer' }}
                  />
                )}
              />
            )
          })}
        </Box>
      </Box>
    </Box>
  )
}

export default SubjectCategorySelect