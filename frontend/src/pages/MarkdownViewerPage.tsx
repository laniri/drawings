import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Box,
  Container,
  CircularProgress,
  Alert,
  TextField,
  Button,
  Stack,
} from '@mui/material';
import { MarkdownViewer } from '../components/MarkdownViewer';
import DescriptionIcon from '@mui/icons-material/Description';

export const MarkdownViewerPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filePath, setFilePath] = useState<string>('');

  // Load file from URL parameter on mount
  useEffect(() => {
    const fileParam = searchParams.get('file');
    if (fileParam) {
      setFilePath(fileParam);
      loadMarkdownFile(fileParam);
    }
  }, [searchParams]);

  const loadMarkdownFile = async (path: string) => {
    if (!path.trim()) {
      setError('Please enter a file path');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // If path doesn't start with docs/, tmp_files/, or an absolute path, prepend docs/
      // This handles documentation files which are stored in docs/ directory
      let fullPath = path;
      if (!path.startsWith('docs/') && !path.startsWith('tmp_files/') && !path.startsWith('/')) {
        fullPath = `docs/${path}`;
      }

      // Try to fetch from the backend API
      const response = await fetch(`/api/v1/files/markdown?path=${encodeURIComponent(fullPath)}`);
      
      if (!response.ok) {
        throw new Error(`Failed to load file: ${response.statusText}`);
      }

      const text = await response.text();
      setContent(text);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load markdown file');
      setContent('');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadFile = () => {
    loadMarkdownFile(filePath);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleLoadFile();
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack spacing={3}>
        <Box>
          <Stack direction="row" spacing={2} alignItems="center">
            <TextField
              fullWidth
              label="Markdown File Path"
              placeholder="e.g., tmp_files/upload_issue_detailed_analysis.md"
              value={filePath}
              onChange={(e) => setFilePath(e.target.value)}
              onKeyPress={handleKeyPress}
              variant="outlined"
            />
            <Button
              variant="contained"
              onClick={handleLoadFile}
              disabled={loading || !filePath.trim()}
              startIcon={<DescriptionIcon />}
              sx={{ minWidth: 120 }}
            >
              Load
            </Button>
          </Stack>
        </Box>

        {loading && (
          <Box display="flex" justifyContent="center" py={4}>
            <CircularProgress />
          </Box>
        )}

        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {content && !loading && (
          <MarkdownViewer content={content} />
        )}

        {!content && !loading && !error && (
          <Alert severity="info">
            Enter a markdown file path and click "Load" to view the content.
          </Alert>
        )}
      </Stack>
    </Container>
  );
};

export default MarkdownViewerPage;
