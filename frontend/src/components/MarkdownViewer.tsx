import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import { Box, Paper, Typography, styled } from '@mui/material';

interface MarkdownViewerProps {
  content: string;
  title?: string;
  maxHeight?: string | number;
}

const StyledMarkdownContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  '& h1': {
    fontSize: '2rem',
    fontWeight: 600,
    marginTop: theme.spacing(3),
    marginBottom: theme.spacing(2),
    borderBottom: `2px solid ${theme.palette.divider}`,
    paddingBottom: theme.spacing(1),
  },
  '& h2': {
    fontSize: '1.5rem',
    fontWeight: 600,
    marginTop: theme.spacing(2.5),
    marginBottom: theme.spacing(1.5),
    borderBottom: `1px solid ${theme.palette.divider}`,
    paddingBottom: theme.spacing(0.5),
  },
  '& h3': {
    fontSize: '1.25rem',
    fontWeight: 600,
    marginTop: theme.spacing(2),
    marginBottom: theme.spacing(1),
  },
  '& h4, & h5, & h6': {
    fontSize: '1.1rem',
    fontWeight: 600,
    marginTop: theme.spacing(1.5),
    marginBottom: theme.spacing(1),
  },
  '& p': {
    marginBottom: theme.spacing(2),
    lineHeight: 1.7,
  },
  '& ul, & ol': {
    marginBottom: theme.spacing(2),
    paddingLeft: theme.spacing(3),
  },
  '& li': {
    marginBottom: theme.spacing(0.5),
    lineHeight: 1.6,
  },
  '& code': {
    backgroundColor: theme.palette.mode === 'dark' ? '#2d2d2d' : '#f5f5f5',
    padding: '2px 6px',
    borderRadius: 4,
    fontSize: '0.9em',
    fontFamily: 'monospace',
  },
  '& pre': {
    backgroundColor: theme.palette.mode === 'dark' ? '#2d2d2d' : '#f5f5f5',
    padding: theme.spacing(2),
    borderRadius: theme.shape.borderRadius,
    overflow: 'auto',
    marginBottom: theme.spacing(2),
    '& code': {
      backgroundColor: 'transparent',
      padding: 0,
    },
  },
  '& blockquote': {
    borderLeft: `4px solid ${theme.palette.primary.main}`,
    paddingLeft: theme.spacing(2),
    marginLeft: 0,
    marginBottom: theme.spacing(2),
    color: theme.palette.text.secondary,
    fontStyle: 'italic',
  },
  '& table': {
    width: '100%',
    borderCollapse: 'collapse',
    marginBottom: theme.spacing(2),
  },
  '& th, & td': {
    border: `1px solid ${theme.palette.divider}`,
    padding: theme.spacing(1),
    textAlign: 'left',
  },
  '& th': {
    backgroundColor: theme.palette.mode === 'dark' ? '#2d2d2d' : '#f5f5f5',
    fontWeight: 600,
  },
  '& a': {
    color: theme.palette.primary.main,
    textDecoration: 'none',
    '&:hover': {
      textDecoration: 'underline',
    },
  },
  '& img': {
    maxWidth: '100%',
    height: 'auto',
    borderRadius: theme.shape.borderRadius,
    marginBottom: theme.spacing(2),
  },
  '& hr': {
    border: 'none',
    borderTop: `1px solid ${theme.palette.divider}`,
    marginTop: theme.spacing(3),
    marginBottom: theme.spacing(3),
  },
}));

export const MarkdownViewer: React.FC<MarkdownViewerProps> = ({
  content,
  title,
  maxHeight,
}) => {
  return (
    <Box>
      {title && (
        <Typography variant="h4" gutterBottom sx={{ mb: 2 }}>
          {title}
        </Typography>
      )}
      <StyledMarkdownContainer
        elevation={2}
        sx={{
          maxHeight: maxHeight || 'none',
          overflow: maxHeight ? 'auto' : 'visible',
        }}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw, rehypeSanitize]}
        >
          {content}
        </ReactMarkdown>
      </StyledMarkdownContainer>
    </Box>
  );
};

export default MarkdownViewer;
