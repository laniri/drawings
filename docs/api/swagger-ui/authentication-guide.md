# Authentication Guide

This document provides guidance on how to authenticate with the API using the interactive Swagger UI.

## Available Authentication Methods

No authentication methods are configured for this API.
## Quick Authentication Tips

### Using the Enhanced Authentication Section
This Swagger UI includes an enhanced authentication section above the API documentation:

1. **Bearer Token**: Enter your token directly (without "Bearer " prefix)
2. **API Key**: Enter your API key value
3. **Basic Auth**: Enter in format `username:password`
4. Click "Apply" to authenticate
5. Use "Clear All Authentication" to remove all credentials

### Keyboard Shortcuts
- **Ctrl/Cmd + K**: Focus the search box
- **Escape**: Clear search or close focused elements

### Testing Endpoints
1. Find the endpoint you want to test
2. Click "Try it out" button
3. Fill in required parameters
4. Click "Execute" to send the request
5. View the response below

### Troubleshooting Authentication
- **401 Unauthorized**: Check that your credentials are correct and properly formatted
- **403 Forbidden**: You may not have permission to access this resource
- **Token Expired**: Refresh your token and re-authenticate

### Security Best Practices
- Never share your API keys or tokens
- Use HTTPS in production environments
- Regularly rotate your credentials
- Monitor API usage for suspicious activity

## Advanced Features

### Search and Filtering
- Use the search box to find specific endpoints
- Click on tag filters to show only endpoints with specific tags
- Combine search and tag filters for precise results

### Export Functionality
- Click "Export Filtered" to download the current filtered API specification
- Use "Expand All" / "Collapse All" to control endpoint visibility

### Copy Code Examples
- Hover over code blocks to see copy buttons
- Click to copy request/response examples to clipboard
