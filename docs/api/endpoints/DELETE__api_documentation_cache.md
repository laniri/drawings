# DELETE /api/documentation/cache

## Summary
Clear Documentation Cache

## Description
Clear documentation generation cache.

Forces regeneration of all documentation by clearing the cache.

## Tags
documentation

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{}
```


## Complete Request Example

```http
DELETE /api/documentation/cache
Content-Type: application/json
Accept: application/json
```

