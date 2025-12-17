# Quick Fix for Training Timeout Issue

## Current Situation
Your server is blocked by a training process that's taking too long. Here's how to fix it:

## Step 1: Kill the Blocked Server
```bash
# Find the uvicorn process
ps aux | grep uvicorn

# Kill it (replace XXXX with the actual process ID)
kill 86656

# Or kill all uvicorn processes
pkill -f uvicorn
```

## Step 2: Use Fast Offline Training
```bash
# Make sure you're in the project directory
cd /Users/itay/Desktop/drawings

# Activate the virtual environment (CRITICAL)
source venv/bin/activate

# Verify you're using the right Python
which python
# Should show: /Users/itay/Desktop/drawings/venv/bin/python

# Run offline training (much faster)
python train_models_offline.py
```

## Step 3: Start Server After Training
```bash
# After offline training completes, start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Why This Works Better
- **No server blocking**: Training runs directly against database
- **3-5x faster**: Optimized for 50 epochs instead of 100
- **Better progress tracking**: Real-time updates
- **No timeouts**: No HTTP request limitations

## If Offline Training Fails
If you get import errors, make sure:
1. You're in the correct directory: `/Users/itay/Desktop/drawings`
2. Virtual environment is activated: `source venv/bin/activate`
3. Dependencies are installed: `pip install -r requirements.txt`

## Alternative: Train via Direct API Calls
If offline training doesn't work, you can train one age group at a time:

```bash
# Start fresh server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Train each age group separately (in different terminal)
curl -X POST "http://localhost:8000/api/v1/models/train" \
     -H "Content-Type: application/json" \
     -d '{"age_min": 3.0, "age_max": 6.0, "min_samples": 10}'

# Wait for completion, then train next group
curl -X POST "http://localhost:8000/api/v1/models/train" \
     -H "Content-Type: application/json" \
     -d '{"age_min": 6.0, "age_max": 9.0, "min_samples": 10}'

curl -X POST "http://localhost:8000/api/v1/models/train" \
     -H "Content-Type: application/json" \
     -d '{"age_min": 9.0, "age_max": 12.0, "min_samples": 10}'
```