# Deployment Setup Guide

This guide explains how to set up automatic deployment to both Streamlit Cloud and Hugging Face Spaces.

## Prerequisites

- GitHub repository: `yeager620/lobster-lab` ✅
- Streamlit Cloud account (deployed manually) ✅
- Hugging Face account

## Setup Instructions

### 1. Create Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Configure:
   - **Space name**: `lobster-viz` (or your preferred name)
   - **License**: MIT
   - **SDK**: Streamlit
   - **Space hardware**: CPU basic (free)
   - **Visibility**: Public
4. Click "Create Space"
5. Note the space name: `your-username/lobster-viz`

### 2. Get Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `github-actions-deploy`
4. Type: Write access
5. Copy the token (you'll need it in the next step)

### 3. Configure GitHub Secrets

1. Go to your GitHub repo: https://github.com/yeager620/lobster-lab
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add these three secrets:

   **Secret 1: HF_TOKEN**
   - Name: `HF_TOKEN`
   - Value: Your Hugging Face token from step 2

   **Secret 2: HF_SPACE_NAME**
   - Name: `HF_SPACE_NAME`
   - Value: `your-username/lobster-viz` (replace with your actual space name)

   **Secret 3: STREAMLIT_APP_URL**
   - Name: `STREAMLIT_APP_URL`
   - Value: Your Streamlit app URL (e.g., `https://lobster-lab.streamlit.app`)

### 4. Commit and Push

```bash
# Add all deployment files
git add .
git commit -m "Add automated deployment workflow"
git push origin main
```

## How It Works

When you push to the `main` branch, the GitHub Action will:

1. ✅ **Streamlit Cloud**: Automatically redeploys (it monitors the GitHub repo)
2. 🤗 **Hugging Face Spaces**: Pushes code to your HF Space repository

## Manual Deployment

You can also trigger deployment manually:

1. Go to **Actions** tab in your GitHub repo
2. Select **Deploy to Streamlit and Hugging Face**
3. Click **Run workflow** → **Run workflow**

## Monitoring Deployments

- **GitHub Actions**: https://github.com/yeager620/lobster-lab/actions
- **Streamlit Cloud**: https://share.streamlit.io/
- **Hugging Face Spaces**: https://huggingface.co/spaces/your-username/lobster-viz

## Troubleshooting

### HF Space deployment fails
- Check that your `HF_TOKEN` has write access
- Verify the space name in `HF_SPACE_NAME` is correct (format: `username/space-name`)

### Streamlit Cloud not updating
- Check that Streamlit Cloud is connected to the correct GitHub repo
- Verify the app is watching the `main` branch

### Python version issues
- Both platforms use Python 3.13 (specified in workflow)
- If you need a different version, update `.github/workflows/deploy.yml`

## Files Created

- `.github/workflows/deploy.yml` - Deployment workflow
- `.streamlit/config.toml` - Streamlit configuration
- `requirements.txt` - Python dependencies
- `packages.txt` - System packages (empty for this project)
- `README_HF.md` - Hugging Face Space README
- `.streamlit/secrets.toml.example` - Example secrets file

## Next Steps

After setup:
1. Push to main → automatic deployment ✨
2. Check both platforms to confirm deployment
3. Share your apps!
   - Streamlit: `https://your-app.streamlit.app`
   - Hugging Face: `https://huggingface.co/spaces/your-username/lobster-viz`
