# GitHub Actions Setup Guide

**Problem:** GitHub Actions deployment fails with "Permission denied to github-actions[bot]"
**Solution:** Configure workflow permissions correctly

---

## The Issue

When GitHub Actions tries to deploy to `gh-pages` branch, you get:

```
remote: Permission to yourname/intro-ai.git denied to github-actions[bot].
fatal: unable to access 'https://github.com/...': The requested URL returned error: 403
Error: Action failed with "The process '/usr/bin/git' failed with exit code 128"
```

**Why?** GitHub changed default GITHUB_TOKEN permissions to **read-only** for security.

---

## Solution 1: Add Permissions to Workflow (DONE! ✅)

**File:** `.github/workflows/deploy-search-module.yml`

**Fixed workflow:**
```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest

    # ADD THIS BLOCK:
    permissions:
      contents: write  # Required to push to gh-pages branch

    steps:
      # ... rest of workflow
```

**This is already fixed in your workflow!** ✅

---

## Solution 2: Check Repository Settings

**If still getting errors after fix above:**

### Step 1: Go to Repository Settings

1. Go to: `https://github.com/YOUR-USERNAME/intro-ai`
2. Click **Settings** tab
3. Click **Actions** in left sidebar
4. Click **General**

### Step 2: Check Workflow Permissions

Scroll to **Workflow permissions** section at bottom.

**Option A: Recommended (More Secure)**
- Select: ☑ Read repository contents and packages permissions
- This is default - keeps using `permissions:` blocks in workflows
- **Our workflow has `permissions: contents: write` so this works!**

**Option B: Permissive (Less Secure)**
- Select: ☑ Read and write permissions
- Makes ALL workflows have write access
- Not recommended (security risk)

**Recommendation:** Use Option A (default) + explicit `permissions:` in workflow

### Step 3: Save Settings

Click **Save** if you changed anything.

---

## Solution 3: Enable GitHub Pages

Even with permissions fixed, you need GitHub Pages enabled:

### Step 1: Enable GitHub Pages

1. Go to: Repository → **Settings**
2. Scroll to **Pages** section (left sidebar)
3. Under **Source**:
   - Branch: `gh-pages`
   - Folder: `/ (root)`
4. Click **Save**

### Step 2: Wait for Initial Deploy

- First deployment creates `gh-pages` branch
- Subsequent deployments update it
- Allow 2-3 minutes for each deployment

---

## Verification

### Check if Workflow is Working

1. Go to: Repository → **Actions** tab
2. Find your workflow run
3. Check status:
   - ✅ Green checkmark = Success!
   - ❌ Red X = Failed (click for details)
   - 🟡 Yellow dot = Running

### Check Deployment

1. Go to: Repository → **Settings** → **Pages**
2. Should show: "Your site is published at https://..."
3. Click the URL to test

### Test the App

Visit: `https://YOUR-USERNAME.github.io/intro-ai/search/`

Should:
- ✅ Load within 20 seconds
- ✅ Show pygame app
- ✅ Respond to keyboard (1-5, SPACE, R, T)

---

## Common Issues & Fixes

### Issue 1: 403 Permission Denied

**Cause:** Missing `permissions: contents: write`

**Fix:**
```yaml
jobs:
  deploy:
    permissions:
      contents: write  # Add this
```

**Status:** ✅ Already fixed in your workflow!

---

### Issue 2: Branch 'gh-pages' Doesn't Exist

**Cause:** First deployment hasn't created branch yet

**Fix:**
```bash
# Create branch manually first time
git checkout --orphan gh-pages
git rm -rf .
touch .gitkeep
git add .
git commit -m "Initialize gh-pages"
git push origin gh-pages

# Then Actions will work
```

Or just wait - the action will create it.

---

### Issue 3: Pages Not Enabled

**Cause:** GitHub Pages not configured in settings

**Fix:**
1. Settings → Pages
2. Source: `gh-pages` branch
3. Save

---

### Issue 4: Deployment Succeeds but 404

**Cause:** Files in wrong location

**Fix:** Check `destination_dir` in workflow:
```yaml
- uses: peaceiris/actions-gh-pages@v3
  with:
    publish_dir: ./web/search/build/web
    destination_dir: search  # Files go to /search/
```

Should match URL: `https://...github.io/intro-ai/search/`

---

## Testing Your Workflow

### Trigger Workflow Manually

1. Go to: Actions tab
2. Select "Deploy Search Module to GitHub Pages"
3. Click "Run workflow"
4. Select branch: `main`
5. Click "Run workflow"

**Watch it run:**
- Should complete in 2-3 minutes
- Check each step for errors
- Green checkmarks = success!

### Test with Commit

```bash
# Make a small change
echo "# Test" >> web/search/README.md

# Commit and push to main
git add .
git commit -m "test: trigger deployment"
git push origin main

# Watch Actions tab
# Should trigger automatically
```

---

## Our Workflow Explained

```yaml
name: Deploy Search Module to GitHub Pages

on:
  push:
    branches: [ main ]          # Triggers on push to main
    paths:                       # Only if these files change
      - 'modules/search/**'
      - 'web/search/**'
  workflow_dispatch:            # Also allow manual trigger

jobs:
  deploy:
    runs-on: ubuntu-latest

    permissions:
      contents: write           # ← FIX: Write permission for gh-pages

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install Pygbag
      run: pip install pygbag

    - name: Build with Pygbag
      run: |
        cd web/search
        pygbag --build main.py

    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}  # Auto-provided
        publish_dir: ./web/search/build/web        # What to deploy
        destination_dir: search                    # Where to deploy
```

---

## Quick Checklist

Before deploying, verify:

- [x] Workflow has `permissions: contents: write` ✅
- [ ] Repository Settings → Actions → Workflow permissions configured
- [ ] GitHub Pages enabled in Settings → Pages
- [ ] Workflow file is in `.github/workflows/`
- [ ] Workflow file is valid YAML (no syntax errors)
- [ ] Pushed to main branch (or manually trigger)

---

## Alternative: Manual Deployment (No GitHub Actions)

If you don't want to use GitHub Actions:

```bash
# 1. Build
cd web/search
pygbag --build main.py

# 2. Deploy
git checkout gh-pages
cp -r build/web/* search/
git commit -am "Update search module"
git push origin gh-pages

# 3. Done!
```

---

## Summary

**Problem:** Permission denied (403 error)

**Root Cause:** Default GITHUB_TOKEN is read-only

**Solution:** Add `permissions: contents: write` to workflow

**Status:** ✅ Fixed in `.github/workflows/deploy-search-module.yml`

**Next:** Push to main or manually trigger workflow in Actions tab

**Result:** Automatic deployment to GitHub Pages! 🎉

---

**Last Updated:** October 7, 2025
**Status:** Issue Resolved
**Workflow:** Ready to use
