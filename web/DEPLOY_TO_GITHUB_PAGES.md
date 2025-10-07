# Deploy to GitHub Pages - Quick Guide

Follow these steps to make your search module accessible to students at a URL.

---

## Prerequisites

- [x] Search module web build complete
- [x] Build files in `web/search/build/web/`
- [x] GitHub repository exists
- [x] Repository is public

---

## Step-by-Step Deployment

### Step 1: Create gh-pages Branch

```bash
# Save your current work
git add .
git commit -m "Save current work before gh-pages"

# Create new orphan branch (clean slate)
git checkout --orphan gh-pages

# Remove all files from staging
git rm -rf .

# Clean working directory
rm -rf modules/ mdp/ reinforcement_learning/ search/ scripts/
rm README.md LICENSE requirements.txt run_search.py .gitignore
# Keep only: .git/, web/, .github/
```

### Step 2: Copy Build Files

```bash
# Copy landing page
cp web/index.html .

# Copy search module build
mkdir -p search
cp -r web/search/build/web/* search/

# Copy student guide
cp web/STUDENT_ACCESS_GUIDE.md .

# Verify structure
ls -l
# Should see:
#   index.html
#   search/
#   STUDENT_ACCESS_GUIDE.md
```

### Step 3: Create .gitignore for gh-pages

```bash
cat > .gitignore << 'EOF'
.DS_Store
__pycache__/
*.pyc
.claude/
EOF
```

### Step 4: Commit and Push

```bash
# Stage all files
git add .

# Check what will be committed
git status

# Commit
git commit -m "Initial GitHub Pages deployment with search module"

# Push to GitHub
git push origin gh-pages
```

### Step 5: Enable GitHub Pages

1. Go to your GitHub repository: `https://github.com/YOUR-USERNAME/intro-ai`
2. Click **Settings** tab
3. Scroll to **Pages** section (left sidebar)
4. Under **Source**:
   - Branch: `gh-pages`
   - Folder: `/ (root)`
5. Click **Save**

**GitHub will show:**
```
✓ Your site is live at https://YOUR-USERNAME.github.io/intro-ai/
```

### Step 6: Wait and Verify

**Wait:** 1-2 minutes for deployment

**Then visit:**
```
https://YOUR-USERNAME.github.io/intro-ai/
```

**Should see:** Landing page with "Search Algorithms" module

**Click:** "Launch Module" button

**Should see:** Search module loads in ~10-20 seconds

---

## Testing Checklist

After deployment, test:

- [ ] Landing page loads
- [ ] Click "Launch Module" goes to search/
- [ ] Search module loads (shows progress bar)
- [ ] App renders (shows maze)
- [ ] Pressing 1-5 selects algorithms
- [ ] Algorithms run and find paths
- [ ] All controls work (SPACE, R, T, S)
- [ ] Statistics update correctly
- [ ] No console errors (F12 to check)

---

## Updating Deployment

When you improve the module:

```bash
# 1. Make changes to modules/search/
# (edit code)

# 2. Test locally
python run_search.py

# 3. Copy to web/search/
cp -r modules/search/{config.py,core,algorithms,ui,heuristics.py} web/search/

# 4. Fix imports
cd web/search
find . -name "*.py" -exec sed -i.bak 's/from modules\.search\./from /g' {} \;
rm -f **/*.bak

# 5. Rebuild
pygbag --build main.py

# 6. Update gh-pages
git checkout gh-pages
cp -r web/search/build/web/* search/
git add .
git commit -m "Update search module"
git push origin gh-pages

# 7. Verify
# Visit URL in 1-2 minutes
```

---

## Automated Updates (If Using GitHub Actions)

If you set up GitHub Actions (`.github/workflows/deploy-search-module.yml`):

```bash
# Just push to main - auto-deploys!
git checkout main
# (make changes to modules/search/)
git add .
git commit -m "Improve search module"
git push origin main

# GitHub Actions will:
# 1. Detect changes
# 2. Build with pygbag
# 3. Deploy to gh-pages
# 4. Update in 2-3 minutes
```

---

## Troubleshooting Deployment

### GitHub Pages Shows 404

**Causes:**
1. Branch not selected in settings
2. Files not in correct location
3. Deployment still processing

**Fixes:**
1. Double-check Settings → Pages → Branch: gh-pages
2. Verify `index.html` in root of gh-pages branch
3. Wait 5 minutes and try again

### Search Module Not Found

**Check:**
```bash
git checkout gh-pages
ls search/
# Should show: index.html, search.apk, favicon.png
```

If missing:
```bash
cp -r web/search/build/web/* search/
git add search/
git commit -m "Add search module files"
git push origin gh-pages
```

### Build Files Missing

If `web/search/build/web/` is empty:

```bash
cd web/search
pygbag --build main.py
# Wait for build to complete
ls build/web/
```

---

## Custom Domain (Optional)

If you want `yourschool.edu/intro-ai` instead of `username.github.io/intro-ai`:

1. Add CNAME file to gh-pages root:
```bash
git checkout gh-pages
echo "ai.yourschool.edu" > CNAME
git add CNAME
git commit -m "Add custom domain"
git push origin gh-pages
```

2. Configure DNS at your domain registrar:
```
Type: CNAME
Name: ai
Value: YOUR-USERNAME.github.io
```

3. Enable in GitHub Settings → Pages → Custom domain

---

## Rollback to Previous Version

If new deployment breaks:

```bash
# Find last working commit
git checkout gh-pages
git log --oneline

# Revert to previous commit
git revert HEAD

# Or reset to specific commit
git reset --hard abc123

# Push
git push origin gh-pages --force
```

---

## Multiple Modules

When deploying additional modules:

```bash
# Build each module
cd web/mdp
pygbag --build main.py

cd ../rl
pygbag --build main.py

# Deploy all
git checkout gh-pages
cp -r web/mdp/build/web/* mdp/
cp -r web/rl/build/web/* rl/
git add .
git commit -m "Deploy MDP and RL modules"
git push origin gh-pages
```

**URLs:**
```
https://username.github.io/intro-ai/         (landing)
https://username.github.io/intro-ai/search/ (search)
https://username.github.io/intro-ai/mdp/    (mdp)
https://username.github.io/intro-ai/rl/     (rl)
```

---

## Status

**Current:**
- ✅ Search module built and ready for deployment
- ✅ Landing page created
- ✅ Student guide created
- ✅ Deployment scripts created
- ✅ GitHub Actions workflow configured

**To deploy:**
Follow steps 1-6 above (15 minutes)

**Result:**
Students can access via browser with zero installation!

---

**Last Updated:** October 7, 2025
**Build Tool:** Pygbag 0.9.2
**Python Version:** 3.11
**Pygame Version:** Included in Pygbag
