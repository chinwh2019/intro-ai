# Web Deployment - Quick Start Guide

**Status:** ✅ Implementation Complete - Ready to Deploy!
**Time to Deploy:** 5 minutes
**Result:** Students access search module in ANY browser

---

## ✅ What's Already Done

- ✅ Pygbag installed (v0.9.2)
- ✅ Web-compatible code created (`web/search/`)
- ✅ Production build generated (`web/search/build/web/`)
- ✅ Deployment scripts ready
- ✅ GitHub Actions configured
- ✅ Landing page created
- ✅ Student guide written

**Everything is ready - just deploy!**

---

## 🚀 Deploy in 5 Minutes

### Step 1: Go to gh-pages Branch (30 seconds)

```bash
git checkout --orphan gh-pages
```

### Step 2: Clean the Branch (30 seconds)

```bash
# Remove everything
git rm -rf .

# Verify clean
ls -la
# Should only show: .git/ and maybe .claude/
```

### Step 3: Copy Web Files (1 minute)

```bash
# Copy landing page
cp web/index.html .

# Copy search module build
mkdir -p search
cp -r web/search/build/web/* search/

# Create gitignore
cat > .gitignore << 'EOF'
.DS_Store
__pycache__/
.claude/
EOF

# Verify
ls -l
# Should show: index.html, search/, .gitignore
```

### Step 4: Commit and Push (1 minute)

```bash
# Stage all
git add .

# Commit
git commit -m "Deploy search module to GitHub Pages"

# Push
git push origin gh-pages
```

### Step 5: Enable GitHub Pages (2 minutes)

1. Go to: `https://github.com/YOUR-USERNAME/intro-ai`
2. Click **Settings** tab
3. Scroll to **Pages** section (left sidebar)
4. Under **Source**:
   - Branch: `gh-pages`
   - Folder: `/ (root)`
5. Click **Save**

**Wait 1-2 minutes...**

### Step 6: Test! (30 seconds)

Visit: `https://YOUR-USERNAME.github.io/intro-ai/`

You should see:
- ✅ Beautiful landing page
- ✅ "Search Algorithms" card
- ✅ Click "Launch Module" → app loads
- ✅ Press 1-5 to select algorithms
- ✅ Everything works!

---

## 🎉 Success!

Your search module is now accessible to:
- ✅ Students with Chromebooks
- ✅ Students without Python
- ✅ Students on locked computers
- ✅ Anyone with a web browser

**Share this URL with students:**
```
https://YOUR-USERNAME.github.io/intro-ai/search/
```

---

## 📱 Tell Your Students

**"Visit this link - no installation needed:"**
```
https://YOUR-USERNAME.github.io/intro-ai/
```

**Controls:**
- Press 1-5 to select algorithm
- Press SPACE to pause/resume
- Press R to reset maze
- Press T for random start/goal

**First load:** 10-20 seconds (downloading runtime)
**After that:** 2-3 seconds (cached)

---

## 🔄 Future Updates

When you improve the module:

**Option A: Automated (Recommended)**
```bash
# 1. Make changes to modules/search/
# 2. Commit and push to main
# 3. GitHub Actions auto-deploys!
# 4. Students see update in 2-3 minutes
```

**Option B: Manual**
```bash
# 1. Rebuild
cd web/search
pygbag --build main.py

# 2. Update gh-pages
git checkout gh-pages
cp -r build/web/* search/
git commit -am "Update search module"
git push origin gh-pages
```

---

## ❓ Troubleshooting

**GitHub Pages shows 404?**
- Wait 2-5 minutes (deployment in progress)
- Check Settings → Pages → Branch is gh-pages
- Verify `index.html` in gh-pages root

**App doesn't load?**
- Check browser console (F12)
- Try different browser (Chrome recommended)
- Verify build files exist: `ls web/search/build/web/`

**Want to test before deploying?**
```bash
cd web/search/build/web
python3 -m http.server 8000
# Open: http://localhost:8000
```

---

## 📚 Full Documentation

For complete details, see:
- `web/DEPLOY_TO_GITHUB_PAGES.md` - Detailed deployment guide
- `web/STUDENT_ACCESS_GUIDE.md` - Student instructions
- `docs/PYGBAG_WEB_DEPLOYMENT_PLAN.md` - Technical details

---

**Ready? Deploy now and transform your class!** 🎓

**Time investment:** 5 minutes
**Student benefit:** Immediate browser access
**Cost:** $0
**Impact:** Game-changing! 🚀
