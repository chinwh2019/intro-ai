# How to Deploy These Wiki Pages to GitHub

Quick guide to get your wiki live on GitHub.

## Step 1: Enable Wiki (30 seconds)

1. Go to: https://github.com/chinwh2019/intro-ai
2. Click **Settings** tab
3. Scroll to **Features** section
4. Check ✅ **Wikis**
5. Click **Save** if needed

## Step 2: Access Your Wiki (10 seconds)

1. Click **Wiki** tab (now visible at top)
2. You'll see "Create the first page"
3. Don't click yet - read below!

## Step 3: Clone Wiki Repository (30 seconds)

GitHub Wiki has its own git repository:

```bash
# Clone the wiki (different from main repo)
git clone https://github.com/chinwh2019/intro-ai.wiki.git

cd intro-ai.wiki
```

## Step 4: Copy Wiki Content (1 minute)

```bash
# Copy all prepared wiki pages
cp ../intro-ai/wiki/*.md .

# Check files copied
ls -la
# Should see: Home.md, Getting-Started.md, etc.
```

## Step 5: Push to Wiki (30 seconds)

```bash
# Add all files
git add .

# Commit
git commit -m "Initial wiki content: tutorials, guides, and troubleshooting"

# Push
git push
```

## Step 6: Verify (10 seconds)

1. Go to GitHub repo
2. Click **Wiki** tab
3. You should see:
   - ✅ Home page with navigation
   - ✅ Sidebar with all pages
   - ✅ Search box working

## Step 7: Polish (Optional, 5 minutes)

### Create Custom Sidebar

**Create `_Sidebar.md` in wiki repo:**

```markdown
**Quick Start**
- [Home](Home)
- [Getting Started](Getting-Started)
- [Troubleshooting](Troubleshooting)

**Search Algorithms**
- [Overview](Search-Algorithms-Overview)
- [BFS Tutorial](BFS-Tutorial)
- [A* Explained](A-Star-Explained)

**MDP**
- [MDP Overview](MDP-Overview)

**Reinforcement Learning**
- [RL Overview](Reinforcement-Learning-Overview)
```

Save and push - now you have custom navigation!

### Create Footer

**Create `_Footer.md`:**

```markdown
[🏠 Home](Home) | [📚 Wiki Index](https://github.com/chinwh2019/intro-ai/wiki/_pages) | [💻 Code Repository](https://github.com/chinwh2019/intro-ai)
```

## Updating Wiki Later

### Method 1: Edit on GitHub (Easiest)

1. Go to wiki page
2. Click **Edit**
3. Make changes
4. Click **Save Page**

### Method 2: Edit Locally (For Bulk Changes)

```bash
cd intro-ai.wiki

# Edit files
vim Home.md  # or use your editor

# Push changes
git add .
git commit -m "Update Home page"
git push
```

### Method 3: From Main Repo

```bash
# Edit in main repo
cd intro-ai
vim wiki/Getting-Started.md

# Copy to wiki repo
cp wiki/*.md ../intro-ai.wiki/

# Push from wiki repo
cd ../intro-ai.wiki
git add .
git commit -m "Update from main repo"
git push
```

## Tips

1. **Keep files in sync:**
   - Maintain master copy in `wiki/` folder
   - Copy to wiki repo when updating
   - Or edit wiki directly and copy back

2. **Use consistent naming:**
   - File: `Getting-Started.md`
   - Link: `[Getting Started](Getting-Started)`
   - Case matters!

3. **Test links:**
   - Click all links after creating pages
   - Fix broken links immediately

4. **Preview before pushing:**
   - Use markdown preview in your editor
   - Or use online markdown viewer

## Student Access

Once wiki is live, students can:

**View:**
- Visit: https://github.com/chinwh2019/intro-ai/wiki
- Browse all pages
- Search for topics

**Contribute (if enabled):**
- Click Edit on any page
- Make changes
- Submit for review

**To enable student edits:**
- Settings → Manage Access → Invite collaborators
- Or: Make repo public and enable wiki edits

## Wiki vs Main Repo

**What goes where:**

**Main Repo (README files):**
- Quick start guide
- Installation instructions
- Basic usage
- File structure

**Wiki:**
- Detailed theory
- Algorithm explanations
- Step-by-step tutorials
- Exercises and assignments
- Troubleshooting
- FAQ

**Rule of thumb:**
- README = quick reference (1-2 pages)
- Wiki = comprehensive guide (unlimited pages)

## Success!

Your wiki is now live at:
```
https://github.com/chinwh2019/intro-ai/wiki
```

Students can access theory, tutorials, and troubleshooting without cloning the repo! 🎉

---

**Total setup time: ~5 minutes**
**Student benefit: Massive!**
**Maintenance: Update as needed**
