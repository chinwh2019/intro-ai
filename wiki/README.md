# Wiki Pages for GitHub Wiki

This directory contains markdown files to be uploaded to the GitHub Wiki.

## How to Set Up the Wiki

### Step 1: Enable Wiki on GitHub

1. Go to your repository: https://github.com/chinwh2019/intro-ai
2. Click **Settings** tab
3. Scroll to **Features** section
4. Check ✅ **Wikis** checkbox

### Step 2: Create Wiki Pages

**Option A: Via GitHub Web Interface**

1. Click **Wiki** tab (appears after enabling)
2. Click **Create the first page**
3. Copy content from `wiki/Home.md`
4. Paste and save

**Then for each additional page:**
1. Click **New Page** button
2. Name it exactly as linked (e.g., "Getting-Started")
3. Copy content from corresponding `.md` file
4. Paste and save

**Option B: Clone Wiki Locally**

```bash
# Clone the wiki (separate from main repo)
git clone https://github.com/chinwh2019/intro-ai.wiki.git

# Copy all wiki markdown files
cp wiki/*.md intro-ai.wiki/

# Commit and push
cd intro-ai.wiki
git add .
git commit -m "Add initial wiki pages"
git push
```

## Wiki Pages Created

### Core Pages
- **Home.md** - Wiki landing page with navigation
- **Getting-Started.md** - Installation and first steps

### Search Algorithms
- **Search-Algorithms-Overview.md** - Introduction to search
- **BFS-Tutorial.md** - Breadth-First Search explained
- **A-Star-Explained.md** - A* algorithm deep dive

### MDP
- **MDP-Overview.md** - Markov Decision Processes introduction

### Reinforcement Learning
- **Reinforcement-Learning-Overview.md** - Q-Learning and RL concepts

### Support
- **Troubleshooting.md** - Common problems and solutions

## Suggested Additional Pages

Create these as you teach (or assign to students):

### Search (Detailed)
- DFS-Tutorial.md - Depth-First Search
- UCS-Tutorial.md - Uniform Cost Search
- Greedy-Tutorial.md - Greedy Best-First
- Heuristic-Design.md - Creating good heuristics
- Algorithm-Comparison.md - Side-by-side comparison

### MDP (Detailed)
- MDP-Fundamentals.md - Deep dive into MDP components
- Bellman-Equations.md - Math explained simply
- Value-Iteration-Tutorial.md - Algorithm walkthrough
- Policy-Iteration.md - Alternative algorithm
- Discount-Factor-Effects.md - Understanding γ

### RL (Detailed)
- Q-Learning-Deep-Dive.md - Algorithm internals
- Epsilon-Greedy-Explained.md - Exploration strategy
- Reward-Shaping.md - Designing good rewards
- RL-Hyperparameters.md - Tuning guide
- TD-Learning.md - Temporal difference methods

### Code Walkthroughs
- BFS-Code-Walkthrough.md - Line-by-line BFS
- A-Star-Code-Walkthrough.md - Line-by-line A*
- Q-Learning-Code-Walkthrough.md - Line-by-line Q-learning

### Exercises
- Search-Exercises.md - Practice problems
- MDP-Exercises.md - Practice problems
- RL-Exercises.md - Practice problems
- Challenge-Problems.md - Advanced challenges

### Reference
- FAQ.md - Frequently asked questions
- Glossary.md - AI terms defined
- Resources.md - External links and readings

## Updating Wiki Pages

### Method 1: Edit on GitHub

1. Go to wiki page
2. Click **Edit** button
3. Make changes
4. Click **Save Page**

### Method 2: Edit Locally

```bash
cd intro-ai.wiki
# Edit .md files
git add .
git commit -m "Update wiki pages"
git push
```

## Wiki Best Practices

1. **Use consistent naming:**
   - File: `Getting-Started.md`
   - Link: `[Getting Started](Getting-Started)`
   - GitHub auto-converts spaces to hyphens

2. **Link extensively:**
   - Cross-link related topics
   - Link to code files with line numbers
   - Link to web demos

3. **Include visuals:**
   - ASCII art diagrams
   - Screenshots (upload to wiki)
   - Flowcharts

4. **Keep it student-friendly:**
   - Plain English
   - Examples before theory
   - "Try it" sections
   - Visual explanations

5. **Update as you teach:**
   - Note confusing points students ask about
   - Add examples that work well
   - Expand based on feedback

## Student Contributions

**Encourage students to:**
- Fix typos (easy first contribution)
- Add examples that helped them
- Document their experiments
- Create tutorial pages

**How:**
1. Students can edit wiki pages directly (if you give permission)
2. Or submit suggestions via Issues
3. Or fork, edit, and create pull request

## Maintenance

### Regular Updates

**Each semester:**
- Update for any code changes
- Add new examples from class
- Fix errors students find
- Expand based on questions

**Track what works:**
- Which pages students use most
- Which explanations are clearest
- What's missing

### Version Control

Wiki has its own git history:
```bash
cd intro-ai.wiki
git log  # See all changes
git diff HEAD~1  # See recent changes
```

## Navigation Tips

**Wiki sidebar automatically shows:**
- All pages alphabetically
- Custom ordering possible via _Sidebar.md

**Create _Sidebar.md for custom navigation:**
```markdown
**Start Here**
- [Home](Home)
- [Getting Started](Getting-Started)

**Algorithms**
- [Search](Search-Algorithms-Overview)
- [MDP](MDP-Overview)
- [RL](Reinforcement-Learning-Overview)

**Support**
- [Troubleshooting](Troubleshooting)
- [FAQ](FAQ)
```

## Success Metrics

**Wiki is successful if:**
- ✅ Students find answers before asking
- ✅ Common questions documented
- ✅ Theory accessible without instructor
- ✅ Students contribute content
- ✅ Used across multiple semesters

**Continuously improve based on:**
- Student questions (add to FAQ)
- Confusion points (create tutorial)
- Great experiments (document as example)

---

**Your wiki is ready to deploy!** 🚀

**Next step:** Enable Wiki on GitHub and start creating pages.
