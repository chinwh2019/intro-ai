#!/bin/bash
# Automated web deployment script for Search Module
# Converts Python/Pygame to WebAssembly using Pygbag

set -e  # Exit on error

echo "🚀 Starting Web Deployment for Search Module"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo ""

# Configuration
MODULE="search"
WEB_DIR="web/$MODULE"
BUILD_DIR="$WEB_DIR/build/web"
PROJECT_ROOT="$(pwd)"

# Step 1: Verify environment
echo "1️⃣  Checking environment..."

command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 not installed"; exit 1; }
echo "   ✓ Python 3 installed"

command -v pygbag >/dev/null 2>&1 || {
    echo "❌ Pygbag not installed"
    echo "   Install with: pip install pygbag"
    exit 1
}
echo "   ✓ Pygbag installed"

if [ ! -d "modules/$MODULE" ]; then
    echo "❌ Module not found: modules/$MODULE"
    exit 1
fi
echo "   ✓ Search module found"

echo ""

# Step 2: Verify web files exist
echo "2️⃣  Verifying web files..."

if [ ! -f "$WEB_DIR/main.py" ]; then
    echo "❌ Web main.py not found at $WEB_DIR/main.py"
    echo "   Make sure you've created the web-compatible version"
    exit 1
fi
echo "   ✓ Web main.py exists"

if [ ! -d "$WEB_DIR/core" ] || [ ! -d "$WEB_DIR/algorithms" ]; then
    echo "❌ Web module files not found"
    echo "   Make sure files are copied to $WEB_DIR"
    exit 1
fi
echo "   ✓ Module files exist"

echo ""

# Step 3: Build with pygbag
echo "3️⃣  Building with Pygbag..."
echo "   This may take 1-2 minutes..."
echo ""

cd "$WEB_DIR"

# Build (suppress verbose output, show errors)
if pygbag --build main.py 2>&1 | grep -i "error\|traceback" && [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "❌ Build failed! Check errors above"
    cd "$PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT"

echo ""
echo "   ✓ Build successful"
echo ""

# Step 4: Verify build output
echo "4️⃣  Verifying build output..."

if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ Build directory not found: $BUILD_DIR"
    exit 1
fi
echo "   ✓ Build directory exists"

if [ ! -f "$BUILD_DIR/index.html" ]; then
    echo "❌ index.html not generated"
    exit 1
fi
echo "   ✓ index.html generated"

# Count build files
FILE_COUNT=$(find "$BUILD_DIR" -type f | wc -l)
echo "   ✓ Generated $FILE_COUNT files"

echo ""

# Step 5: Success summary
echo "✅ Deployment Build Complete!"
echo ""
echo "📂 Build location: $BUILD_DIR"
echo ""
echo "🧪 To test locally:"
echo "   cd $BUILD_DIR"
echo "   python3 -m http.server 8000"
echo "   Open: http://localhost:8000"
echo ""
echo "🌐 To deploy to GitHub Pages:"
echo "   1. git checkout gh-pages"
echo "   2. mkdir -p search"
echo "   3. cp -r $BUILD_DIR/* search/"
echo "   4. git add ."
echo "   5. git commit -m 'Deploy search module'"
echo "   6. git push origin gh-pages"
echo ""
echo "🎉 Your search module is ready for the web!"
echo ""
