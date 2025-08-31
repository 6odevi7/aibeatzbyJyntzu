#!/bin/bash
# PythonAnywhere Auto-Update Script
# Save this in your PythonAnywhere files and run when you want to update

echo "🚀 Updating AiBeatzbyJyntzu Backend..."

# Navigate to home directory
cd /home/Jyntzu

# Remove old temp directory
rm -rf aibeatzbyJyntzu-temp

# Clone latest from GitHub
echo "📥 Downloading latest code from GitHub..."
git clone https://github.com/jintsu/aibeatzbyJyntzu.git aibeatzbyJyntzu-temp

# Copy backend file
echo "📋 Updating backend code..."
cp aibeatzbyJyntzu-temp/src/ai/backend.py /home/Jyntzu/mysite/flask_app.py

# Copy exports directory (create if doesn't exist)
mkdir -p /home/Jyntzu/exports
cp -r aibeatzbyJyntzu-temp/exports/* /home/Jyntzu/exports/ 2>/dev/null || true

# Copy requirements if exists
if [ -f "aibeatzbyJyntzu-temp/requirements.txt" ]; then
    cp aibeatzbyJyntzu-temp/requirements.txt /home/Jyntzu/
    echo "📦 Installing requirements..."
    pip3.10 install --user -r requirements.txt
fi

# Clean up
rm -rf aibeatzbyJyntzu-temp

# Reload web app
echo "🔄 Reloading web app..."
touch /var/www/jyntzu_pythonanywhere_com_wsgi.py

echo "✅ Backend updated successfully!"
echo "🌐 Your API is now running the latest code"