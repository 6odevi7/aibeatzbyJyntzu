@echo off
echo Updating AiBeatzbyJyntzu...

echo 1. Rebuilding app...
npm run build

echo 2. Pushing to GitHub...
git add .
git commit -m "App update - %date% %time%"
git push origin main

echo 3. Update complete!
echo Run this on PythonAnywhere to update backend:
echo cd /home/Jyntzu ^&^& rm -rf temp ^&^& git clone https://github.com/6odevi7/aibeatzbyJyntzu.git temp ^&^& cp temp/src/ai/backend.py /home/Jyntzu/mysite/flask_app.py ^&^& rm -rf temp ^&^& touch /var/www/jyntzu_pythonanywhere_com_wsgi.py

pause