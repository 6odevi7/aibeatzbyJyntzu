@echo off
echo Deploying AiBeatzbyJyntzu Backend...
echo =====================================

echo Adding files to git...
git add .

echo Committing changes...
git commit -m "Update backend code"

echo Pushing to GitHub...
git push origin main

echo.
echo SUCCESS! Code pushed to GitHub
echo.
echo Now run this command on PythonAnywhere console:
echo.
echo cd /home/Jyntzu ^&^& rm -rf temp ^&^& git clone https://github.com/6odevi7/aibeatzbyJyntzu.git temp ^&^& cp temp/src/ai/backend.py /home/Jyntzu/mysite/flask_app.py ^&^& rm -rf temp ^&^& touch /var/www/jyntzu_pythonanywhere_com_wsgi.py
echo.
pause