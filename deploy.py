#!/usr/bin/env python3
"""
Auto-deploy script for PythonAnywhere
Run this locally to push changes to your server
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and show output"""
    print(f"\nRunning {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"SUCCESS: {description} completed")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"ERROR: {description} failed:")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def deploy_to_pythonanywhere():
    """Deploy backend to PythonAnywhere"""
    
    print("Deploying AiBeatzbyJyntzu Backend to PythonAnywhere")
    print("=" * 60)
    
    # Step 1: Commit and push to GitHub
    if not run_command("git add .", "Adding files to git"):
        return False
    
    commit_msg = input("Enter commit message (or press Enter for default): ").strip()
    if not commit_msg:
        commit_msg = "Update backend code"
    
    if not run_command(f'git commit -m "{commit_msg}"', "Committing changes"):
        print("INFO: No changes to commit or commit failed")
    
    if not run_command("git push origin main", "Pushing to GitHub"):
        return False
    
    # Step 2: SSH commands for PythonAnywhere
    ssh_commands = [
        "cd /home/Jyntzu",
        "rm -rf aibeatzbyJyntzu-temp",
        "git clone https://github.com/jintsu/aibeatzbyJyntzu.git aibeatzbyJyntzu-temp",
        "cp aibeatzbyJyntzu-temp/src/ai/backend.py /home/Jyntzu/mysite/flask_app.py",
        "cp -r aibeatzbyJyntzu-temp/exports/* /home/Jyntzu/exports/ 2>/dev/null || true",
        "rm -rf aibeatzbyJyntzu-temp",
        "touch /var/www/jyntzu_pythonanywhere_com_wsgi.py"
    ]
    
    print("\nRun these commands on PythonAnywhere console:")
    print("-" * 50)
    for cmd in ssh_commands:
        print(f"  {cmd}")
    print("-" * 50)
    
    print("\nOr use this one-liner:")
    one_liner = " && ".join(ssh_commands)
    print(f"  {one_liner}")
    
    print("\nDeployment commands ready!")
    print("Go to PythonAnywhere console and run the commands above")
    print("Your backend will be updated automatically")

if __name__ == "__main__":
    deploy_to_pythonanywhere()