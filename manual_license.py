#!/usr/bin/env python3
"""
Manual license creation for testing
"""

import json
import uuid
import hashlib
from datetime import datetime

def generate_username():
    return f"user_{hash(str(datetime.now()))%99999:05d}"

def generate_password():
    return hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]

def generate_license_key():
    return f"AIBZ-{uuid.uuid4().hex[:8].upper()}-{uuid.uuid4().hex[:8].upper()}"

# Create test license
license_data = {
    'license_key': generate_license_key(),
    'username': generate_username(),
    'password': generate_password(),
    'type': 'lifetime',
    'created': datetime.now().isoformat(),
    'expires': None,
    'banned': False,
    'download_count': 0,
    'last_login': None
}

print("Test License Created:")
print(f"Username: {license_data['username']}")
print(f"Password: {license_data['password']}")
print(f"License Key: {license_data['license_key']}")
print(f"Type: {license_data['type']}")

# Test login with this data
import requests

try:
    response = requests.post("https://jyntzu.pythonanywhere.com/login", json={
        "username": license_data['username'],
        "password": license_data['password']
    })
    print(f"\nLogin Test: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Login Test Error: {e}")