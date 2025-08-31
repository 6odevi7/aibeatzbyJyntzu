#!/usr/bin/env python3
import requests

API_BASE = "https://jyntzu.pythonanywhere.com"

def test_endpoints():
    print("Testing API endpoints...")
    
    # Test health
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Health: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Health error: {e}")
    
    # Test licenses endpoint
    try:
        response = requests.get(f"{API_BASE}/licenses")
        print(f"Licenses: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Licenses error: {e}")
    
    # Test create license
    try:
        response = requests.post(f"{API_BASE}/licenses/create", json={"type": "lifetime"})
        print(f"Create License: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Create License error: {e}")

if __name__ == "__main__":
    test_endpoints()