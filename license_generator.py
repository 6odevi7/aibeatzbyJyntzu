#!/usr/bin/env python3
"""
AiBeatzbyJyntzu License Generator
Generate licenses and download links for customers
"""

import requests
import json
import sys

API_BASE = "https://jyntzu.pythonanywhere.com"

def create_license(license_type="lifetime", custom_username=None, custom_password=None):
    """Create a new license"""
    data = {
        "type": license_type,
        "username": custom_username,
        "password": custom_password
    }
    
    try:
        response = requests.post(f"{API_BASE}/licenses/create", json=data)
        result = response.json()
        
        if result["status"] == "success":
            license_data = result["license"]
            print(f"\n✅ {license_type.title()} License Created Successfully!")
            print(f"📧 Username: {license_data['username']}")
            print(f"🔑 Password: {license_data['password']}")
            print(f"🎫 License Key: {license_data['license_key']}")
            print(f"📅 Created: {license_data['created']}")
            if license_data.get('expires'):
                print(f"⏰ Expires: {license_data['expires']}")
            
            return license_data
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None

def generate_download_link(license_key):
    """Generate download link for a license"""
    data = {"license_key": license_key}
    
    try:
        response = requests.post(f"{API_BASE}/licenses/generate_download", json=data)
        result = response.json()
        
        if result["status"] == "success":
            print(f"\n📥 Download Link Generated!")
            print(f"🔗 URL: {result['download_url']}")
            print(f"⏰ Expires: {result['expires']}")
            print(f"👤 Username: {result['username']}")
            print(f"🔑 Password: {result['password']}")
            
            return result
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None

def list_licenses():
    """List all licenses"""
    try:
        response = requests.get(f"{API_BASE}/licenses")
        result = response.json()
        
        licenses = result.get("licenses", [])
        
        if not licenses:
            print("📋 No licenses found")
            return
            
        print(f"\n📋 Total Licenses: {len(licenses)}")
        print("-" * 80)
        
        for license_data in licenses:
            status = "🚫 BANNED" if license_data.get('banned') else "✅ ACTIVE"
            expires = license_data.get('expires', 'Never')
            
            print(f"🎫 {license_data['license_key']}")
            print(f"   👤 {license_data['username']} | 🔑 {license_data['password']}")
            print(f"   📊 {status} | 📅 {license_data['type']} | ⏰ Expires: {expires}")
            print(f"   📥 Downloads: {license_data.get('download_count', 0)}")
            print("-" * 80)
            
    except Exception as e:
        print(f"❌ Connection error: {e}")

def main():
    print("🎵 AiBeatzbyJyntzu License Generator")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python license_generator.py create [type] [username] [password]")
        print("  python license_generator.py download <license_key>")
        print("  python license_generator.py list")
        print("")
        print("License Types: lifetime, monthly, yearly, weekly")
        print("")
        print("Examples:")
        print("  python license_generator.py create lifetime")
        print("  python license_generator.py create monthly john_doe mypassword123")
        print("  python license_generator.py download AIBZ-12345678-87654321")
        print("  python license_generator.py list")
        return
    
    command = sys.argv[1].lower()
    
    if command == "create":
        license_type = sys.argv[2] if len(sys.argv) > 2 else "lifetime"
        username = sys.argv[3] if len(sys.argv) > 3 else None
        password = sys.argv[4] if len(sys.argv) > 4 else None
        
        license_data = create_license(license_type, username, password)
        
        if license_data:
            print(f"\n💡 To generate download link:")
            print(f"python license_generator.py download {license_data['license_key']}")
    
    elif command == "download":
        if len(sys.argv) < 3:
            print("❌ Please provide license key")
            return
            
        license_key = sys.argv[2]
        generate_download_link(license_key)
    
    elif command == "list":
        list_licenses()
    
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()