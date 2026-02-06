import requests
import sys

BASE_URL = "http://127.0.0.1:8000"

def verify():
    print(f"Connecting to {BASE_URL}...")
    
    # 1. Login
    try:
        resp = requests.post(f"{BASE_URL}/auth/login", json={"username": "admin", "password": "admin"})
        if resp.status_code != 200:
            print(f"Login failed: {resp.status_code} {resp.text}")
            sys.exit(1)
        
        token = resp.json()["access_token"]
        print("Login successful. Token acquired.")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {token}"}

    # 2. Verify New Endpoints
    endpoints = [
        "/execution/tca",
        "/risk/var",
        "/models/explainability"
    ]

    for ep in endpoints:
        print(f"Checking {ep}...", end=" ")
        resp = requests.get(f"{BASE_URL}{ep}", headers=headers)
        if resp.status_code == 200:
            print("OK")
            # print(resp.json()) # Debug
        else:
            print(f"FAILED: {resp.status_code}")
            sys.exit(1)

    print("\nAll endpoints verified successfully.")

if __name__ == "__main__":
    verify()
