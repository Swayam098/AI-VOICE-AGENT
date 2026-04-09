import httpx
import time

def run_tests():
    base_url = "http://localhost:8000"
    
    # 1. Health check
    try:
        r = httpx.get(f"{base_url}/health", timeout=5)
        print(f"GET /health: {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"GET /health failed: {e}")

    # 2. Status
    try:
        r = httpx.get(f"{base_url}/api/status", timeout=5)
        print(f"GET /api/status: {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"GET /api/status failed: {e}")

    # 3. Test chat
    try:
        payload = {"text": "Hello! How are you?"}
        r = httpx.post(f"{base_url}/api/test/chat", json=payload, timeout=60)
        print(f"POST /api/test/chat: {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"POST /api/test/chat failed: {e}")

if __name__ == "__main__":
    run_tests()
