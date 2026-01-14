
import json

def simulate_frontend_logic(backend_response):
    print("--- Simulating Frontend Logic ---")
    result = backend_response
    payload = None

    if "data" in result:
        if result["data"].get("kind") == "DataProduct" and "payload" in result["data"]:
            print("📦 Nested Data Product detected")
            payload = result["data"]["payload"]
        else:
            print("⚠️ Direct/Unwrapped data detected")
            payload = result["data"]
    elif result.get("kind") == "DataProduct" and "payload" in result:
        print("📦 Data Product (QFS) detected at root")
        payload = result["payload"]
    else:
        print("⚠️ Legacy format detected")
        payload = result

    # Simulate updateAPUTable
    print("\n--- Simulating updateAPUTable ---")
    apus = []
    if payload and "processed_apus" in payload:
        apus = payload["processed_apus"]
    elif payload and "payload" in payload and "processed_apus" in payload["payload"]:
        apus = payload["payload"]["processed_apus"]
    elif payload and "data" in payload and "processed_apus" in payload["data"]:
        apus = payload["data"]["processed_apus"]

    print(f"APUs found: {len(apus)}")
    if len(apus) > 0:
        print(f"Sample APU: {apus[0].get('CODIGO_APU', 'N/A')}")
        return True
    else:
        print("No APUs found.")
        return False

# Test Case 1: Backend returns unwrapped data in 'data' key (Standard app.py behavior)
response_1 = {
    "success": True,
    "data": {
        "processed_apus": [{"CODIGO_APU": "101", "VALOR": 100}],
        "presupuesto": []
    }
}
print("\nTest Case 1 (Standard app.py):")
simulate_frontend_logic(response_1)

# Test Case 2: Backend returns DataProduct in 'data' key (Hypothetical)
response_2 = {
    "success": True,
    "data": {
        "kind": "DataProduct",
        "payload": {
            "processed_apus": [{"CODIGO_APU": "102", "VALOR": 200}]
        }
    }
}
print("\nTest Case 2 (Nested DataProduct):")
simulate_frontend_logic(response_2)

# Test Case 3: Backend returns DataProduct at root (Hypothetical QFS native)
response_3 = {
    "kind": "DataProduct",
    "payload": {
        "processed_apus": [{"CODIGO_APU": "103", "VALOR": 300}]
    }
}
print("\nTest Case 3 (Root DataProduct):")
simulate_frontend_logic(response_3)
