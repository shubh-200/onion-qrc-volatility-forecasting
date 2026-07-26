#!/usr/bin/env python3
"""test_qbraid_api.py — Diagnostic script to inspect qBraid SDK job listing methods.
"""
import os
import sys

api_key = os.environ.get("QBRAID_API_KEY")
print(f"API key present: {bool(api_key)}")

import qbraid
print(f"qbraid version: {getattr(qbraid, '__version__', 'unknown')}")

import qbraid.runtime as qrt
print(f"qbraid.runtime dir: {[d for d in dir(qrt) if not d.startswith('_')]}")

# Try QbraidProvider
try:
    from qbraid.runtime import QbraidProvider
    provider = QbraidProvider(api_key=api_key)
    print(f"QbraidProvider created successfully.")
    if hasattr(provider, "get_jobs"):
        jobs = provider.get_jobs()
        print(f"provider.get_jobs() returned: {len(jobs)} jobs")
except Exception as e:
    print(f"QbraidProvider error: {e}")

# Try qbraid_core or Client
try:
    from qbraid_core import QbraidClient
    client = QbraidClient(api_key=api_key)
    print("QbraidClient created successfully.")
    if hasattr(client, "search_jobs"):
        res = client.search_jobs()
        print(f"client.search_jobs() returned: {res}")
except Exception as e:
    print(f"QbraidClient error: {e}")
