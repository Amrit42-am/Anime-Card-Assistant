#!/usr/bin/env python3
"""
Inspect and extract card database from animecardclashcalculator.com
"""

import json
import re
import urllib.request

URL = "https://animecardclashcalculator.com/_next/static/chunks/3744af37d64c7ffe.js"

req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
with urllib.request.urlopen(req, timeout=20) as resp:
    content = resp.read().decode("utf-8")

# Look for JSON.parse('...')
pattern = r'JSON\.parse\(\'((?:\\.|[^\'\\])*)\'\)'
matches = re.findall(pattern, content)
print(f"Found {len(matches)} JSON.parse strings")

for i, m in enumerate(matches):
    try:
        # Unescape escaped characters like \", \\, etc.
        unescaped = m.encode("utf-8").decode("unicode_escape")
        data = json.loads(unescaped)
        if isinstance(data, list):
            print(f"Match {i}: list of length {len(data)}")
            if data and isinstance(data[0], dict) and "name" in data[0]:
                print(f"Card list found in match {i}! Total cards: {len(data)}")
                print("First card sample:", data[0])
                print("Last card sample:", data[-1])
                # Check for duplicates or missing images
                names = [c.get("name") for c in data]
                print(f"Unique names: {len(set(names))} out of {len(names)}")
    except Exception as e:
        print(f"Match {i} decode error: {e}")
