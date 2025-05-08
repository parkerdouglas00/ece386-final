
# from ollama import Client

import requests

LLM_MODEL: str = "gemma3:27b"
OLLAMA_URL = "http://ai.dfec.xyz:11434/"

PROMPT_PREFIX = """
I need to extract location information and return a string that can be accepted when calling the wttr.in API. Return the location, formatted with "+" instead of spaces. If and only if the location is a landmark, not a city or country, add a tilde before the output. Do not include any punctuation or symbols.

For example:
Input: What's the weather in Rio Rancho?; Output: Rio+Rancho
Input: What's the weather in New York?; Output: New+York
Input: What's the weather at the Eiffel Tower?; Output: ~Eiffel+Tower

The location I need you to return is:
"""

def llm_parse_for_wttr(input_str: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "prompt": PROMPT_PREFIX + input_str,
        "stream": False
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama error: {e}")

# Test cases
test_cases = [
    {"input": "What's the weather in Rio Rancho?", "expected": "Rio+Rancho"},
    {"input": "What's the weather in New York?", "expected": "New+York"},
    {"input": "What's the weather in New York City?", "expected": "New+York+City"},
    {"input": "What's the weather in Los Angeles?", "expected": "Los+Angeles"},
    {"input": "What's the weather in San Francisco?", "expected": "San+Francisco"},
    {"input": "What's the weather at the Eiffel Tower?", "expected": "~Eiffel+Tower"},
    {"input": "What's the weather at the Garden of the Gods?", "expected": "~Garden+of+The+Gods"}
]

def run_tests(test_cases):
    num_passed = 0
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['input']}")
        try:
            result = llm_parse_for_wttr(test["input"])
            print("LLM Output  :", result)
            print("Expected    :", test["expected"])

            if result == test["expected"]:
                print("✅ PASS")
                num_passed += 1
            else:
                print("❌ FAIL")
        except Exception as e:
            print("💥 ERROR:", e)

    print(f"\nSummary: {num_passed} / {len(test_cases)} tests passed.")

# Run the tests
if __name__ == "__main__":
    print("You pressed the button!")
    run_tests(test_cases)

