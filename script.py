
from ollama import Client
import requests
import sounddevice as sd
import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
import sys
import time
from Jetson import GPIO


LLM_MODEL: str = "gemma3:27b"
client: Client = Client(
    host = "http://10.1.69.213:11434/"
)

PROMPT_PREFIX = """
I need to extract location information and return a string that can be accepted when calling the wttr.in API. Return the location, formatted with "+" instead of spaces. If and only if the location is a landmark, not a city or country, add a tilde before the output. Lastly, if it is an airport, output the 3-letter airport code in all lower-case letters. Do not include any punctuation or symbols. Ensure that the response does not have extra white space surrounding the output and that the string that is returned is exactly as specified with no extra characters or white space.

For example:
Input: What's the weather in Rio Rancho?; Output: Rio+Rancho
Input: What's the weather in New York?; Output: New+York
Input: What's the weather at the Eiffel Tower?; Output: ~Eiffel+Tower
Input: What's the weather at the Los Angeles Airport?; Output: lax
Input: What's the weather at the Denver Airport?; Output: den

The location I need you to return is:
"""

def llm_parse_for_wttr(input_str: str) -> str:
    response = client.chat(
        model = LLM_MODEL,
        messages = [
            {'role': 'system', 'content': PROMPT_PREFIX},
            {'role': 'user', 'content': input_str}
        ])
    return response['message']['content']

# Test cases
test_cases = [
    {"input": "What's the weather in Rio Rancho?", "expected": "Rio+Rancho"},
    {"input": "What's the weather in New York?", "expected": "New+York"},
    {"input": "What's the weather in New York City?", "expected": "New+York+City"},
    {"input": "What's the weather in Los Angeles?", "expected": "Los+Angeles"},
    {"input": "What's the weather in San Francisco?", "expected": "San+Francisco"},
    {"input": "What's the weather at the Eiffel Tower?", "expected": "~Eiffel+Tower"},
    {"input": "What's the weather at the Garden of the Gods?", "expected": "~Garden+of+the+Gods"}
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


def get_weather(location: str) -> str:
    weather_str = requests.get(f"https://wttr.in/{location}")
    return weather_str.text


def build_pipeline(model_id: str, torch_dtype: torch.dtype, device: str) -> Pipeline:
    """Creates a Hugging Face automatic-speech-recognition pipeline on the given device."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

def record_audio(duration_seconds: int = 10) -> npt.NDArray:
    """Record duration_seconds of audio from default microphone.
    Return a single channel numpy array."""
    sample_rate = 16000  # Hz
    samples = int(duration_seconds * sample_rate)
    # Will use default microphone; on Jetson this is likely a USB WebCam
    sd.default.device = (24, None)
    audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype=np.float32)
    # Blocks until recording complete
    sd.wait()
    # Model expects single axis
    return np.squeeze(audio, axis=1)


    
def load_speech_model():
# Get model as argument, default to "distil-whisper/distil-medium.en" if not given
    model_id = sys.argv[1] if len(sys.argv) > 1 else "distil-whisper/distil-medium.en"
    #print(f"Using model_id {model_id}", flush=True)
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    #print(f"Using device {device}.", flush=True)

    #print("Building model pipeline...", flush=True)
    pipe = build_pipeline(model_id, torch_dtype, device)
    print(type(pipe))
    #print("Done1", flush=True)
    return pipe

def record(pipe):
    print("Recording...", flush=True)
    audio = record_audio()
    print("Done", flush=True)

    #print("Transcribing...", flush=True)
    start_time = time.time_ns()
    speech = pipe(audio)
    end_time = time.time_ns()
    #print("Done", flush=True)

    #print(speech)
    #print(f"Transcription took {(end_time - start_time) / 1e9:.2f} seconds", flush=True)
    return speech['text']


# Run the tests
if __name__ == "__main__":
    print("Loading model. Please wait...", flush=True)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(29, GPIO.IN)
    pipe = load_speech_model()
    #print("pipe loaded", pipe)  # Add this

    while True:
        print("Press button when ready...", flush=True)
        GPIO.wait_for_edge(29, GPIO.RISING)
        #print("Pressed")
        speech = record(pipe)
        location = llm_parse_for_wttr(speech)
        weather = get_weather(location)
        print(weather)
