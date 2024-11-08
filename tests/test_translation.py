import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from services.malay2sql_service import translate_malay_to_english

def test_translation():
    # Print GPU information
    print("\nGPU Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("-" * 50)

    # Test cases with Malay sentences
    test_cases = [
        "ko memang cam babi sial",
        "Siapa nama kamu?",
        "Berapa harga barang ini?"
    ]

    print("\nTesting Malay to English translation:")
    print("-" * 50)
    
    for malay_text in test_cases:
        translation = translate_malay_to_english(malay_text)
        print(f"Malay: {malay_text}")
        print(f"English: {translation}")
        print("-" * 50)

if __name__ == "__main__":
    test_translation()