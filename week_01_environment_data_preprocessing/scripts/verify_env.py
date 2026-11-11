
"""
Verify the Python environment and required AI/ML dependencies.

Checks installed library versions and determines whether CUDA/GPU
acceleration is available. CPU execution is supported when CUDA
is unavailable.
"""

import sys
import importlib


def main() -> None:
    """
    Check Python, required libraries, and GPU/CUDA availability.

    Prints the Python version, installed versions of required packages,
    and the result of the CUDA availability check.

    Returns:
        None. Results are printed to the terminal.
    """
    print("=" * 45)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 45)

    print("\nPython version:")
    print(sys.version)

    packages = {
        "pandas": "pandas",
        "sentence-transformers": "sentence_transformers",
        "ChromaDB": "chromadb",
        "spaCy": "spacy",
        "PyTorch": "torch"
    }

    print("\nChecking required libraries:")

    for name, module_name in packages.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(
                module, "__version__", "version not available"
            )
            print(f"[PASS] {name}: {version}")

        except Exception as error:
            print(f"[FAIL] {name}: {error}")

    print("\nChecking GPU/CUDA:")

    try:
        import torch

        if torch.cuda.is_available():
            print("[PASS] CUDA is available")
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            print("[INFO] CUDA is not available; CPU can be used.")

    except Exception as error:
        print("[FAIL] GPU check:", error)

    print("\nEnvironment check completed.")


if __name__ == "__main__":
    main()
