#!/usr/bin/env python3
"""
Install and setup script for aicache dependencies.
"""

import ssl
import subprocess
from pathlib import Path


def install_nltk_data():
    """Download NLTK data with SSL fix."""
    print("📦 Installing NLTK data...")
    try:
        import nltk

        # Fix SSL certificate issue
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Download required NLTK data
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        print("✅ NLTK data installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install NLTK data: {e}")
        return False


def check_ollama():
    """Check if Ollama is available and suggest installation."""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is available")
            return True
    except FileNotFoundError:
        pass

    print("⚠️  Ollama not found. LLM features will be limited.")
    print("   To install Ollama, visit: https://ollama.ai")
    return False


def setup_config():
    """Create default configuration."""
    config_dir = Path.home() / ".config" / "aicache"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        config_content = """# aicache configuration
cache:
  max_size_mb: 1000
  max_age_days: 30

semantic_cache:
  enabled: true
  model_name: "all-MiniLM-L6-v2"
  similarity_threshold: 0.8

behavioral_learning:
  enabled: true
  pattern_learning: true
  predictive_prefetching: true

intelligent_management:
  enabled: true
  max_size_mb: 1000
  max_age_days: 30
"""
        with open(config_file, "w") as f:
            f.write(config_content)
        print(f"✅ Created default config at {config_file}")
    else:
        print(f"✅ Config file already exists at {config_file}")


def main():
    print("🚀 Setting up aicache dependencies...")

    success_count = 0
    total_checks = 3

    # Install NLTK data
    if install_nltk_data():
        success_count += 1

    # Check Ollama
    if check_ollama():
        success_count += 1

    # Setup config
    setup_config()
    success_count += 1

    print(f"\n📊 Setup completed: {success_count}/{total_checks} components ready")

    if success_count == total_checks:
        print("🎉 All dependencies are set up correctly!")
    else:
        print("⚠️  Some optional components are missing but aicache will still work")

    print("\n📝 Next steps:")
    print("1. Run 'make setup' or 'python -m pip install .' to install aicache")
    print("2. Run 'aicache install --setup-wrappers' to install CLI wrappers")
    print("3. Ensure ~/.local/bin is in your PATH")


if __name__ == "__main__":
    main()
