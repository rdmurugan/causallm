# Contributing to CausalLLM

Thank you for considering contributing to **CausalLLM**! We welcome contributions from the community to improve causal reasoning capabilities in LLM workflows.

---

## 🤝 How to Contribute

### 1. Fork & Clone

```bash
git clone https://github.com/your-username/causal-llm.git
cd causal-llm
git checkout -b my-feature-branch
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Make Your Changes

* Add new modules or improve existing ones
* Write or update unit tests in the `tests/` folder
* Add usage examples or notebooks under `examples/`

### 4. Run Tests

```bash
python -m unittest discover tests/
```

### 5. Commit & Push

```bash
git add .
git commit -m "Add explanation module for SCM extraction"
git push origin my-feature-branch
```

### 6. Submit a Pull Request

* Open a PR on GitHub to `main`
* Describe what your PR does and reference any related issues

---

## 🧪 Testing Guidelines

* Place all tests in `tests/` with filenames like `test_*.py`
* Use `unittest` framework (default)
* Aim for test coverage on new functions/classes

---

## 📦 Code Style

* Follow [PEP8](https://pep8.org/)
* Use docstrings for all public classes and functions
* Keep imports organized and minimal

---

## 💡 Suggestions for Contributions

* New prompt templates (e.g., mediation, path-specific effects)
* Integration with causal ML libraries (e.g., DoWhy, EconML)
* More LLM adapters (Anthropic, Claude, HuggingFace models)
* Streamlit UI or command-line utility
* Advanced notebooks for causal decision support

---

## 🛡️ Code of Conduct

Please be respectful in your communication. All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md) (coming soon).

---

## 🧠 Need Help?

Open an issue or start a discussion. We’re happy to support contributors at all levels.

---

Let’s build interpretable, powerful causal AI tools together!

— Durai
