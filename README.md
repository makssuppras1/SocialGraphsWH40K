# Social Graphs and Interactions - Final Assignment
This project uses [uv](https://github.com/astral-sh/uv) as the package manager for dependency management.

### Installing uv
Install it with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or on macOS with Homebrew:
```bash
brew install uv
```

### Set up the project
1. **Create a virtual environment and install dependencies:**
   ```bash
   uv venv sogra
   source sogra/bin/activate
   uv pip install --editable .
   ```

   Or if the `sogra` virtual environment already exists:
   ```bash
   source sogra/bin/activate
   uv pip install --editable .
   ```
   
   Or use `uv sync` with the active environment:
   ```bash
   source sogra/bin/activate
   uv sync --active
   ```

2. **Activate the virtual environment:**
   ```bash
   # Linux/macOS
   source sogra/bin/activate
   
   # Windows
   sogra\Scripts\activate
   ```
