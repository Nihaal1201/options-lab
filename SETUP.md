# Setup Guide (one-time, ~30 minutes)

Follow these in order. If anything shows an error, copy the exact message and send it to Claude.

## 1. Install Python
1. Go to https://www.python.org/downloads/ and download the latest Python 3 (3.12 or newer).
2. **Windows:** on the first installer screen, tick **"Add python.exe to PATH"** before clicking Install. (This is the #1 beginner mistake.)
3. **Mac:** just run the installer.
4. Check it worked: open Terminal (Mac) or Command Prompt (Windows) and type:
   ```
   python --version
   ```
   (On Mac, use `python3 --version`.) You should see something like `Python 3.12.x`.

## 2. Install VS Code
1. Download from https://code.visualstudio.com/ and install.
2. Open VS Code → click the **Extensions** icon on the left sidebar (four squares) → search and install:
   - **Python** (by Microsoft)
   - **Jupyter** (by Microsoft), useful later for experiments

## 3. Open the project
1. Unzip `options-lab.zip` somewhere easy, e.g. `Documents/options-lab`.
2. In VS Code: **File → Open Folder…** → choose the `options-lab` folder.
3. Open the built-in terminal: **Terminal → New Terminal** (or Ctrl+` ). Everything below is typed there.

## 4. Create a virtual environment
A virtual environment ("venv") is a private box of packages just for this project, so projects don't clash with each other.

**Mac:**
```
python3 -m venv .venv
source .venv/bin/activate
```
**Windows:**
```
python -m venv .venv
.venv\Scripts\activate
```
You'll now see `(.venv)` at the start of your terminal line. **Every time** you reopen the project, run the activate line again.

If VS Code pops up "We noticed a new environment… select it?", click **Yes**.

> Windows error about "running scripts is disabled"? Run this once, then try again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 5. Install the packages
```
pip install -r requirements.txt
```

## 6. Run your first quant code
```
python scripts/module1_demo.py
```
You should see option prices printed and a chart saved to `images/module1_price_curves.png`.

Then run the tests:
```
python -m pytest
```
All tests should say **passed**.

That's it, you're set up.
