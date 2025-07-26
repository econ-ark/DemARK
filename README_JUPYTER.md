# 🚀 DemARK Jupyter Lab Access

## **Quick Start**

**Run this command from any terminal:**

```bash
./start-jupyter-final.sh
```

## **What This Does**

1. **Starts the DemARK container** (if not running)
2. **Installs Jupyter Lab** (if not installed)
3. **Starts Jupyter Lab** on an available port
4. **Provides clear instructions** for accessing it

## **How to Access Jupyter Lab**

After running the script, you'll see instructions like:

```
✅ Jupyter Lab is running in container on port 8890

🌐 ACCESS INSTRUCTIONS:
========================

Jupyter Lab is running in the container.

To access it from your browser, run this command in a NEW terminal:

  socat TCP-LISTEN:8890,fork TCP:localhost:8890

Then open your browser to: http://localhost:8890
```

## **Step-by-Step Instructions**

1. **Run the launcher**: `./start-jupyter-final.sh`
2. **Open a NEW terminal window**
3. **Run the socat command** shown in the output
4. **Open your browser** to the URL shown
5. **Start using Jupyter Lab!**

## **Requirements**

- **Docker Desktop** must be running
- **socat** (install with `brew install socat` on macOS)

## **What You'll Find**

- **`test_jupyter.ipynb`** - Test notebook with math and HARK
- **`notebooks/`** - All DemARK example notebooks
- **Full scientific environment** - Python, NumPy, Matplotlib, HARK

## **Troubleshooting**

**If you get "command not found" for socat:**
```bash
brew install socat
```

**If the port is in use:**
The script automatically finds the next available port.

**If the container won't start:**
Make sure Docker Desktop is running.

## **Success Indicators**

- ✅ Jupyter Lab interface loads in browser
- ✅ File browser shows DemARK notebooks
- ✅ No authentication required
- ✅ All scientific packages available

**🎉 That's it! You're ready to explore DemARK!** 