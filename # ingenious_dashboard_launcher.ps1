# ingenious_dashboard_launcher.ps1

# Set your project directory
Set-Location "C:\Users\alpha\OneDrive\Desktop\IngeniousIrrigation"

# Activate your Python environment if needed
# & "C:\Path\To\venv\Scripts\Activate.ps1"   # ← uncomment and edit if you use a virtual environment

# Run the Flask dashboard
python ingenious_irrigation_dashboard.py

# Keep window open after run (optional)
Read-Host -Prompt "Press Enter to close this window"
