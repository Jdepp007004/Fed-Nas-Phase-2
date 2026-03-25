Step 1 — Download & split the dataset (run once, on your machine)
powershell
# From repo root
pip install requests pandas scikit-learn
python download_and_split.py --max-cases 10000 --n-clients 4
This will create:

data/
  full_dataset.csv   ← all records
  client_1.csv       ← for you / teammate 1
  client_2.csv       ← teammate 2
  client_3.csv       ← teammate 3
  client_4.csv       ← teammate 4
Step 2 — Send each teammate
Give them:

The fl_platform folder (zip or GitHub)
Their client_X.csv file
Your ngrok URL
Project ID: 193b8223-311e-4de4-809d-68d431da46ab
Step 3 — Each teammate runs this
powershell
# Install client deps (one time)
pip install -r client/requirements.txt
# Run (replace client_1 with their number)
python client/client_app.py `
  --server  https://YOUR_NGROK_URL `
  --username hospital_1 `
  --password secret1 `
  --hospital "Hospital 1" `
  --email    admin1@hospital.org `
  --csv      data/client_1.csv `
  --proj     193b8223-311e-4de4-809d-68d431da46ab
Or they can use the browser UI — just open client/client_ui.html and go through the 4-step wizard.

Step 4 — You approve them on the dashboard
Go to http://localhost:8000/dashboard → click the yellow ✓ approve badge next to each client → training starts automatically.