## Running the UI

Follow the steps below to run the UI locally:

### 1. Create and Activate a Python Environment
Create a virtual environment and activate it.

```bash
python -m venv venv
```

Activate the environment:

**Linux / Mac**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 2. Install Required Libraries
Install all dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit UI
Navigate to the `ui` folder and start the Streamlit application.

```bash
cd ui
streamlit run streamlit_v2.py
```

The UI will start and open in your browser.