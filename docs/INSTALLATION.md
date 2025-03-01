## System Requirements
- Python 3.11 or higher
- Stable internet connection (min 1 Mbps)
- Active Deriv.com account
- VS Code (recommended)

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/your-username/deriv-ml-trading-bot.git
cd deriv-ml-trading-bot
```

### 2. Virtual Environment Setup
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Copy `.env.example` to `.env`:
```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Edit `.env` with your API tokens:
```
DERIV_API_TOKEN_DEMO=your_demo_token
DERIV_API_TOKEN_REAL=your_real_token
DERIV_BOT_ENV=demo
APP_ID=1089
```

### 5. Verify Installation
```bash
python test_api_connectivity.py
```

For detailed VS Code setup, see [VS Code Setup Guide](VSCODE_SETUP.md).