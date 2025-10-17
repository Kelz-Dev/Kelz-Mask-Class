# Setup script for Streamlit Cloud deployment

echo "🔧 Installing system dependencies..."

# Update system packages
apt-get update -y
apt-get install -y python3-opencv libglib2.0-0 libsm6 libxrender1 libxext6

echo "📦 Installing Python dependencies..."
pip install --upgrade pip

# Install required libraries
pip install -r requirements.txt

echo "✅ Setup complete! Ready to launch Streamlit app."
