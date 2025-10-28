# 1. Navigate to project directory
cd /Users/vinhhaphoi/Documents/mas202

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install required packages
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# 6. Verify installation
pip list

# 7. Run the analysis
python analysis.py

# 8. Generate report
python report_generator.py
