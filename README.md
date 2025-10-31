# MAS202 - How to install and run the project

## Prerequisites
Before you begin, ensure you have met the following requirements:
- You have installed Python 3.7 or higher.
- You have installed git on your machine.
- You have installed Visual Studio Code or any other code editor of your choice.

## How to install the prerequisites
To install the required packages, you can use pip. Run the following command in your terminal:
- Install python: 
- For Windows, download and install Python from [python.org](https://www.python.org/downloads/windows/).
- For macOS, you can use Homebrew:
```bash
brew install python
```
- Install git:
```bash
# For Windows
choco install git

# For macOS
brew install git
```

- Install Visual Studio Code:
- For Windows, download and install VS Code from [code.visualstudio.com](https://code.visualstudio.com/download).
- For macOS, you can use Homebrew Cask:
```bash
brew install --cask visual-studio-code
```

## Step 1: Clone the repository
First, clone the repository to your local machine using the following command (make sure you select a directory where you want to clone the project, like Desktop or Documents):
```bash
git clone https://github.com/vinhhaphoi/mas202-bank-marketing-dataset.git
cd mas202-bank-marketing-dataset
```

## Step 2: Create a virtual environment and run the project
- On Windows: run the install_run.sh script
```bash
./install_run.sh
```

- On macOS: run the commands line by line below
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Step 3: Select options on terminal for see the result
- After running the project, you will be prompted to select options in the terminal. Follow the on-screen instructions to see the results.