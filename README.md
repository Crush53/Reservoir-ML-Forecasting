[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/O6idXAzd)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13088519&assignment_repo_type=AssignmentRepo)

Project Overview
This project includes a suite of tools and scripts for processing data, building models, and visualizing results within a Django-based web application. It's structured as follows:

res_tool: The main directory containing the virtual environment and Django project.
process_data.py: A script for importing data into the database.
model_scripts: A directory containing scripts to individually run models.
model_creation_scripts: Contains functions for building models and processing data.
res_view: A Django app within res_tool responsible for model training/testing and graph creation.
Initial Setup
1. Duplicate Virtual Environment
Before you begin, ensure you have Python installed on your machine. Then, follow these steps:

Navigate to the Project Directory

1- Open your terminal or command prompt and navigate to the res_tool directory:

```bash
cd cd .\res_tool\
```

2- Duplicate the Virtual Environment

Once inside res_tool, duplicate the virtual environment:

```bash
# For Windows
venv\Scripts\activate

# For Unix or MacOS
source venv/bin/activate
```
3- Ensure all required Python packages are installed:

```bash
pip install -r requirements.txt
```
Before populating you database, you will need a .env file containing:

DB_HOST= your_host
DB_NAME=your_db_name
DB_USER=your_username
DB_PASS=your_password

4- To populate your database with the necessary data, run the process_data.py script: 

```bash
python process_data.py
```

Running Models and Scripts:

Individual Models: You can run individual models from the model_scripts folder.
Model Creation: Use scripts in model_creation_scripts for more specialized model building and data processing.


5- If you already have the data in your database and the virtual environment is active, you can start the Django server:

```bash
python manage.py runserver
```

Notes:

Patience Required: Training and testing models, as well as graph creation (in res_tool\res_view\views.py and res_tool\res_view\utils.py), can be time-consuming. Please be patient during these processes.
Web Interface: Once the Django server is running, you can access the web interface by navigating to http://localhost:8000/ in your web browser.
