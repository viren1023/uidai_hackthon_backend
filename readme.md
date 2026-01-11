### Create python virtual environment
python -m venv venv

### To copy dependencys in requirements.txt
pip freeze > requirements.txt

### To install dependencys from requirements.txt
pip install -r requirements.txt

### Start server
uvicorn app.main:app --reload

### Server URL
http://127.0.0.1:8000/docs