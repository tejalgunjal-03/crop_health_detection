import requests
import json
import os

url = "http://127.0.0.1:8000/api/uav/analyze"
# First we need a session. Let's create one.
sess_res = requests.post("http://127.0.0.1:8000/api/sessions", json={"monitoring_mode": "UAV"}, headers={"Authorization": "Bearer TEST_TOKEN_NOT_NEEDED"})
# Wait, auth is required.
