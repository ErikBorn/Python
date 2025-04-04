from flask import Flask, render_template, request, jsonify
import requests
import json
from datetime import datetime, timezone
from dateutil.parser import isoparse
import pytz
app = Flask(__name__)

# === Configuration ===
API_TOKEN = "13716~4EH4Bvhh8UPVxw9WyfP689F6wQYhmeDXARQyRQ3BQuF9xzHZEK4QXUk4nFDFMBQD"  # Replace with your Canvas API token
API_TOKEN_test = "13716~zVzPUBGwZzy7T7rDVK7CCm7R9nNA9UBfRk9Dmu29QkeaJFG7yHFn8Eu2Ear287Dh"  # Replace with your Canvas API token
BASE_URL = "https://sma.instructure.com/api/v1"

test = True
if test:
    API_TOKEN = API_TOKEN_test
    BASE_URL = "https://sma.test.instructure.com/api/v1"  # Replace with your Canvas instance base URL

LOCAL_TIMEZONE = pytz.timezone("America/Denver")  # Replace with your timezone

# === Helper Functions ===
def get_headers():
    return {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }

def convert_to_utc(local_time_str):
    local_time = datetime.strptime(local_time_str, "%H:%M:%S")
    local_time = LOCAL_TIMEZONE.localize(local_time)
    utc_time = local_time.astimezone(pytz.utc)
    return utc_time.strftime("%H:%M:%S")

def get_assignments(course_id):
    assignments = []
    url = f"{BASE_URL}/courses/{course_id}/assignments"
    while url:
        response = requests.get(url, headers=get_headers(), params={"per_page": 100})
        response.raise_for_status()
        assignments.extend(response.json())
        next_link = [
            link[link.find("<") + 1 : link.find(">")]
            for link in response.headers.get("Link", "").split(",")
            if 'rel="next"' in link
        ]
        url = next_link[0] if next_link else None
    return assignments

def get_sections(course_id):
    url = f"{BASE_URL}/courses/{course_id}/sections"
    response = requests.get(url, headers=get_headers())
    response.raise_for_status()
    return response.json()

def create_or_update_override(assignment_id, section_id, due_date, due_time, course_id):
    utc_due_time = convert_to_utc(due_time)
    url = f"{BASE_URL}/courses/{course_id}/assignments/{assignment_id}/overrides"
    payload = {
        "assignment_override": {
            "course_section_id": section_id,
            "due_at": f"{due_date}T{utc_due_time}Z",
        }
    }
    response = requests.post(url, headers=get_headers(), json=payload)
    response.raise_for_status()
    return response.json()

# === Flask Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        course_id = request.form['course_id']
        sections = []
        for i in range(1, 7):
            section_name = request.form.get(f'section_name_{i}')
            due_time = request.form.get(f'due_time_{i}')
            if section_name and due_time:
                sections.append({"name": section_name, "due_time": due_time})

        # Get assignments and sections
        assignments = get_assignments(course_id)
        sections_data = get_sections(course_id)
        section_map = {s["name"]: s["id"] for s in sections_data}

        results = []
        now = datetime.now(timezone.utc)
        for assignment in assignments:
            assignment_id = assignment["id"]
            assignment_due_at = assignment.get("due_at")
            if assignment_due_at and isoparse(assignment_due_at) > now:
                for section in sections:
                    section_name = section["name"]
                    if section_name in section_map:
                        section_id = section_map[section_name]
                        due_date = assignment_due_at.split("T")[0]
                        result = create_or_update_override(assignment_id, section_id, due_date, section["due_time"],course_id)
                        results.append({"assignment_id": assignment_id, "section_name": section_name, "result": result})
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)