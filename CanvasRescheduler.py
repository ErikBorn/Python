import requests
import json
from datetime import datetime, timezone
from dateutil.parser import isoparse
import pytz

# === Canvas API Configuration ===
# Instructions:
# 1. Log in to your Canvas instance.
# 2. Navigate to "Account" > "Settings".
# 3. Scroll down to "Approved Integrations" and generate a new access token.
# 4. Copy the token and keep it secure. Paste it below where indicated.
# 5. Find your Canvas base URL (e.g., "https://yourinstitution.instructure.com").

# === Configuration ===
API_TOKEN = "13716~4EH4Bvhh8UPVxw9WyfP689F6wQYhmeDXARQyRQ3BQuF9xzHZEK4QXUk4nFDFMBQD"  # Replace with your Canvas API token
API_TOKEN_test = "13716~zVzPUBGwZzy7T7rDVK7CCm7R9nNA9UBfRk9Dmu29QkeaJFG7yHFn8Eu2Ear287Dh"  # Replace with your Canvas API token
BASE_URL = "https://sma.instructure.com/api/v1"

test = True
if test:
    API_TOKEN = API_TOKEN_test
    BASE_URL = "https://sma.test.instructure.com/api/v1"  # Replace with your Canvas instance base URL

# === Script Configuration ===
# Enter the course ID and desired due times for sections.
COURSE_ID = "5278"  # Replace with your course ID
SECTION_DUE_TIMES = {
    "9th Grade Literature - Passin - 1100": "08:30:00",  # Replace "A Block" with the section name and time in HH:MM:SS
    "9th Grade Literature - Passin - 1200": "10:30:00"
}

LOCAL_TIMEZONE = pytz.timezone("America/Denver")  # Replace with your local timezone
DEBUG = True  # Set to False to reduce verbosity
VDEBUG = False #Verbose debug

# Safety window for assignments
START_DATE = "2024-01-01"  # Inclusive start date in YYYY-MM-DD
END_DATE = "2024-12-31"    # Inclusive end date in YYYY-MM-DD

# === Helper Functions ===
def get_headers():
    """Return headers for Canvas API requests."""
    return {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }

def convert_to_utc(local_time_str):
    """Convert a local time string (HH:MM:SS) to UTC time string."""
    local_time = datetime.strptime(local_time_str, "%H:%M:%S")
    local_time = LOCAL_TIMEZONE.localize(local_time)
    utc_time = local_time.astimezone(pytz.utc)
    return utc_time.strftime("%H:%M:%S")

def get_assignments(course_id):
    """Fetch all assignments for a course with pagination."""
    assignments = []
    url = f"{BASE_URL}/courses/{course_id}/assignments"
    params = {"per_page": 100}  # Fetch up to 100 assignments per page (adjust as needed)

    while url:
        response = requests.get(url, headers=get_headers(), params=params)
        if DEBUG and VDEBUG:
            print("Debug: API Response -", response.status_code, response.text)  # Debugging line
        response.raise_for_status()
        assignments.extend(response.json())

        # Get the "next" page link from the headers if it exists
        links = response.headers.get("Link", "")
        next_link = None
        for link in links.split(","):
            if 'rel="next"' in link:
                next_link = link[link.find("<") + 1 : link.find(">")]
                break

        url = next_link  # Update the URL to the next page

    return assignments

def get_sections(course_id):
    """Fetch all sections for a course."""
    url = f"{BASE_URL}/courses/{course_id}/sections"
    response = requests.get(url, headers=get_headers())
    response.raise_for_status()
    return response.json()

def create_override(assignment_id, section_id, due_date, due_time):
    """Create or update a section-specific override for an assignment."""
    utc_due_time = convert_to_utc(due_time)
    url = f"{BASE_URL}/courses/{COURSE_ID}/assignments/{assignment_id}/overrides"
    try:
        # Fetch existing overrides for the assignment
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        existing_overrides = response.json()

        # Check if an override exists for the section
        existing_override = next((o for o in existing_overrides if o["course_section_id"] == section_id), None)

        if existing_override:
            # Update the existing override if the due time is different
            existing_due_at = existing_override["due_at"]
            desired_due_at = f"{due_date}T{utc_due_time}Z"

            if existing_due_at != desired_due_at:
                update_url = f"{url}/{existing_override['id']}"
                payload = {
                    "assignment_override": {
                        "due_at": desired_due_at,  # ISO 8601 format
                    }
                }
                update_response = requests.put(update_url, headers=get_headers(), data=json.dumps(payload))
                update_response.raise_for_status()
                if DEBUG:
                    print(f"Updated override for section {section_id}. Response: {update_response.json()}")
                return update_response.json()
            else:
                if DEBUG:
                    print(f"No changes needed for override in section {section_id}.")
                return existing_override
        else:
            # Create a new override if none exists
            payload = {
                "assignment_override": {
                    "student_group_id": None,
                    "course_section_id": section_id,
                    "due_at": f"{due_date}T{utc_due_time}Z",  # ISO 8601 format
                }
            }
            create_response = requests.post(url, headers=get_headers(), data=json.dumps(payload))
            create_response.raise_for_status()
            if DEBUG:
                print(f"Created new override for section {section_id}. Response: {create_response.json()}")
            return create_response.json()
    except requests.exceptions.RequestException as e:
        if DEBUG:
            print(f"Failed to create/update override for section {section_id}: {e}")
        return None

def update_assignment_due_time(assignment_id, due_date, due_time):
    """Update the native due time for an assignment if there is only one section."""
    utc_due_time = convert_to_utc(due_time)
    url = f"{BASE_URL}/courses/{COURSE_ID}/assignments/{assignment_id}"
    payload = {
        "assignment": {
            "due_at": f"{due_date}T{utc_due_time}Z",  # ISO 8601 format
        }
    }
    try:
        response = requests.put(url, headers=get_headers(), data=json.dumps(payload))
        response.raise_for_status()
        if DEBUG:
            print(f"Successfully updated native due time for assignment {assignment_id}. Response: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        if DEBUG:
            print(f"Failed to update native due time for assignment {assignment_id}: {e}")
        return None

def main():
    """Main logic for the script."""
    print("Fetching assignments...")
    assignments = get_assignments(COURSE_ID)

    print("Fetching sections...")
    sections = get_sections(COURSE_ID)
    section_map = {s["name"]: s["id"] for s in sections}

    # Parse safety window dates
    start_date = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Get the current time to filter future assignments
    now = datetime.now(timezone.utc)

    for assignment in assignments:
        try:
            assignment_id = assignment["id"]
            assignment_name = assignment["name"]
            due_at = assignment.get("due_at")
            overrides = assignment.get("overrides", [])  # Use overrides from the assignment object

            if not due_at:
                if DEBUG:
                    print(f"Skipping assignment '{assignment_name}' (ID: {assignment_id}) with no due date.")
                continue

            # Parse due date to a datetime object
            assignment_due_date = isoparse(due_at)

            if DEBUG:
                print(f"Assignment due date/time: {assignment_due_date}")  # Debugging

            # Check if assignment is within the safety window
            if not (start_date <= assignment_due_date <= end_date):
                if DEBUG:
                    print(f"Skipping assignment '{assignment_name}' (ID: {assignment_id}) outside safety window.")
                continue

            # Process only future assignments
            if assignment_due_date > now:
                print(f"Processing assignment: {assignment_name} (ID: {assignment_id})")

                if len(sections) == 1:
                    # Update native due time if there is only one section
                    print(f"Only one section found. Updating native due time for assignment {assignment_name}.")
                    update_assignment_due_time(assignment_id, due_at.split("T")[0], list(SECTION_DUE_TIMES.values())[0])
                else:
                    # Create or update overrides for sections
                    for section_name, due_time in SECTION_DUE_TIMES.items():
                        if section_name in section_map:
                            section_id = section_map[section_name]
                            if DEBUG:
                                print(f"Processing section: {section_name} (ID: {section_id})")

                            # Check if an override exists for this section
                            existing_override = next((o for o in overrides if o["course_section_id"] == section_id), None)
                            if DEBUG:
                                print(f"Existing override for section {section_name}: {existing_override}")

                            if existing_override:
                                # Compare existing due time with desired due time
                                existing_due_at = existing_override["due_at"]
                                desired_due_at = f"{due_at.split('T')[0]}T{convert_to_utc(due_time)}Z"
                                if DEBUG:
                                    print(f"Existing due date: {existing_due_at}, Desired due date: {desired_due_at}")

                                existing_due_dt = isoparse(existing_due_at)
                                desired_due_dt = isoparse(desired_due_at)

                                if existing_due_dt != desired_due_dt:
                                    print(f"Updating override for section: {section_name}")
                                    create_override(assignment_id, section_id, due_at.split("T")[0], due_time)
                                else:
                                    print(f"No changes needed for section: {section_name} (Existing: {existing_due_at})")
                            else:
                                # Create a new override if none exists
                                print(f"Creating override for section: {section_name}")
                                create_override(assignment_id, section_id, due_at.split("T")[0], due_time)
                        else:
                            print(f"Section {section_name} not found in course.")
            else:
                if DEBUG:
                    print(f"Skipping past assignment: {assignment_name} (ID: {assignment_id})")
        except Exception as e:
            if DEBUG:
                print(f"Error processing assignment {assignment_name} (ID: {assignment_id}): {e}")

if __name__ == "__main__":
    main()
