from dotenv import load_dotenv
import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CLIENT_SECRET_FILE = os.getenv("GOOGLE_CREDENTIALS")

print("Using credentials file:", CLIENT_SECRET_FILE)

creds = None

# Load saved token if exists
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)

# If not valid, login
if not creds or not creds.valid:
    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRET_FILE, SCOPES
    )
    creds = flow.run_local_server(port=0)

    with open("token.json", "w") as token:
        token.write(creds.to_json())

# Build Drive service
drive_service = build("drive", "v3", credentials=creds)

# List files
results = drive_service.files().list(
    pageSize=10,
    fields="files(id, name)"
).execute()

items = results.get("files", [])

if not items:
    print("No files found.")
else:
    print("Files:")
    for item in items:
        print(f"{item['name']} ({item['id']})")
