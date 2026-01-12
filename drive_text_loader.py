from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv
import os

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CLIENT_SECRET_FILE = os.getenv("GOOGLE_CREDENTIALS")

if not CLIENT_SECRET_FILE:
    raise ValueError("GOOGLE_CREDENTIALS not found in .env file")

# ---------- AUTH ----------
creds = None
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)

if not creds or not creds.valid:
    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRET_FILE, SCOPES
    )
    creds = flow.run_local_server(port=0)
    with open("token.json", "w") as token:
        token.write(creds.to_json())

drive_service = build("drive", "v3", credentials=creds)

# ---------- RAG READY FUNCTION ----------
def load_drive_texts():
    texts = []

    results = drive_service.files().list(
        q="mimeType='text/plain' and name contains 'project'",
        fields="files(id, name)"
    ).execute()

    files = results.get("files", [])

    for file in files:
        request = drive_service.files().get_media(fileId=file["id"])
        content = request.execute()
        text = content.decode("utf-8", errors="ignore")

        texts.append(
            f"Drive File: {file['name']}\n{text}"
        )

    return texts
