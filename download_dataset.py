
import io
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

FILE_ID = "1o8FfiIjvlbq7msw_Uga39zCX29OvMQxk"
OUTPUT = Path("sample_download.7z")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

creds = Credentials.from_authorized_user_file("token.json", SCOPES)
service = build("drive", "v3", credentials=creds)

request = service.files().get_media(fileId=FILE_ID)

print("Starting download...")

with io.FileIO(str(OUTPUT), "wb") as fh:
    downloader = MediaIoBaseDownload(
        fh, request, chunksize=10 * 1024 * 1024
    )

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"Download: {int(status.progress() * 100)}%")

print("Download completed!")
print(f"File size: {OUTPUT.stat().st_size / (1024 * 1024):.2f} MB")
