import os.path
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from bs4 import BeautifulSoup  # Optional: to clean HTML emails

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    """Authenticates the user and returns the Gmail service object."""
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                return None # Setup not complete
                
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

def fetch_recent_emails(max_count=20):
    """Fetches the latest emails and returns them as text documents."""
    service = get_gmail_service()
    if not service:
        return "MISSING_CREDS", []

    # Call the Gmail API
    results = service.users().messages().list(userId='me', maxResults=max_count).execute()
    messages = results.get('messages', [])

    email_docs = []
    
    print(f"Fetching {len(messages)} emails...")

    for msg in messages:
        try:
            txt = service.users().messages().get(userId='me', id=msg['id']).execute()
            
            # Extract Headers
            headers = txt['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
            sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown")
            date = next((h['value'] for h in headers if h['name'] == 'Date'), "")

            # Extract Body (Snippet is safer/faster than full body parsing)
            snippet = txt.get('snippet', '')
            
            # Create a structured text block for the bot
            content = f"""
            --- EMAIL START ---
            From: {sender}
            Date: {date}
            Subject: {subject}
            
            Content:
            {snippet}
            --- EMAIL END ---
            """
            
            email_docs.append({
                "text": content,
                "meta": {"source": f"Gmail: {subject[:30]}...", "type": "email"}
            })
            
        except Exception as e:
            print(f"Error reading email {msg['id']}: {e}")
            continue

    return "SUCCESS", email_docs