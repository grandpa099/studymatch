STUDY MATCH - AI TUTOR / GLOBAL LEARNING UPDATE

FILES
- next.html: Study Match frontend
- app.py: Flask backend with AI answer, text-to-speech, and AI Tutor file-reading endpoints
- requirements.txt: Python dependencies
- .env.example: example environment configuration

RUN LOCALLY (Python 3.10+ recommended)
1. Extract this folder.
2. Open a terminal in this folder.
3. Install dependencies:
   python -m pip install -r requirements.txt
4. Copy .env.example to .env.
5. Put your private Gemini API key in .env. Never publish your real API key or put it into the HTML.
6. Start the app:
   python app.py
7. Open http://127.0.0.1:5000/

WHAT WAS ADDED
- AI Tutor can accept PDF, TXT, MD, CSV, JPG, PNG, and WEBP attachments up to 10 MB and ask the AI to explain the uploaded material. This requires a working Gemini API key and internet access.
- Study Tutor retains its existing topic-based, conversational teaching and voice features.
- Wallet now includes Starter, Study Plus, and Study Pro plan cards and a request-selection demo.
- Removed the Market, Jobs & Gigs, Services, and Play page sections from this build; they are not part of the available navigation.
- Existing Learn, Tutor, Watch, Study AI, My Notes, Communities, Planner, and Wallet pages are retained.

IMPORTANT LIMITATIONS
- The plan cards are a frontend demonstration only. They do not charge users or enforce subscriptions.
- Real global payments require a payment provider account, server-side payment verification/webhooks, and country-specific configuration.
- Production global use still requires secure authentication, a database, server-side subscription enforcement, backups, moderation, privacy/legal review, deployment/HTTPS, and usage/cost controls. This package does not claim those services have already been configured.
- The app currently uses the existing browser-side persistence for some tools; it does not yet sync all data between devices.
- Do not deploy with debug mode or expose your .env file/API key.
