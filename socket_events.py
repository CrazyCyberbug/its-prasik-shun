# this is the room aware version. 


from flask import request
from flask_socketio import join_room, leave_room, send, emit, close_room
import io
import wave
import os
import uuid
import torchaudio
from datetime import datetime, timedelta
from database import room_sessions_collection, chat_history_collection, users_collection
from utils.model import pdf_translation
from utils.test import text_to_text_translation
# from final_server2 import transcribe_speech, transcribe_text_only
import torch
# english transcription = ""
import base64
import time

# --------------------------UNCOMMENT LINE 378 to 395---------------------------------
import weasyprint
from flask_socketio import SocketIO
from pprint import pprint
from io import BytesIO

import numpy as np
import threading
import mimetypes


# pip install fpdf
# ============================================
import os
import soundfile as sf
from datetime import datetime
# ===========================================

# ----------------------------------------------------------
AUDIO_DIR = "audio_files"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)


import requests
import json
import logging

# Configure logging
# logging.basicConfig(
#     filename='app.log',   # Log file name
#     level=logging.INFO,    # Logging level
#     format='%(asctime)s - %(levelname)s - %(message)s'  # Format of log messages
# )

# Writing logs





instructor_language = None
is_recording = True

rooms = {}
frames = []
all_transcriptions = []
user_sockets = {}
user_session_details = {}
old_len = 0
room_chats = {} 
chat_history = {}
audio_buffer = np.array([], np.float32)


sampling_rate = 16000
batch_size = sampling_rate * 3
keep_samples = int(sampling_rate * 0.15)

now = None
languages = {
    "vietnamese": "vie",
    "kazakh": "kaz",
    "russian": "rus",
    "french": "fra",
    "bengali": "ben",
    "arabic": "arb",
    "english": "eng",
    "hindi": "hin",
    "telugu": "tel",
}

class AudioConfig:
    SAMPLERATE = 16000
    CHANNELS = 1

class ServerConfig:
    server_host = "127.0.0.1"
    server_port = "8090"
    server_url = f"http://{server_host}:{server_port}/speech-translation"
    headers = {"Content-Type": "application/json"}

def get_unique_words_percentage(sent):
    # lower
    sent = sent.lower()
    
    # remove punctuation
    sent = sent.replace(".", "")
    sent = sent.replace(",", "")

    words = sent.split()
    unique_words = list(set(words))
    unique_percent = len(unique_words)/max(len(words), 1) 
    return unique_percent


# This is the previously working version of the Audio Handler compatible with one 
# class AudioHandler:
#     def __init__(self):
#         self.threads = {}
#         self.audio_buffers = {}
#         self.buffer_locks = {}

#         self.stop_thread_flag = False
#         self.batch_size = AudioConfig.SAMPLERATE * 1
#         self.socketio = None
        
#         self.current_speaker_id = None
#         self.language_mapping = None
#         self.session_deails = None
#         self.user_count = 0 

#     def update_language_mappings(self):
#         self.language_mapping = {lang_code: [] for lang_code in list(languages.values())}
#         print(f"language_mapping1:{self.language_mapping}")            
#         recipients = user_session_details
#         for recipient_id, details in recipients.items():
#                 tgt_lang = details.get("language", "Unknown")
#                 tgt_lang_code = languages[tgt_lang]
#                 self.language_mapping[tgt_lang_code].append(recipient_id)
        
#         print(f"language_mapping:{self.language_mapping}")            

#     def process_audio(self, data):
#         return data
    
#     def add_audio_to_buffer(self, user_id, audio):
#         if user_id not in self.buffer_locks:
#             self.buffer_locks[user_id] = threading.Lock()
#             self.audio_buffers[user_id] = np.array([], np.float32)
            
#         tensor_samples = torch.tensor(audio, dtype=torch.float32)                    
#         audio =  torchaudio.functional.resample(tensor_samples, orig_freq=48000, new_freq=16_000).cpu().numpy()
#         with self.buffer_locks[user_id]:
#             if len(self.language_mapping[user_id]) > 0:            
#                 self.audio_buffers[user_id] = np.append(self.audio_buffers[user_id], audio)
#             # print(f"User {user_id} now has {len(self.audio_buffers[user_id])} samples.")

#     def process_inference(self, user_id, tgt_lang, direct = False):
#         global user_sockets    
               
#         try:
#             print(f"Starting inference thread for {user_id}")
#             while not self.stop_thread_flag:
#                 samples = np.array([], np.float32)
                
#                 with self.buffer_locks[user_id]:
                    
#                     if len(self.audio_buffers[user_id]) > self.batch_size:
#                         samples = self.audio_buffers[user_id][:16000]
#                         self.audio_buffers[user_id] = self.audio_buffers[user_id][16000:]
#                         print(f"{user_id} buffers still contains {len(self.audio_buffers[user_id])/16000} of unprocessed audio.")
                    
#                 recipient_ids = [recipient_id for  recipient_id in self.language_mapping.get(user_id)]

#                 if len(samples) > 0:
#                     start = time.time()
#                     response = send_server_request(user_id, samples, tgt_lang)
#                     print(f'server_responded in {time.time() - start} secs.')
#                     if response:
#                         try:
#                             transcriptions = eval(response.text).get('transcriptions', "")
#                             if transcriptions:
#                                 text, audio = eval(transcriptions)
#                                 print(f"[translations]: {text}")
#                                 if text[0] != "[" and get_unique_words_percentage(text) > 0.4:    
#                                     for recipient_id in recipient_ids:
#                                         target_socket_id = user_sockets.get(recipient_id)                                
#                                         try:
#                                             # emitting audio.
#                                             audio_np = np.array(audio, dtype=np.float32)
#                                             sf.write(f"{user_id}_translated_audio.wav", audio_np, 16000)
#                                             if recipient_id != self.current_speaker_id:                                        
#                                                 with open(f"{user_id}_translated_audio.wav", "rb") as f:
#                                                     self.socketio.emit(
#                                                         "receive_audio",
#                                                         {"audio": f.read()},
#                                                         room=target_socket_id
#                                                     )
                                            
#                                             # emitting text.
#                                             self.socketio.emit(
#                                                 "transcription",
#                                                 {"english": "", "translated": text, "sender_user_id":self.current_speaker_id},
#                                                 to=target_socket_id,
#                                             )

#                                         except Exception as f:
#                                             print(f"emit is the problem, {f}")
                                    
#                         except Exception as e:
#                             print(f"Failed to decode response JSON: {e}")
                            
#         except Exception as e:
#             print(f"Exception in process_inference for {user_id}: {e}")
               
#     def handle_audio(self, data, socketio, room_data=None):
#         if not self.socketio:
#             self.socketio = socketio

#         audio = data

#         if not self.threads:
#             self.start_threads(start_up=True)

#         for lang_code in self.language_mapping:
#             self.add_audio_to_buffer(lang_code, audio)

#     def start_threads(self, start_up=False):
#         self.stop_thread_flag = False

#         if start_up:
#             self.update_language_mappings()
#             for lang_code in self.language_mapping:
#                 if len(self.language_mapping[lang_code]) > 0:
#                     send_server_request(lang_code, [], "eng", "true")
#                     print(f"Instantiating thread for {lang_code}.")
                    
#                     # Ensure buffers and locks are initialized
#                     self.audio_buffers[lang_code] = np.array([], np.float32)
#                     self.buffer_locks[lang_code] = threading.Lock()

#                     # Create and start the thread
#                     thread = threading.Thread(target=self.process_inference, args=(lang_code, lang_code))
#                     thread.daemon = True  # Set as daemon thread
#                     self.threads[lang_code] = thread
#                     print(f"Starting thread for user {lang_code}")
#                     thread.start()

#     def stop_threads(self):
#         self.stop_thread_flag = True
#         # clearing old buffers for the users on server side.
#         recipients = user_session_details
#         for recipient_id, details in recipients.items():
#             print(f"resetting buffers for {recipient_id}")
#             send_server_request(recipient_id, [], "eng", "true")

#         for thread in self.threads.values():
#             thread.join()

#         self.threads.clear()
#         self.audio_buffers.clear()

# room aware audio handler
# class AudioHandler:
#     def __init__(self):
#         # Room-specific data structures
#         self.threads = {}  # {room_id: {lang_code: thread}}
#         self.audio_buffers = {}  # {room_id: {lang_code: buffer}}
#         self.buffer_locks = {}  # {room_id: {lang_code: lock}}
#         self.language_mapping = {}  # {room_id: {lang_code: [user_ids]}}
        
#         self.stop_thread_flags = {}  # {room_id: bool}
#         self.batch_size = AudioConfig.SAMPLERATE * 1
#         self.socketio = None
        
#         self.current_speaker_id = {}
#         self.user_counts = {}  # {room_id: count}

#     def get_room_user_id(self, room_id, user_id):
#         """Generate combined room and user identifier."""
#         return f"{room_id}_{user_id}"

#     def update_language_mappings(self, room_id):
#         if room_id not in self.language_mapping:
#             self.language_mapping[room_id] = {}
        
#         self.language_mapping[room_id] = {lang_code: [] for lang_code in list(languages.values())}
#         print(f"language_mapping1 for room {room_id}: {self.language_mapping[room_id]}")
        
#         # Filter users by room_id
#         room_users = {uid: details for uid, details in user_session_details.items() 
#                      if details.get("room_id") == room_id}
        
#         for recipient_id, details in room_users.items():
#             tgt_lang = details.get("language", "Unknown")
#             tgt_lang_code = languages[tgt_lang]
#             self.language_mapping[room_id][tgt_lang_code].append(recipient_id)
        
#         print(f"language_mapping for room {room_id}: {self.language_mapping[room_id]}")

#     def add_audio_to_buffer(self, room_id, lang_code, audio):
#         if room_id not in self.buffer_locks:
#             self.buffer_locks[room_id] = {}
#             self.audio_buffers[room_id] = {}
            
#         if lang_code not in self.buffer_locks[room_id]:
#             self.buffer_locks[room_id][lang_code] = threading.Lock()
#             self.audio_buffers[room_id][lang_code] = np.array([], np.float32)
            
#         tensor_samples = torch.tensor(audio, dtype=torch.float32)
#         audio = torchaudio.functional.resample(tensor_samples, orig_freq=48000, new_freq=16_000).cpu().numpy()
        
#         with self.buffer_locks[room_id][lang_code]:
#             if len(self.language_mapping[room_id][lang_code]) > 0:
#                 self.audio_buffers[room_id][lang_code] = np.append(self.audio_buffers[room_id][lang_code], audio)

#     def process_inference(self, room_id, lang_code, tgt_lang):
#         global user_sockets
        
#         try:
#             print(f"Starting inference thread for room {room_id}, language {lang_code}")
#             while not self.stop_thread_flags.get(room_id, False):
#                 samples = np.array([], np.float32)
                
#                 with self.buffer_locks[room_id][lang_code]:
#                     if len(self.audio_buffers[room_id][lang_code]) > self.batch_size:
#                         samples = self.audio_buffers[room_id][lang_code][:16000]
#                         self.audio_buffers[room_id][lang_code] = self.audio_buffers[room_id][lang_code][16000:]
                
#                 recipient_ids = self.language_mapping[room_id].get(lang_code, [])
#                 recipient_usernames = [user_session_details.get(recipient_id)["username"] for recipient_id in recipient_ids]
#                 # logging.info(f"{room_id} has the recipients {recipient_usernames}") 
                
                
#                 if len(samples) > 0:
#                     start = time.time()
#                     room_user_id = self.get_room_user_id(room_id, lang_code)
#                     response = send_server_request(room_user_id, samples, tgt_lang)
#                     print(f'server_responded in {time.time() - start} secs.')
                    
#                     if response:
#                         try:
#                             transcriptions = eval(response.text).get('transcriptions', "")
#                             if transcriptions:
#                                 text, audio = eval(transcriptions)
#                                 print(f"[translations]: {text}")
#                                 if text[0] != "[" and get_unique_words_percentage(text) > 0.4:
#                                     for recipient_id in recipient_ids:
#                                         target_socket_id = user_sockets.get(recipient_id)
#                                         try:
#                                             # Emit audio
#                                             audio_np = np.array(audio, dtype=np.float32)
#                                             audio_file = f"{room_id}_{lang_code}_translated_audio.wav"
#                                             sf.write(audio_file, audio_np, 16000)
                                            
#                                             if recipient_id != self.current_speaker_id.get(room_id, "unknown"):
#                                                 print(f"sending to {room_id}")
#                                                 with open(audio_file, "rb") as f:
#                                                     self.socketio.emit(
#                                                         "receive_audio",
#                                                         {"audio": f.read()},
#                                                         room=target_socket_id
#                                                     )
                                            
#                                             # Emit text
#                                             print(f"sending to {room_id}")
#                                             self.socketio.emit(
#                                                 "transcription",
#                                                 {
#                                                     "english": "",
#                                                     "translated": text,
#                                                     "sender_user_id": self.current_speaker_id.get(room_id)
#                                                 },
#                                                 to=target_socket_id,
#                                             )
#                                         except Exception as f:
#                                             print(f"Emit error: {f}")
#                         except Exception as e:
#                             print(f"JSON decode error: {e}")
#         except Exception as e:
#             print(f"Exception in process_inference for room {room_id}, language {lang_code}: {e}")

#     def handle_audio(self, data, socketio, room_id):
#         if not self.socketio:
#             self.socketio = socketio

#         if room_id not in self.threads:
#             self.start_threads(room_id, start_up=True)

#         for lang_code in self.language_mapping.get(room_id, {}):
#             self.add_audio_to_buffer(room_id, lang_code, data)

#     def start_threads(self, room_id, start_up=False):
#         self.stop_thread_flags[room_id] = False

#         if start_up:
#             self.update_language_mappings(room_id)
            
#             if room_id not in self.threads:
#                 self.threads[room_id] = {}
            
#             for lang_code in self.language_mapping[room_id]:
#                 if len(self.language_mapping[room_id][lang_code]) > 0:
#                     room_user_id = self.get_room_user_id(room_id, lang_code)
#                     send_server_request(room_user_id, [], "eng", "true")
#                     print(f"Instantiating thread for room {room_id}, language {lang_code}")
                    
#                     # Initialize room-specific buffers and locks
#                     if room_id not in self.audio_buffers:
#                         self.audio_buffers[room_id] = {}
#                         self.buffer_locks[room_id] = {}
                    
#                     self.audio_buffers[room_id][lang_code] = np.array([], np.float32)
#                     self.buffer_locks[room_id][lang_code] = threading.Lock()

#                     # Create and start the thread
#                     thread = threading.Thread(
#                         target=self.process_inference,
#                         args=(room_id, lang_code, lang_code)
#                     )
#                     thread.daemon = True
#                     self.threads[room_id][lang_code] = thread
#                     print(f"Starting thread for room {room_id}, language {lang_code}")
#                     thread.start()

#     def stop_threads(self, room_id=None):
#         if room_id:
#             # Stop threads for specific room
#             self.stop_thread_flags[room_id] = True
#             room_users = {uid: details for uid, details in user_session_details.items() 
#                          if details.get("room_id") == room_id}
            
#             for recipient_id in room_users:
#                 print(f"Resetting buffers for {recipient_id} in room {room_id}")
#                 room_user_id = self.get_room_user_id(room_id, recipient_id)
#                 send_server_request(room_user_id, [], "eng", "true")

#             if room_id in self.threads:
#                 for thread in self.threads[room_id].values():
#                     thread.join()
#                 del self.threads[room_id]
#                 del self.audio_buffers[room_id]
#                 del self.buffer_locks[room_id]
#                 del self.language_mapping[room_id]
#         else:
#             # Stop all threads across all rooms
#             for room_id in self.threads.keys():
#                 self.stop_threads(room_id)


# audio handler that works till buffers are empty
class AudioHandler:
    def __init__(self):
        # Room-specific data structures
        self.threads = {}  # {room_id: {lang_code: thread}}
        self.audio_buffers = {}  # {room_id: {lang_code: buffer}}
        self.buffer_locks = {}  # {room_id: {lang_code: lock}}
        self.language_mapping = {}  # {room_id: {lang_code: [user_ids]}}
        
        self.stop_thread_flags = {}  # {room_id: bool}
        self.processing_complete = {}  # {room_id: {lang_code: Event}}
        self.batch_size = AudioConfig.SAMPLERATE * 1
        self.socketio = None
        
        self.current_speaker_id = {}
        self.user_counts = {}  # {room_id: count}
        self.shutdown_in_progress = {}  # {room_id: bool}

    def handle_audio(self, data, socketio, room_id):
        """Handle incoming audio data for a specific room."""
        if not self.socketio:
            self.socketio = socketio

        # Don't accept new audio if shutdown is in progress
        if self.shutdown_in_progress.get(room_id, False):
            return

        # Start threads if not already running for this room
        if room_id not in self.threads:
            self.start_threads(room_id, start_up=True)

        # Add audio data to buffers for each language in the room
        for lang_code in self.language_mapping.get(room_id, {}):
            self.add_audio_to_buffer(room_id, lang_code, data)

    def process_inference(self, room_id, lang_code, tgt_lang):
        global user_sockets
        
        try:
            print(f"Starting inference thread for room {room_id}, language {lang_code}")
            while True:
                # Check if we should stop and if there's no more audio
                if (self.stop_thread_flags.get(room_id, False) and 
                    not self.has_remaining_audio(room_id, lang_code)):
                    print(f"Stopping inference for room {room_id}, language {lang_code}")
                    break

                samples = np.array([], np.float32)
                
                with self.buffer_locks[room_id][lang_code]:
                    if len(self.audio_buffers[room_id][lang_code]) > 0:
                        # Process in smaller chunks to ensure smooth handling
                        chunk_size = min(self.batch_size, len(self.audio_buffers[room_id][lang_code]))
                        samples = self.audio_buffers[room_id][lang_code][:chunk_size]
                        self.audio_buffers[room_id][lang_code] = self.audio_buffers[room_id][lang_code][chunk_size:]
                
                if len(samples) > 0:
                    recipient_ids = self.language_mapping[room_id].get(lang_code, [])
                    room_user_id = self.get_room_user_id(room_id, lang_code)
                    
                    # Process the audio chunk
                    response = send_server_request(room_user_id, samples, tgt_lang)
                    if response:
                        self.handle_inference_response(response, room_id, lang_code, recipient_ids)
                        
                        # Small delay to prevent overwhelming the system
                        time.sleep(0.01)
                else:
                    # Smaller sleep when no audio to process
                    time.sleep(0.05)
            
            # Signal completion before exiting
            if room_id in self.processing_complete and lang_code in self.processing_complete[room_id]:
                print(f"Setting completion event for room {room_id}, language {lang_code}")
                self.processing_complete[room_id][lang_code].set()
                
        except Exception as e:
            print(f"Exception in process_inference for room {room_id}, language {lang_code}: {e}")
            if room_id in self.processing_complete and lang_code in self.processing_complete[room_id]:
                self.processing_complete[room_id][lang_code].set()

    def stop_threads(self, room_id=None):
        """Stop threads gracefully, ensuring all buffered audio is processed."""
        if room_id:
            print(f"Initiating shutdown for room {room_id}")
            for lang_code, buffer in self.audio_buffers[room_id].items():
                print(f"{lang_code} has {len(buffer)/16000} seconds of unprocessed audio during shut down")
            # Mark shutdown in progress
            self.shutdown_in_progress[room_id] = True
            
            # Set stop flag for the room
            self.stop_thread_flags[room_id] = True          
            
            
            # Initialize completion events for each language
            self.processing_complete[room_id] = {
                lang_code: threading.Event()
                for lang_code in self.threads.get(room_id, {}).keys()
            }
            
            # Wait for all threads to finish processing remaining audio
            print(f"Waiting for processing completion in room {room_id}")
            for lang_code, event in self.processing_complete[room_id].items():
                # Longer timeout to ensure processing completes
                if not event.wait(timeout=10.0):
                    print(f"Warning: Timeout waiting for {lang_code} processing in room {room_id}")
                print(f"All process have completed. Initializing shutdown.")  
            
            for lang_code, buffer in self.audio_buffers[room_id].items():
                print(f"{lang_code} has {len(buffer)/16000} seconds of unprocessed audio during shut down")
                        
            # Clean up room resources
            print(f"Cleaning up resources for room {room_id}")
            room_users = {uid: details for uid, details in user_session_details.items() 
                         if details.get("room_id") == room_id}
            
            for recipient_id in room_users:
                room_user_id = self.get_room_user_id(room_id, recipient_id)
                send_server_request(room_user_id, [], "eng", "true")

            # Clean up room-specific data structures
            if room_id in self.threads:
                for thread in self.threads[room_id].values():
                    thread.join(timeout=2.0)
                
                del self.threads[room_id]
                del self.audio_buffers[room_id]
                del self.buffer_locks[room_id]
                del self.language_mapping[room_id]
                del self.processing_complete[room_id]
                del self.shutdown_in_progress[room_id]
                
            print(f"Shutdown complete for room {room_id}")
        else:
            # Stop all threads across all rooms
            for room_id in list(self.threads.keys()):
                self.stop_threads(room_id)

    # [Previous helper methods remain unchanged]
    def get_room_user_id(self, room_id, user_id):
        return f"{room_id}_{user_id}"

    def has_remaining_audio(self, room_id, lang_code):
        with self.buffer_locks[room_id][lang_code]:
            return len(self.audio_buffers[room_id][lang_code]) > 0

    def add_audio_to_buffer(self, room_id, lang_code, audio):
        if room_id not in self.buffer_locks:
            self.buffer_locks[room_id] = {}
            self.audio_buffers[room_id] = {}
            
        if lang_code not in self.buffer_locks[room_id]:
            self.buffer_locks[room_id][lang_code] = threading.Lock()
            self.audio_buffers[room_id][lang_code] = np.array([], np.float32)
            
        tensor_samples = torch.tensor(audio, dtype=torch.float32)
        audio = torchaudio.functional.resample(tensor_samples, orig_freq=48000, new_freq=16_000).cpu().numpy()
        
        with self.buffer_locks[room_id][lang_code]:
            if len(self.language_mapping[room_id][lang_code]) > 0:
                self.audio_buffers[room_id][lang_code] = np.append(self.audio_buffers[room_id][lang_code], audio)

    def handle_inference_response(self, response, room_id, lang_code, recipient_ids):
        try:
            transcriptions = eval(response.text).get('transcriptions', "")
            if transcriptions:
                text, audio = eval(transcriptions)
                if text[0] != "[" and get_unique_words_percentage(text) > 0.4:
                    self.broadcast_results(room_id, lang_code, text, audio, recipient_ids)
        except Exception as e:
            print(f"Error handling inference response: {e}")

    def broadcast_results(self, room_id, lang_code, text, audio, recipient_ids):
        for recipient_id in recipient_ids:
            target_socket_id = user_sockets.get(recipient_id)
            try:
                if recipient_id != self.current_speaker_id.get(room_id, "unknown"):
                    # Broadcast audio
                    audio_np = np.array(audio, dtype=np.float32)
                    audio_file = f"{room_id}_{lang_code}_translated_audio.wav"
                    sf.write(audio_file, audio_np, 16000)
                    
                    with open(audio_file, "rb") as f:
                        self.socketio.emit(
                            "receive_audio",
                            {"audio": f.read()},
                            room=target_socket_id
                        )
                
                # Broadcast text
                self.socketio.emit(
                    "transcription",
                    {
                        "english": "",
                        "translated": text,
                        "sender_user_id": self.current_speaker_id.get(room_id)
                    },
                    to=target_socket_id,
                )
            except Exception as e:
                print(f"Broadcast error: {e}")
                
            sender_id = self.current_speaker_id.get(room_id)
            sender_user_details = user_session_details.get(sender_id)
            sender_code = languages.get(sender_user_details['language'], "eng")
            print(sender_user_details,".............//..............")
            
                
            transcription_data = {
                    "translated_text": text,
                    "src_language_code": sender_code,
                    "target_language_code": lang_code,
                    "timestamp": datetime.utcnow()
                }
            
            global instructor_language
            
            if instructor_language is None:
                for user_id, user_details in user_session_details.items():
                    if user_details['role'] == 'instructor':
                        instructor_language = user_details['language']
                        instructor_lang = languages.get(instructor_language, "eng")
                        
            else:
                instructor_lang = instructor_language                     
            
            
            room_sessions_collection.update_one(
                {"room": room_id, "user_id": self.current_speaker_id.get(room_id, ""),"instructor_lang": instructor_lang},
                {"$push": {"transcriptions": transcription_data}},
                upsert=True
            )

    def update_language_mappings(self, room_id):
        if room_id not in self.language_mapping:
            self.language_mapping[room_id] = {}
        
        self.language_mapping[room_id] = {lang_code: [] for lang_code in list(languages.values())}
        
        room_users = {uid: details for uid, details in user_session_details.items() 
                     if details.get("room_id") == room_id}
        
        for recipient_id, details in room_users.items():
            tgt_lang = details.get("language", "Unknown")
            tgt_lang_code = languages[tgt_lang]
            self.language_mapping[room_id][tgt_lang_code].append(recipient_id)

    def start_threads(self, room_id, start_up=False):
        self.stop_thread_flags[room_id] = False
        self.shutdown_in_progress[room_id] = False

        if start_up:
            self.update_language_mappings(room_id)
            
            if room_id not in self.threads:
                self.threads[room_id] = {}
            
            for lang_code in self.language_mapping[room_id]:
                if len(self.language_mapping[room_id][lang_code]) > 0:
                    room_user_id = self.get_room_user_id(room_id, lang_code)
                    send_server_request(room_user_id, [], "eng", "true")
                    print(f"Instantiating thread for room {room_id}, language {lang_code}")
                    
                    # Initialize room-specific buffers and locks
                    if room_id not in self.audio_buffers:
                        self.audio_buffers[room_id] = {}
                        self.buffer_locks[room_id] = {}
                    
                    self.audio_buffers[room_id][lang_code] = np.array([], np.float32)
                    self.buffer_locks[room_id][lang_code] = threading.Lock()

                    # Create and start the thread
                    thread = threading.Thread(
                        target=self.process_inference,
                        args=(room_id, lang_code, lang_code)
                    )
                    thread.daemon = True
                    self.threads[room_id][lang_code] = thread
                    print(f"Starting thread for room {room_id}, language {lang_code}")
                    thread.start()

handler = AudioHandler()

def send_server_request(user_id,  audio, tgt_lang, reset_buffers = "false"):
    try:

        if  reset_buffers == "true":
            response = requests.post(f"http://{ServerConfig.server_host}:{ServerConfig.server_port}/cleanup/{user_id}")
            return response


        else:
            payload = {
                "client_id": user_id,
                "audio_data": audio.tolist(),
                "sampling_rate": AudioConfig.SAMPLERATE,
                "tgt_lang": tgt_lang,
                "reset_buffers": reset_buffers
            }
            
            response = requests.post(
                ServerConfig.server_url,
                data=json.dumps(payload),
                headers=ServerConfig.headers
            )
            
            
            if response.status_code == 200:
                # print(f"server responded with {response.text}")
                return response
            
        print(f"Request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Server request error: {e}")
    return None

def handle_response(response):
    if response is not None:
        try:
            transcriptions = eval(response.text).get('transcriptions', "")
            # print(f"transcripts:{transcriptions} of type {type(transcriptions)}")
                                    
            if transcriptions:
                text = transcriptions

                if text[0] != "[":
                    print(f"Transcribed Text: {text}")
                    # all_transcriptions.append(text)
                return text

        except Exception as e:
            print(f"Failed to decode response JSON: {e}")

    return None

def register_socket_handlers(socketio):
    @socketio.on("checkRoom")
    def check_room(data):
        room_id = data["roomId"]
        role = data.get("role")

        # Check if the room exists in the `rooms` dictionary
        room_exists = room_id in rooms

        # Allow instructors to create new rooms
        if not room_exists and role == "instructor":
            rooms[room_id] = []  # Initialize the room
            room_exists = True

        # Respond back to the frontend
        return {"exists": room_exists}

    @socketio.on("join")
    def on_join(data):
        room = data["room"]
        user_details = data["userDetails"]
        user_role = user_details.get("role")
        
         # Check if the room exists, and handle based on the user's role
        if room not in rooms:
            if user_role != "instructor":
                socketio.emit("join-error", {"error": "Room not created by an instructor."}, to=request.sid)
                return
            else:
                rooms[room] = []  # Instructors can create the room


        print("00000000000000000000 : ", user_details)

        # user_id = request.sid  # Unique Socket ID for the user
        user_id = user_details["user_id"]
        user_details['room_id'] = room
        user_sockets[user_id] = request.sid
        user_session_details[user_id] = user_details

        # Add user to room's user list
        # if room not in rooms:
        #     rooms[room] = []

        # Check if the user is already in the room
        if user_id not in rooms[room]:
            rooms[room].append(user_id)
        # join_room(room)
        # emit('room_update', {'users': rooms[room]}, room=room)
        # rooms[room].append(user_id)

        join_room(room)
        send(f"User {user_id} has entered the room {room}", to=room)

        # Print number of users in the room
        print(f"Room {room} has {len(rooms[room])} users: {rooms[room]}")

        # Prepare user details for all users in the room
        room_user_details = [user_session_details[uid] for uid in rooms[room]]

        print("all user details : ", room_user_details)
        # Send the updated list of users and their details to everyone in the room
        socketio.emit(
            "updateUsers",
            {"users": rooms[room], "userDetails": room_user_details},
            room=room,
        )
        print("room id and request id :", room, request.sid)
        socketio.emit('new-participant', {
            'socketId': request.sid,
        }, room=room)
        print("Emitting 'new-participant' with socket ID:", request.sid)
          
# -------------------------------------main code ----------------------------------------
    # @socketio.on("join")
    # def on_join(data):
    #     room = data["room"]
    #     user_details = data["userDetails"]

    #     print("00000000000000000000 : ", user_details)
    #     print(room,".......................joining_room.......................")
    #     # user_id = request.sid  # Unique Socket ID for the user
    #     user_id = user_details["user_id"]
    #     user_sockets[user_id] = request.sid
    #     user_session_details[user_id] = user_details
    #     role = user_details["role"]


    #     # Add user to room's user list
    #     if room not in rooms:
    #         print(rooms,"....inside room...........")
    #         rooms[room] = []

    #     # Check if the user is already in the room
    #     if user_id not in rooms[room]:
    #         print("...............already in room.............")
    #         rooms[room].append(user_id)
    #     # join_room(room)
    #     # emit('room_update', {'users': rooms[room]}, room=room)
    #     # rooms[room].append(user_id)

    #     join_room(room)
    #     print(join_room,"..........join_room..........")
    #     send(f"User {user_id} has entered the room {room}", to=room)

    #     # Print number of users in the room
    #     print(f"Room {room} has {len(rooms[room])} users: {rooms[room]}")

    #     # Prepare user details for all users in the room
    #     room_user_details = [user_session_details[uid] for uid in rooms[room]]

    #     print("all user details : ", room_user_details)
    #     # Send the updated list of users and their details to everyone in the room
    #     socketio.emit(
    #         "updateUsers",
    #         {"users": rooms[room], "userDetails": room_user_details},
    #         room=room,
    #     )



    #     print("room id and request id :", room, request.sid)
    #     socketio.emit('new-participant', {
    #         'socketId': request.sid,
    #     }, room=room)
    #     print("Emitting 'new-participant' with socket ID:", request.sid)
# ------------------------------------------------------original down work-----------------------------
    # @socketio.on("join")
    # def on_join(data):
    #     room = data["room"]
    #     user_details = data["userDetails"]

    #     print("00000000000000000000 : ", user_details)
    #     print(room,".......................joining_room.......................")
    #     # user_id = request.sid  # Unique Socket ID for the user
    #     user_id = user_details["user_id"]
    #     user_sockets[user_id] = request.sid
    #     user_session_details[user_id] = user_details
    #     role = user_details["role"]
    #     print(f"{user_id} attempting to join room {room} with role {role}.")

    #     if role == "instructor":
    #         if room not in rooms or len(rooms[room]) == 0 or len(rooms[room]) > 0:
    #             print(rooms,"....inside room...........")
    #             rooms[room] = []
    #         # else:
    #         if user_id not in rooms[room]:
    #             rooms[room].append(user_id)
    #             join_room(room)
    #             user_sockets[user_id] = request.sid
    #             user_session_details[user_id] = user_details
    #         print(f"Instructor {user_id} joined room {room}.")
    #         print(f"Room {room} now has {len(rooms[room])} users: {rooms[room]}.")
    #     elif role == "student":
    #         if room not in rooms:
    #             print(f"Student {user_id} attempted to join non-existent room {room}.")
    #             socketio.emit(
    #                 "errorMessage",
    #                 {"message": "The room has not been created by the instructor."},
    #                 room=request.sid,
    #             )
    #             return
    #         if user_id not in rooms[room]:
    #             rooms[room].append(user_id)
    #             join_room(room)
    #             user_sockets[user_id] = request.sid
    #             user_session_details[user_id] = user_details
    #         print(f"Student {user_id} joined room {room}.")
    #         print(f"Room {room} now has {len(rooms[room])} users: {rooms[room]}.")

    #     else:
    #         print(f"Unknown role {role} for user {user_id}.")
    #         socketio.emit(
    #             "errorMessage",
    #             {"message": "Unknown role. Access denied."},
    #             room=request.sid,
    #         )
    #         return
                
    #     join_room(room)
    #     print(join_room,"..........join_room..........")
    #     send(f"User {user_id} has entered the room {room}", to=room)

    #     # Print number of users in the room
    #     print(f"Room {room} has {len(rooms[room])} users: {rooms[room]}")

    #     # Prepare user details for all users in the room
    #     room_user_details = [user_session_details[uid] for uid in rooms[room]]

    #     print("all user details : ", room_user_details)
    #     # Send the updated list of users and their details to everyone in the room
    #     socketio.emit(
    #         "updateUsers",
    #         {"users": rooms[room], "userDetails": room_user_details},
    #         room=room,
    #     )

    #     print("room id and request id :", room, request.sid)
    #     socketio.emit('new-participant', {
    #         'socketId': request.sid,
    #     }, room=room)
    #     print("Emitting 'new-participant' with socket ID:", request.sid)
# ----------------------------------------last of working-----------------------------------
    # @socketio.on("join")
    # def on_join(data):
    #     room = data["room"]
    #     user_details = data["userDetails"]
    #     user_id = user_details["user_id"]
    #     role = user_details["role"]

    #     print(f"{user_id} attempting to join room {room} with role {role}.")

    #     # Check if the room exists
    #     if room not in rooms:
    #         if role == "instructor":
    #             # Instructor creates the room
    #             rooms[room] = [user_id]
    #             user_sockets[user_id] = request.sid
    #             user_session_details[user_id] = user_details
    #             join_room(room)
    #             print(f"Instructor {user_id} created and joined room {room}.")
    #             send(f"Room {room} created by instructor {user_id}.", to=room)
    #         else:
    #             # If a student tries to join a non-existent room
    #             print(f"Student {user_id} attempted to join non-existent room {room}.")
    #             socketio.emit("error", {"message": "Room has not been created by an instructor."}, to=request.sid)
    #             return
    #     else:
    #         # If the room exists, allow joining
    #         if role == "student":
    #             # Add student only if room exists
    #             if user_id not in rooms[room]:
    #                 rooms[room].append(user_id)
    #                 user_sockets[user_id] = request.sid
    #                 user_session_details[user_id] = user_details
    #                 join_room(room)
    #                 send(f"Student {user_id} has joined room {room}.", to=room)
    #                 print(f"Student {user_id} joined room {room}.")
    #             else:
    #                 print(f"Student {user_id} already in room {room}.")
    #         elif role == "instructor":
    #             # Prevent multiple instructors in the same room
    #             print(f"Instructor {user_id} attempted to join room {room} which already exists.")
    #             socketio.emit("error", {"message": "This room is already active. Multiple instructors are not allowed."}, to=request.sid)
    #             return

    #     # Prepare user details for all users in the room
    #     room_user_details = [user_session_details[uid] for uid in rooms[room]]

    #     # Send the updated user list to the room
    #     socketio.emit(
    #         "updateUsers",
    #         {"users": rooms[room], "userDetails": room_user_details},
    #         room=room,
    #     )
    #     print(f"Room {room} now has {len(rooms[room])} users: {rooms[room]}.")

    #     # Notify new participant
    #     socketio.emit("new-participant", {"socketId": request.sid}, room=room)


    @socketio.on('offer')
    def handle_offer(data):
        print("text2---------", data)
        room = data.get('room')
        offer = data.get('offer')
        target = data.get('target')

        print("text3------", room, offer, target)
        if not room or not offer or not target:
            print(f"Missing room ID {room}, offer {offer}, or target {target}")
            return "Missing room ID, offer, or target", 400
        
        print(f"Received offer for room {room} from {request.sid} to {target}")
        socketio.emit('offer', {
            'offer': offer,
            'socketId': request.sid
        }, room=target)

    # Handle WebRTC answer
    @socketio.on('answer')
    def handle_answer(data):
        print("text3-------------------------", data)
        room = data.get('room')
        answer = data.get('answer')
        target = data.get('target')

        print("text-4----------------", room,answer,target)
        if not room or not answer or not target:
            print(f"Missing room ID {room}, answer {answer}, or target {target}")
            return "Missing room ID, answer, or target", 400

        print(f"Received answer for room {room} from {request.sid} to {target}")
        socketio.emit('answer', {
            'answer': answer,
            'socketId': request.sid
        }, room=target)

    # Handle ICE candidates
    @socketio.on('ice-candidate')
    def handle_ice_candidate(data):
        print("entering into ICE CCCCCCCCCC", data)
        room = data.get('room')
        candidate = data.get('candidate')
        target = data.get('target')
        user_id = data.get('userId')
        print("printing test 1 -----------------------", room, candidate,target, user_id)
        if not room or not candidate or not target:
            print(f"Missing room ID {room}, candidate {candidate}, or target {target}")
            return "Missing room ID, ICE candidate, or target", 400

        print(f"Received ICE candidate for room {room} from {request.sid} to {target}")
        socketio.emit('ice-candidate', {
            'candidate': candidate,
            'socketId': request.sid
        }, room=target)

# ==============================================================================
    @socketio.on("screen-sharing-status")
    def screen_sharing_status(data):
        print('==========ssssssssssssssssssssssss=============')
        print(data)
        print('==========sssssssssssssssssssssssss=============')
        stop_screen=False
        if data == False:
            stop_screen=True
        
        # Respond back to the frontend
        socketio.emit('screenshare', stop_screen)

    @socketio.on("stop_screen")
    def stop_screen(data):
        print('==========eeeeeeeeeeeeeeeeeeeeee=============')
        print(data)
        print('==========eeeeeeeeeeeeeeeeeeeeee=============')
        stop_screen=False
        if data == False:
            stop_screen=True
        
        # Respond back to the frontend
        socketio.emit('stopscreen', stop_screen)
# ================================================================================================

#------------------------------------------------------------------------------------------    

        # Send the updated list of users to everyone in the room
        # socketio.emit("updateUsers", rooms[room], room=room)

    @socketio.on("leave")
    def on_leave(data):
        room = data["room"]
        user_id = data["user_id"]  # Unique Socket ID for the user
        print(rooms,"..............the rooms_data...............")
        # Remove user from the room's user list
        if room in rooms and user_id in rooms[room]:
            rooms[room].remove(user_id)
            print(rooms,"..............POP_DATA_REMOVED...............")

        leave_room(room)
        send(f"User {user_id} has left the room {room}", to=room)

        # Print number of users remaining in the room
        print(f"Room {room} has {len(rooms[room])} users: {rooms[room]}")
        # Remove the user from user_sockets when they leave
        user_sockets.pop(user_id, None)

        # Prepare user details for all users in the room
        room_user_details = [user_session_details[uid] for uid in rooms[room]]

        print("all user details : ", room_user_details)
        # Send the updated list of users and their details to everyone in the room
        socketio.emit(
            "updateUsers",
            {"users": rooms[room], "userDetails": room_user_details},
            room=room,
        )

        # Send the updated list of users to everyone in the room
        # socketio.emit("updateUsers", rooms[room], room=room)
# ---------------added for ON and OFF mic ---------------------------------------
    @socketio.on("toggle_mic")
    def toggle_mic(data):
        socketio.emit('update_mic',data)
# ---------------------------------------------------------------------------------

    @socketio.on("close_room")
    def on_close_room(data):
        room = data["room"]

        print(rooms,"..............the rooms_data...............")
        if room in rooms:
            send(f"The room {room} is now closed", to=room)
            close_room(room)
            # Clear the room's user list
            del rooms[room]
            print(rooms,"..............after pop....................")
        print(f"Room {room} has been closed.")

        socketio.emit("roomClosed",  to=room)

    @socketio.on("end_call")
    def on_end_call(data):
        room = data["room"]
        print(room,"...............end_call..........")
        instructor_id = data.get("user_id")  # Instructor's user ID

        if room in rooms:
            # Notify everyone in the room that the call is ending
            socketio.emit("roomClosed", {"message": "The call has been ended by the instructor."}, to=room)

            # Iterate through users in the room
            user_ids = rooms[room]
            for user_id in user_ids:
                user_socket = user_sockets.get(user_id)  # Get the user's socket ID
                if user_socket:
                    # Notify the individual user
                    socketio.emit(
                        "userDisconnected",
                        {"message": "The call has been ended by the instructor."},
                        room=user_socket
                    )
                    # Remove the user from the room
                    leave_room(room, sid=user_socket)

            # Clear chat history for the room
            chat_history.pop(room, None)
            print(rooms,"..............the rooms_data...............")
            # Clear the room data
            rooms.pop(room, None)
            print(rooms,"..............after pop....................")
            for user_id in user_ids:
                user_sockets.pop(user_id, None)
                user_session_details.pop(user_id, None)

            print(f"Room {room} ended by instructor {instructor_id}. All users disconnected.")
        else:
            print(f"Room {room} does not exist or has already been closed.")
    
    # @socketio.on("end_call")
    # def on_end_call(data):
    #     room = data["room"]
    #     instructor_id = data.get("user_id")  # Instructor's user ID

    #     if room in rooms:
    #         if instructor_id in rooms[room] and user_session_details[instructor_id]["role"] == "instructor":
    #             # Notify everyone in the room
    #             socketio.emit("roomClosed", {"message": "The call has been ended by the instructor."}, to=room)

    #             # Disconnect all users and clean up
    #             user_ids = rooms[room]
    #             for user_id in user_ids:
    #                 user_socket = user_sockets.get(user_id)
    #                 if user_socket:
    #                     socketio.emit("userDisconnected", {"message": "The call has been ended by the instructor."}, room=user_socket)
    #                     leave_room(room, sid=user_socket)

    #             # Clear room data
    #             rooms.pop(room, None)
    #             for user_id in user_ids:
    #                 user_sockets.pop(user_id, None)
    #                 user_session_details.pop(user_id, None)

    #             print(f"Room {room} ended by instructor {instructor_id}. All users disconnected.")
    #         else:
    #             print(f"User {instructor_id} is not authorized to end the call for room {room}.")
    #             socketio.emit("error", {"message": "You are not authorized to end this call."}, to=request.sid)
    #     else:
    #         print(f"Room {room} does not exist or has already been closed.")
    #         socketio.emit("error", {"message": "Room does not exist or has already been closed."}, to=request.sid)


    def convert_file_to_base64(file, file_name):
        """Convert binary file to Base64 with its MIME type."""
        base64_file = base64.b64encode(file).decode('utf-8')
        # Guess MIME type based on the file name or default to application/octet-stream
        mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        return f"data:{mime_type};base64,{base64_file}"
# =============================================================================================================================
    @socketio.on("message")
    def handle_message(data):
        print("Enter into message ------------------------------------------------yes")
        room = data["room"]
        message = data["message"]
        sender_user_id = data["user_id"]
        name = data["name"]

        file = data.get("file")  # This is expected to be the binary content of the file
        file_name = data.get("file_name", "unknown_file")  # Get the file name with extension
        timestamp = data.get("timestamp") 

        print("namaename000000000000000000",name)

        current_user_details = user_session_details.get(sender_user_id, {})
        username = current_user_details.get("username", "Unknown User")
        src_language = current_user_details.get("language", "en")

        # Prepare message data to be sent
        message_data = {
            "sender_username": username,
            "sender_id": sender_user_id,
            "is_private": False,
            "timestamp": timestamp, 
        }
        sender_message_data = {
            "sender_username": username,
            "sender_id": sender_user_id,
            "is_private": False,
            "timestamp": timestamp, 
        }
        receiver_message_data = {
            "sender_username": username,
            "sender_id": sender_user_id,
            "is_private": False,
            "timestamp": timestamp, 
        }
        if file:
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension == ".pdf":
               
                print(f"Processing PDF file: {file_name}")

                # Directory to store uploaded PDFs
                upload_dir = "uploaded_pdfs"
                # upload_dir = "uploaded_pdfs"output_doc
                os.makedirs(upload_dir, exist_ok=True)
                pdf_path = os.path.join(upload_dir, file_name)

                # Save the uploaded PDF
                with open(pdf_path, "wb") as pdf_file:
                    pdf_file.write(file)

                for user_id, details in user_session_details.items():
                    target_socket_id = user_sockets.get(user_id)
                    target_language = details.get("language", "en")

                    # If the current user is the sender, send the original PDF
                    if sender_user_id == user_id:
                        print(file,".............file.........")
                        print(file_name,".............file_name.........")
                        translated_file = convert_file_to_base64(file, file_name)
                        sender_message_data["file"] = translated_file
                        sender_message_data["file_name"] = file_name 
                        
                        
                        socketio.emit("message", sender_message_data, to=target_socket_id)
                    else:
                        # Process and translate the PDF for other users
                        translated_md_path = pdf_translation(pdf_path, src_language, target_language)
                        print(translated_md_path, ".............md_path...............")
                        if translated_md_path:
                            # Read the translated PDF file in binary mode
                            translated_pdf_path = translated_md_path.replace(".md", ".pdf")
                            if os.path.exists(translated_pdf_path):
                                with open(translated_pdf_path, "rb") as translated_pdf_file:
                                    translated_pdf_data = translated_pdf_file.read()

                                # Convert the translated PDF to base64
                                translated_file = convert_file_to_base64(translated_pdf_data, os.path.basename(translated_pdf_path))
                                receiver_message_data = {
                                    "file": translated_file,
                                    "file_name": os.path.basename(translated_pdf_path),
                                    "sender_username": username,
                                    "timestamp": timestamp, 
                                }
                                print("receiver_message_data",receiver_message_data)
                                # Send the translated PDF to the frontend
                                socketio.emit("message", receiver_message_data, to=target_socket_id)
                        
            else :
                # Convert the file to base64 with its MIME type
                translated_file = convert_file_to_base64(file, file_name)
                message_data["file"] = translated_file
                message_data["file_name"] = file_name 

                if message:
                    message_data["message"] = message
        elif message:
            message_data["message"] = message

        # Send the message with the necessary data
        send(message_data, to=room)

        # Translate and send the message to other users (excluding the sender)
        if message:
            for user_id, details in user_session_details.items():
                if sender_user_id == user_id:
                    continue

                target_socket_id = user_sockets.get(user_id)
                translated_text = text_to_text_translation(
                    message,
                    src_language,
                    details["language"],
                )

                print("translated text  : ", translated_text)
                # return translated_text
                socketio.emit(
                    "text-translation-completed", translated_text, to=target_socket_id
                )
                # =====added=
                socketio.emit(
                    "chat_history", translated_text, to=target_socket_id
                )
    @socketio.on("private_message")
    def handle_private_message(data):
        target_user = data["targetUser"]
        message = data["message"]
        user_id = data["user_id"]

        file = data.get("file")
        file_name = data.get("file_name", "unknown_file")
        timestamp = data.get("timestamp") 


        print("======================private_message===============================")
        # print("==============file_name========",file_name, "Received private message at:", timestamp)

        # Check if target user exists in the user_sockets mapping
        target_socket_id = user_sockets.get(target_user)
        sender_socket_id = user_sockets.get(user_id)
        target_user_details = user_session_details.get(target_user)
        current_user_details = user_session_details.get(user_id)
        target_username = target_user_details.get("username", "Unknown Target User")
        sender_username = current_user_details.get("username", "Unknown Sender")

        print("target_user_details : ", target_user_details, current_user_details)
        print(
            "target_user_details : ",
            target_user_details["language"],
            current_user_details["language"],
        )

        # Send private message to the target user only

        # Prepare the message payload
        private_message_data = {
            "message": message,
            "sender_username": sender_username,
            "receiver_username": target_username,
            "sender_id": user_id,
            "receiver_id": target_user,
            "is_private": True,
            "timestamp": timestamp,

        }
        sender_message_data = {
            "message": message,
            "sender_username": sender_username,
            "receiver_username": target_username,
            "sender_id": user_id,
            "receiver_id": target_user,
            "is_private": True,
            "timestamp": timestamp,

        }
        receiver_message_data = {
            "message": message,
            "sender_username": sender_username,
            "receiver_username": target_username,
            "sender_id": user_id,
            "receiver_id": target_user,
            "is_private": True,
            "timestamp": timestamp,

        }
        # Add the file if it exists
        if file:
            file_extension = os.path.splitext(file_name)[1].lower()
            # ------------------added-----
            if file_extension == ".pdf":
               
                print(f"Processing PDF file: {file_name}")

                # Directory to store uploaded PDFs
                upload_dir = "uploaded_pdfs"
                # upload_dir = "uploaded_pdfs"output_doc
                os.makedirs(upload_dir, exist_ok=True)
                pdf_path = os.path.join(upload_dir, file_name)

                # Save the uploaded PDF
                with open(pdf_path, "wb") as pdf_file:
                    pdf_file.write(file)
                src_language = current_user_details.get("language")
                target_language = target_user_details.get("language")
                if sender_socket_id:
                    translated_file = convert_file_to_base64(file, file_name)
                    # sender_message_data = {
                    #     "file": translated_file,
                    #     "file_name": file_name,
                    # }
                    sender_message_data["file"] = translated_file
                    sender_message_data["file_name"] = file_name 
                    
                    socketio.emit("message", sender_message_data, to=sender_socket_id)

                translated_md_path = pdf_translation(pdf_path, src_language, target_language)
                print(translated_md_path, ".............md_path...............")
                if translated_md_path:
                    # Read the translated PDF file in binary mode
                    translated_pdf_path = translated_md_path.replace(".md", ".pdf")
                    if os.path.exists(translated_pdf_path):
                        with open(translated_pdf_path, "rb") as translated_pdf_file:
                            translated_pdf_data = translated_pdf_file.read()

                        # Convert the translated PDF to base64
                        translated_file = convert_file_to_base64(translated_pdf_data, os.path.basename(translated_pdf_path))
                        receiver_message_data = {
                            "file": translated_file,
                            "file_name": os.path.basename(translated_pdf_path),
                            "sender_username": username,
                        }

                        # Send the translated PDF to the frontend
                        socketio.emit("message", receiver_message_data, to=target_socket_id)
                        # socketio.emit("message", private_message_data, to=sender_socket_id)

            # -----------added---------
            else:
                translated_file = convert_file_to_base64(file, file_name)  # Your existing utility function
                private_message_data["file"] = translated_file
                private_message_data["file_name"] = file_name

                print("Private message data being sent:", private_message_data)
        if target_socket_id:
            socketio.emit("message", private_message_data, to=target_socket_id)
            socketio.emit("message", private_message_data, to=sender_socket_id)

            # Translate the text part of the message if applicable
            if message:
                translated_text = text_to_text_translation(
                    message,
                    current_user_details.get("language"),
                    target_user_details.get("language"),
                )
                socketio.emit("text-translation-completed", translated_text, to=target_socket_id)
                socketio.emit("chat_history", translated_text, to=target_socket_id)

        else:
            # send(f"User {user_id}: {message}", to=data["room"])
            # Handle the case where the target user is not connected
            send(f"User {target_user} is not online", to=sender_socket_id)
   
    @socketio.on("translate-text")
    def handle_translate_text_to_local(data):
        """

        As of now for public messages we can't see the Translation, because below error.

        Need to Resolve this ERROR :
        The phrase "Already borrowed" indicates that the processor_m4t object is already in use in another thread,
          and you're trying to reuse it without releasing it first.

        """

        text = data["text"]
        sender_id = data["sender_id"]
        user_id = data["user_id"]
        # message_index = data["message_index"]

        sender_user_details = user_session_details.get(sender_id)
        current_user_details = user_session_details.get(user_id)

        # src_language_code = src_language[:3]

        # sender_language_code = sender_user_details["language"][:3]
        # current_user_language_code = current_user_details["language"][:3]
        sender_language_code = sender_user_details["language"]
        current_user_language_code = current_user_details["language"]

        print(
            "helooooooo master heart miss aayeyyyy : ",
            sender_language_code,
            current_user_language_code,
        )

        sender_socket_id = user_sockets.get(user_id)
        translated_text = text_to_text_translation(
            text, sender_language_code, current_user_language_code
        )

        print("translated text  : ", translated_text)
        # return translated_text
        socketio.emit(
            "text-translation-completed", translated_text, to=sender_socket_id
        )
        # =====added=
        socketio.emit(
            "chat_history", translated_text, to=sender_socket_id
        )
        # socketio.emit(
        #     "text-translation-completed", {"message_index": message_index, "translated_text": translated_text}, to=sender_socket_id
        # )
    

#----------------------------------New audio to text by Pavan kalyan-----------------------------------------------------------------------
    # @socketio.on("audio")
    # def handle_audio(data, room_data):
    #     global frames, is_recording  # Access global frames
    #     room = room_data["room"]
    #     user_id = room_data["user_id"]
    #     src_language = room_data["src_language"]
    #     sampling = room_data["sampleRate"]
    #     print(sampling,"...... The Audio sampling rate that is coming from the frontend............")
    #     translated_text = ""

    #     if not is_recording:
    #         return
        
    #     print(type(frames),"--------------------------------------------------------")
    #     if data:
    #         frames.append(data)
    #         print(type(frames),"....................................................")
    #         try:
    #             audio_data = b"".join(frames)  # Combine audio frames
    #             print(len(audio_data),"..................lenght...of_audio_data..................")
    #             # print(type(audio_data),";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
    #             # Load the audio data and resample if necessary
    #             waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))
    #             if sample_rate != 16000:
    #                 print("..................................Re-sampling...................................")
    #                 waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    #             # Calculate RMS energy to determine if the audio is significant
    #             rms_energy = np.sqrt(np.mean(waveform.numpy()**2))
    #             print(f"RMS Energy: {rms_energy}","-------------it should be more then 0.02------------------")

    #             # Skip transcription if audio does not meet the threshold
    #             MIN_RMS_THRESHOLD = 0.04  # Adjust based on your needs ------------quiet audio--------0.1 to 0.1, 0.7 to 1.0 indicates very loud audio.,   0.0 indicates complete silence
    #             if rms_energy < MIN_RMS_THRESHOLD:
    #                 print("Audio below threshold. Skipping transcription.")
    #                 return

    #             # Prepare input for the transcription model
    #             audio_inputs = processor_m4t(audios=waveform.squeeze().numpy(), return_tensors="pt").to(device)

    #             # Get English transcription
    #             src_language_code = "eng" if src_language.lower() == "english" else languages.get(src_language.lower(), None)
    #             output_tokens_eng = model_m4t.generate(**audio_inputs, tgt_lang=src_language_code, generate_speech=False)
    #             english_transcription = processor_m4t.decode(output_tokens_eng[0].tolist()[0], skip_special_tokens=True)

    #             print("English Transcription--", english_transcription)
    #             all_transcriptions.append(english_transcription)

    #             # Perform translations
    #             for user_id_receiver, details in user_session_details.items():
    #                 user_language = details.get("language", "Unknown")
    #                 target_socket_id = user_sockets.get(user_id_receiver)
    #                 if user_language.lower() == "french":
    #                     user_language_code = "fra"
    #                 elif user_language.lower() == "arabic":
    #                     user_language_code = "arb"
    #                 else:
    #                     user_language_code = user_language[:3]

    #                 # user_language_code = "fra" if user_language.lower() == "french" else user_language[:3]
    #                 # user_language_code = "arb" if user_language.lower() == "arabic" else user_language[:3]
    #                 print(user_language_code,".......................................pppp...............................")
    #                 if user_language_code != "kaz":
    #                     output_tokens_tel = model_m4t.generate(**audio_inputs, tgt_lang=user_language_code, generate_speech=False)
    #                     translated_text = processor_m4t.decode(output_tokens_tel[0].tolist()[0], skip_special_tokens=True)
    #                 elif user_language_code == "kaz":
    #                     output_tokens_tel = model.generate(**audio_inputs, tgt_lang=user_language_code, generate_speech=False)
    #                     translated_text = processor.decode(output_tokens_tel[0].tolist()[0], skip_special_tokens=True)

    #                 print("Translated Text from the model --", translated_text)

    #                 # ------------------speech to speech------------------
    #                 audio_array_from_audio = model.generate(
    #                     **audio_inputs, tgt_lang=user_language_code
    #                 )[0].cpu().numpy().squeeze()
                    
    #                 sf.write(f"{user_id_receiver}_translated_audio.wav", audio_array_from_audio, 16000)
                    
    #                 # Send translated audio back to user
    #                 with open(f"{user_id_receiver}_translated_audio.wav", "rb") as f:
    #                     if user_id != user_id_receiver:
    #                         socketio.emit(
    #                             "receive_audio",
    #                             {"audio": f.read()},
    #                             room=target_socket_id
    #                         )
    # #                 # ----------------------------------------------------
                    

    #                 # Send transcription to the user
    #                 emit("transcription", {
    #                     "english": english_transcription, 
    #                     "translated": translated_text, 
    #                     "sender_user_id": user_id
    #                 }, to=target_socket_id)

    #                 # Save transcription data to MongoDB
    #                 transcription_data = {
    #                     "english_transcription": english_transcription,
    #                     "translated_text": translated_text,
    #                     "src_language_code": src_language_code,
    #                     "target_language_code": user_language_code,
    #                     "timestamp": datetime.utcnow()
    #                 }

    #                 room_sessions_collection.update_one(
    #                     {"room": room, "user_id": user_id_receiver},
    #                     {"$push": {"transcriptions": transcription_data}},
    #                     upsert=True
    #                 )
    #                 # frames.clear()

    #         except Exception as e:
    #             print(f"Error processing audio data for transcription: {e}")

    # # Handle stop event to combine audio and save the file
    # @socketio.on("stop")
    # def handle_stop():
    #     is_recording = False
    #     if frames:
    #         unique_filename = str(uuid.uuid4())
    #         file_path = os.path.join(AUDIO_DIR, f"audio_{unique_filename}.wav")
    #         with wave.open(file_path, "wb") as wf:
    #             wf.setnchannels(1)
    #             wf.setsampwidth(2)
    #             wf.setframerate(44100)
    #             wf.writeframes(b"".join(frames))
    #     print("Now clearing all the frames to the Empty List")
    #     frames.clear()
    #     print("Now we will see the count of the frames :-----",len(frames))

    #     # Clear GPU memory
    #     print("Clearing GPU memory...")
    #     torch.cuda.empty_cache()
    #     print("GPU memory cleared.")


# -----------------------------------------old len included  by ananth ------------------------------------------------------------------
    # @socketio.on("audio")
    # def handle_audio(data, room_data):
    #     global frames, is_recording, old_len, audio_buffer  # Access global frames



    #     # Room details.
    #     room = room_data["room"]
    #     user_id = room_data["user_id"]
    #     src_language = room_data["src_language"]
    #     src_language_code = languages.get(src_language.lower(), None)
    #     sampling = room_data["sampleRate"]


    #     translated_text = ""

    #     if data and is_recording:

    #         # Load and resample data.
    #         frames.append(data)
    #         try:
    #             audio_data = b"".join(frames)  
    #             waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))
    #             if sample_rate != 16000:                  
    #                 waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    #             # Buffering incoming audio data.
    #             audio_array = waveform.numpy().flatten()
    #             chunk = audio_array[old_len:] 
    #             old_len = len(audio_array)
    #             audio_buffer = np.append(audio_buffer,  chunk)                
                
    #             # Audio config.
    #             sample_rate = 16000
    #             n_batch_samples = 5 * sample_rate
    #             keep_samples = int(sample_rate * 0.1)
                
    #             # Polling and extracting samples.
    #             if len( audio_buffer >= n_batch_samples):
    #                 samples = audio_buffer[:n_batch_samples]
    #                 audio_buffer = audio_buffer[n_batch_samples- keep_samples:]

    #             else:
    #                 return
                
                
    #             if is_silence(samples, threshold = 0.04):
    #                 return
    


    #             english_transcription = transcribe_text_only(input_array = samples, 
    #                                                          tgt_lang =  "eng")
                
    #             print("English Transcription--", english_transcription)
    #             all_transcriptions.append(english_transcription)     


    #             # Perform translations
    #             for user_id_receiver, details in user_session_details.items():

    #                 user_language_code = resolve_language_code(details)

    #                 text, audio = transcribe_speech(input_array= samples,
    #                                                 tgt_lang = user_language_code)
                    
    #                 translated_text = text
    #                 translated_audio = np.array(audio, dtype=np.float32)

                    
    #                 target_socket_id = user_sockets.get(user_id_receiver)
    #                 sf.write(f"{user_id_receiver}_translated_audio.wav", translated_audio, 16000)

    #                 with open(f"{user_id_receiver}_translated_audio.wav", "rb") as f:
    #                     if user_id != user_id_receiver:
    #                         socketio.emit(
    #                             "receive_audio",
    #                             {"audio": f.read()},
    #                             room=target_socket_id
    #                         )
    #                         print("Sending translated audio to user", user_id_receiver)

    #                 # Send transcription to the user
    #                 emit("transcription", {
    #                     "english": english_transcription, 
    #                     "translated": translated_text, 
    #                     "sender_user_id": user_id
    #                 }, to=target_socket_id)

    #                 # Save transcription data to MongoDB
    #                 transcription_data = {
    #                     "english_transcription": english_transcription,
    #                     "translated_text": translated_text,
    #                     "src_language_code": src_language_code,
    #                     "target_language_code": user_language_code,
    #                     "timestamp": datetime.utcnow()
    #                 }

    #                 room_sessions_collection.update_one(
    #                     {"room": room, "user_id": user_id_receiver},
    #                     {"$push": {"transcriptions": transcription_data}},
    #                     upsert=True
    #                 )
    #                 # frames.clear()

    #         except Exception as e:
    #             print(f"Error processing audio data for transcription: {e}")

    # # Handle stop event to combine audio and save the file
    # @socketio.on("stop")
    # def handle_stop():
    #     is_recording = False
    #     if frames:
    #         unique_filename = str(uuid.uuid4())
    #         file_path = os.path.join(AUDIO_DIR, f"audio_{unique_filename}.wav")
    #         with wave.open(file_path, "wb") as wf:
    #             wf.setnchannels(1)
    #             wf.setsampwidth(2)
    #             wf.setframerate(44100)
    #             wf.writeframes(b"".join(frames))
    #     print("Now clearing all the frames to the Empty List")
    #     # frames.clear()
    #     print("Now we will see the count of the frames :-----",len(frames))

    #     # Clear GPU memory
    #     print("Clearing GPU memory...")
    #     torch.cuda.empty_cache()
    #     print("GPU memory cleared.")

# -------------------threaded--------------------------------------------------------------------------------------------

    # older implementation of handle_audio
    # @socketio.on("audio")
    # def handle_audio(data, room_data):
    #     global handler, old_len, full_audio, frames, now
    #     time_elapsed = (time.time() - now) if now is not None else 0
    #     print("time elapsed : ", time_elapsed)
    #     now = time.time()

    #     # user_data
    #     room = room_data["room"]
    #     user_id = room_data["user_id"]
    #     src_language = room_data["src_language"]
    #     sample_rate = room_data["sampleRate"]

    #     if handler.current_speaker_id != user_id:
    #         handler.current_speaker_id = user_id
            
    #     if handler.user_count != len(user_session_details):
    #         handler.user_count = len(user_session_details)
    #         handler.update_language_mappings()
    #         print("calling update mapping")
            


    #     if data:
    #         frames.append(data)
    #         audio_data = b"".join(frames)
    #         audio_array, sample_rate = torchaudio.load(io.BytesIO(audio_data))
    #         audio_array = audio_array.cpu().numpy().flatten()
    #         handler.handle_audio(audio_array[old_len:], socketio)            
    #         old_len = len(audio_array)
    
    # room aware implementation.
    @socketio.on("audio")
    def handle_audio(data, room_data):
        global handler, old_len, full_audio, frames, now
        time_elapsed = (time.time() - now) if now is not None else 0
        print("time elapsed : ", time_elapsed)
        now = time.time()

        # Extract room data
        room = room_data["room"]
        user_id = room_data["user_id"]
        src_language = room_data["src_language"]
        sample_rate = room_data["sampleRate"]

        # Update current speaker
        if room not in handler.current_speaker_id or handler.current_speaker_id[room] != user_id:
            handler.current_speaker_id[room] = user_id
            print(handler.current_speaker_id)
            
        

        
        # Get users for the current room
        room_users = {}
        current_room_id = user_session_details.get('room_id')
        
        if current_room_id == room:
            # Get all user entries (excluding the room_id key)
            room_users = {k: v for k, v in user_session_details.items() 
                        if k != 'room_id' and isinstance(v, dict)}
        
        # Initialize room-specific user count if needed
        if room not in handler.user_counts:
            handler.user_counts[room] = 0
        
        # Check if user count changed for this room
        if handler.user_counts[room] != len(room_users):
            handler.user_counts[room] = len(room_users)
            handler.update_language_mappings(room)
            print("calling update mapping")
            print(f"Current room {room} users: {room_users}")
            print(f"Total user session details: {user_session_details}")

        # Process audio data
        if data:
            frames.append(data)
            audio_data = b"".join(frames)
            audio_array, sample_rate = torchaudio.load(io.BytesIO(audio_data))
            audio_array = audio_array.cpu().numpy().flatten()
            
            # Pass room context to handle_audio
            handler.handle_audio(audio_array[old_len:], socketio, room)
            old_len = len(audio_array)

    # Handle stop event to combine audio and save the file
    @socketio.on("stop")
    def handle_stop():
        global full_audio, frames
        print('stop audio is being called')
        if frames:
            unique_filename = str(uuid.uuid4())
            file_path = os.path.join(AUDIO_DIR, f"audio_{unique_filename}.wav")
            with wave.open(file_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(b"".join(frames))
            
            
            handler.stop_threads()
            # frames.clear()




# ---------------------------------------------------------------------------------------------------------------------------------------
    # @socketio.on("audio")
    # def handle_audio(data, room_data):
    #     global frames, is_recording, old_len, audio_buffer  # Access global frames
    #     room = room_data["room"]
    #     user_id = room_data["user_id"]
    #     src_language = room_data["src_language"]
    #     sampling = room_data["sampleRate"]
    #     print(sampling,"...... The Audio sampling rate that is coming from the frontend............")
    #     translated_text = ""

    #     if not is_recording:
    #         return
        
    #     print(type(frames),"--------------------------------------------------------")
    #     if data:
    #         frames.append(data)
    #         print(type(frames),"....................................................")
    #         try:
    #             audio_data = b"".join(frames)  # Combine audio frames
    #             print(len(audio_data),"..................lenght...of_audio_data..................")
    #             # print(type(audio_data),";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
    #             # Load the audio data and resample if necessary
    #             waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))
    #             if sample_rate != 16000:
    #                 print("..................................Re-sampling...................................")
    #                 waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    #                 audio_array = waveform.numpy().flatten()   #--- added
    #                 old_len = 0 if len(audio_array) < old_len else old_len
    #             chunk = audio_array[old_len:]
    #             # audio_buffer = np.append(audio_buffer, audio_array[old_len:])
    #             # # Buffering
    #             # n_batch_size = 16000*3
    #             # keep_samples = int(16000*.1)
    #             # if len(audio_buffer) >= n_batch_size:
    #             #     chunk = audio_buffer[:n_batch_size]
    #             #     audio_buffer = audio_buffer[n_batch_size - keep_samples:]
                
    #             # else:
    #             #     return

    #             # Calculate RMS energy to determine if the audio is significant
    #             rms_energy = np.sqrt(np.mean(chunk**2))
    #             # rms_energy = np.sqrt(np.mean(waveform.numpy()**2))
    #             print(f"RMS Energy: {rms_energy}","-------------it should be more then 0.02------------------")

    #             # Skip transcription if audio does not meet the threshold
    #             MIN_RMS_THRESHOLD = 0.04  # Adjust based on your needs ------------quiet audio--------0.1 to 0.1, 0.7 to 1.0 indicates very loud audio.,   0.0 indicates complete silence
    #             if rms_energy < MIN_RMS_THRESHOLD:
    #                 print("Audio below threshold. Skipping transcription.")
    #                 return

    #             # Prepare input for the transcription model
    #             audio_inputs = processor_m4t(audios=chunk, return_tensors="pt").to(device)
    #             # audio_inputs = processor_m4t(audios=waveform.squeeze().numpy(), return_tensors="pt").to(device)
    #             old_len = len(audio_array)
    #             # Get English transcription
    #             src_language_code = "eng" if src_language.lower() == "english" else languages.get(src_language.lower(), None)
    #             output_tokens_eng = model_m4t.generate(**audio_inputs, tgt_lang=src_language_code, generate_speech=False)
    #             english_transcription = processor_m4t.decode(output_tokens_eng[0].tolist()[0], skip_special_tokens=True)

    #             print("English Transcription--", english_transcription)
    #             all_transcriptions.append(english_transcription)

    #             # Perform translations
    #             for user_id_receiver, details in user_session_details.items():
    #                 user_language = details.get("language", "Unknown")
    #                 target_socket_id = user_sockets.get(user_id_receiver)
    #                 if user_language.lower() == "french":
    #                     user_language_code = "fra"
    #                 elif user_language.lower() == "arabic":
    #                     user_language_code = "arb"
    #                 else:
    #                     user_language_code = user_language[:3]

    #                 # user_language_code = "fra" if user_language.lower() == "french" else user_language[:3]
    #                 # user_language_code = "arb" if user_language.lower() == "arabic" else user_language[:3]
    #                 print(user_language_code,".......................................pppp...............................")
    #                 if user_language_code != "kaz":
    #                     output_tokens_tel = model_m4t.generate(**audio_inputs, tgt_lang=user_language_code, generate_speech=False)
    #                     translated_text = processor_m4t.decode(output_tokens_tel[0].tolist()[0], skip_special_tokens=True)
    #                 elif user_language_code == "kaz":
    #                     output_tokens_tel = model.generate(**audio_inputs, tgt_lang=user_language_code, generate_speech=False)
    #                     translated_text = processor.decode(output_tokens_tel[0].tolist()[0], skip_special_tokens=True)

    #                 print("Translated Text from the model --", translated_text)

    #                 # ------------------speech to speech------------------
    #                 # audio_array_from_audio = model.generate(
    #                 #     **audio_inputs, tgt_lang=user_language_code
    #                 # )[0].cpu().numpy().squeeze()
                    
    #                 # sf.write(f"{user_id_receiver}_translated_audio.wav", audio_array_from_audio, 16000)
                    
    #                 # # Send translated audio back to user
    #                 # with open(f"{user_id_receiver}_translated_audio.wav", "rb") as f:
    #                 #     if user_id != user_id_receiver:
    #                 #         socketio.emit(
    #                 #             "receive_audio",
    #                 #             {"audio": f.read()},
    #                 #             room=target_socket_id
    #                 #         )
    #                 # ----------------------------------------------------
                    
    #                 # Send transcription to the user
    #                 emit("transcription", {
    #                     "english": english_transcription, 
    #                     "translated": translated_text, 
    #                     "sender_user_id": user_id
    #                 }, to=target_socket_id)

    #                 # Save transcription data to MongoDB
    #                 transcription_data = {
    #                     "english_transcription": english_transcription,
    #                     "translated_text": translated_text,
    #                     "src_language_code": src_language_code,
    #                     "target_language_code": user_language_code,
    #                     "timestamp": datetime.utcnow()
    #                 }

    #                 room_sessions_collection.update_one(
    #                     {"room": room, "user_id": user_id_receiver},
    #                     {"$push": {"transcriptions": transcription_data}},
    #                     upsert=True
    #                 )
    #                 frames  = frames[10:] if len(frames) > 11 else frames # ------------- added
    #                 print(frames,"......................last..................")
    #                 # frames.clear()

    #         except Exception as e:
    #             print(f"Error processing audio data for transcription: {e}")

    # # Handle stop event to combine audio and save the file
    # @socketio.on("stop")
    # def handle_stop():
    #     is_recording = False
    #     if frames:
    #         unique_filename = str(uuid.uuid4())
    #         file_path = os.path.join(AUDIO_DIR, f"audio_{unique_filename}.wav")
    #         with wave.open(file_path, "wb") as wf:
    #             wf.setnchannels(1)
    #             wf.setsampwidth(2)
    #             wf.setframerate(44100)
    #             wf.writeframes(b"".join(frames))
    #     print("Now clearing all the frames to the Empty List")
    #     frames.clear()
    #     print("Now we will see the count of the frames :-----",len(frames))

#---------------------------------------------------------------------------------------------------------

#----------------------------------- SPEECH TO SPEECH ----------------------------------------------------------------------
    # @socketio.on("speech_to_speech")
    # def handle_audio(data, room_data):
    #     global frames, is_recording  # Access global frames
    #     room = room_data["room"]
    #     user_id = room_data["user_id"]
    #     src_language = room_data["src_language"]
    #     sampling = room_data["sampleRate"]
    #     print(sampling, "...... The Audio sampling rate that is coming from the frontend............")
    #     translated_text = ""

    #     if not is_recording:
    #         return

    #     print(type(frames), "--------------------------------------------------------")
    #     if data:
    #         frames.append(data)
    #         print(type(frames), "....................................................")
    #         try:
    #             audio_data = b"".join(frames)  # Combine audio frames
    #             print(len(audio_data), "..................length...of_audio_data..................")
                
    #             # Load the audio data and resample if necessary
    #             waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))
    #             if sample_rate != 16000:
    #                 print("..................................Re-sampling...................................")
    #                 waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    #             # Calculate RMS energy to determine if the audio is significant
    #             rms_energy = np.sqrt(np.mean(waveform.numpy()**2))
    #             print(f"RMS Energy: {rms_energy}", "-------------it should be more than 0.02------------------")

    #             # Skip transcription if audio does not meet the threshold
    #             MIN_RMS_THRESHOLD = 0.01  # Adjust based on your needs
    #             if rms_energy < MIN_RMS_THRESHOLD:
    #                 print("Audio below threshold. Skipping transcription.")
    #                 return

    #             # Prepare input for the transcription model
    #             audio_inputs = processor_m4t(audios=waveform.squeeze().numpy(), return_tensors="pt").to(device)

    #             # Perform translations for each user
    #             for user_id_receiver, details in user_session_details.items():
    #                 user_language = details.get("language", "Unknown")
    #                 target_socket_id = user_sockets.get(user_id_receiver)

    #                 if user_language.lower() == "french":
    #                     user_language_code = "fra"
    #                 elif user_language.lower() == "arabic":
    #                     user_language_code = "arb"
    #                 else:
    #                     user_language_code = user_language[:3]

    #                 print(user_language_code, ".......................................pppp...............................")

    #                 audio_array_from_audio = model_m4t.generate(**audio_inputs, tgt_lang=user_language_code)[0].cpu().numpy().squeeze()

    #                 # Create the output directory if it does not exist
    #                 output_path = "speech_to_speech"
    #                 if not os.path.exists(output_path):
    #                     os.mkdir(output_path)

    #                 # Generate a unique filename
    #                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #                 filename = f"user_{user_id_receiver}_lang_{user_language_code}_{timestamp}.wav"
    #                 file_path = os.path.join(output_path, filename)

    #                 # Save the translated audio
    #                 sf.write(file_path, audio_array_from_audio, 16000)
    #                 print(f"Saved translated audio to {file_path}")

    #                 # Clear frames after processing
    #                 # frames.clear()

    #         except Exception as e:
    #             print(f"Error processing audio data for transcription: {e}")
    # @socketio.on("speech_to_speech")
    # def handle_speech(data, room_data):
    #     global frames, is_recording
    #     room = room_data["room"]
    #     user_id = room_data["user_id"]
    #     src_language = room_data["src_language"]
    #     sampling = room_data["sampleRate"]

    #     if not is_recording:
    #         return

    #     if data:
    #         frames.append(data)
    #         try:
    #             # Combine audio frames and process
    #             audio_data = b"".join(frames)
    #             waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))
                
    #             if sample_rate != 16000:
    #                 waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
                
    #             rms_energy = np.sqrt(np.mean(waveform.numpy()**2))
    #             if rms_energy < 0.04:  # Skip low-energy audio
    #                 return
                
    #             audio_inputs = processor_m4t(audios=waveform.squeeze().numpy(), return_tensors="pt").to(device)
                
    #             for user_id_receiver, details in user_session_details.items():
    #                 if user_id_receiver == user_id:
    #                     continue 
    #                 user_language = details.get("language", "Unknown")
    #                 target_socket_id = user_sockets.get(user_id_receiver)
                    
    #                 user_language_code = {
    #                     "french": "fra",
    #                     "arabic": "arb"
    #                 }.get(user_language.lower(), user_language[:3])
                    
    #                 audio_array_from_audio = model_m4t.generate(
    #                     **audio_inputs, tgt_lang=user_language_code
    #                 )[0].cpu().numpy().squeeze()
                    
    #                 sf.write(f"{user_id_receiver}_translated_audio.wav", audio_array_from_audio, 16000)
                    
    #                 # Send translated audio back to user
    #                 with open(f"{user_id_receiver}_translated_audio.wav", "rb") as f:
    #                     socketio.emit(
    #                         "receive_audio",
    #                         {"audio": f.read()},
    #                         room=target_socket_id
    #                     )
                
    #             # frames.clear()  # Clear frames after processing
                
    #         except Exception as e:
    #             print(f"Error processing audio data: {e}")

# -----------------------------------------------------------------------------------------------------------------------------------------------
    # @socketio.on("speech_to_speech")
    # def handle_audio(data, room_data):
    #     global frames, is_recording  # Access global frames
    #     room = room_data["room"]
    #     user_id = room_data["user_id"]
    #     src_language = room_data["src_language"]
    #     sampling = room_data["sampleRate"]
    #     print(sampling,"...... The Audio sampling rate that is coming from the frontend............")
    #     translated_text = ""

    #     if not is_recording:
    #         return
        
    #     print(type(frames),"--------------------------------------------------------")
    #     if data:
    #         frames.append(data)
    #         print(type(frames),"....................................................")
    #         try:
    #             audio_data = b"".join(frames)  # Combine audio frames
    #             print(len(audio_data),"..................lenght...of_audio_data..................")
    #             # print(type(audio_data),";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
    #             # Load the audio data and resample if necessary
    #             waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))
    #             if sample_rate != 16000:
    #                 print("..................................Re-sampling...................................")
    #                 waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    #             # Calculate RMS energy to determine if the audio is significant
    #             rms_energy = np.sqrt(np.mean(waveform.numpy()**2))
    #             print(f"RMS Energy: {rms_energy}","-------------it should be more then 0.02------------------")

    #             # Skip transcription if audio does not meet the threshold
    #             MIN_RMS_THRESHOLD = 0.01  # Adjust based on your needs ------------quiet audio--------0.1 to 0.1, 0.7 to 1.0 indicates very loud audio.,   0.0 indicates complete silence
    #             if rms_energy < MIN_RMS_THRESHOLD:
    #                 print("Audio below threshold. Skipping transcription.")
    #                 return

    #             # Prepare input for the transcription model
    #             audio_inputs = processor_m4t(audios=waveform.squeeze().numpy(), return_tensors="pt").to(device)

    #             # Get English transcription
    #             src_language_code = "eng" if src_language.lower() == "english" else languages.get(src_language.lower(), None)
    #             output_tokens_eng = model_m4t.generate(**audio_inputs, tgt_lang=src_language_code, generate_speech=True)
    #             # english_transcription = processor_m4t.decode(output_tokens_eng[0].tolist()[0], skip_special_tokens=True)

    #             # print("English Transcription--", english_transcription)
    #             # all_transcriptions.append(english_transcription)

    #             # Perform translations
    #             for user_id_receiver, details in user_session_details.items():
    #                 user_language = details.get("language", "Unknown")
    #                 target_socket_id = user_sockets.get(user_id_receiver)
    #                 if user_language.lower() == "french":
    #                     user_language_code = "fra"
    #                 elif user_language.lower() == "arabic":
    #                     user_language_code = "arb"
    #                 else:
    #                     user_language_code = user_language[:3]

    #                 print(user_language_code,".......................................pppp...............................")
    #                 audio_array_from_audio = model_m4t.generate(**audio_inputs, tgt_lang = user_language_code)[0].cpu().numpy().squeeze()
    #                 print(audio_array_from_audio,"...............translated_audio........................")
    #                 import soundfile as sf
    #                 output_path = "speech_to_speech"
    #                 if not os.path.exists(output_path):
    #                     os.mkdir("speech_to_speech")
    #                     sf.write(os.path.join(output_path, f"{user_id_receiver}.wav"),audio_array_from_audio, 16000)
    #                 # with open("translated_audio.wav", "wb") as f:
    #                 #     sf.write(f, audio_array_from_audio, 16000)
    #                 # print("Translated Text from the model --", translated_text)
                    
    #                 # # Send transcription to the user
    #                 # emit("transcription", {
    #                 #     "english": english_transcription, 
    #                 #     "translated": translated_text, 
    #                 #     "sender_user_id": user_id
    #                 # }, to=target_socket_id)

    #                 # frames.clear()

    #         except Exception as e:
    #             print(f"Error processing audio data for transcription: {e}")

#---------------------------------------------------------------------------------------------------------

    @socketio.on("chat_history")
    def chat_history_data(data):
        room_id = data.get("roomId")
        instructor_name = data.get("instructorname")
        role = data.get("role")
        date_str = data.get("date")
        date_obj = datetime.strptime(date_str, "%d/%m/%Y")
        formatted_date = date_obj.strftime("%d/%m/%Y")
        print(formatted_date)
        sender_name = data.get("senderName")
        print(sender_name,"........................")
        selected_language = data.get("selectedLanguage")
        timestamp_str = data.get("timestamp")
        timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y, %H:%M:%S")
        # Message object
        details = users_collection.find({"username":instructor_name})
        typing = data.get("action")
        print(typing,".../././/.//........")
        for i in details:
            if i["role"] == "instructor":
                if typing == "sent":
                    message_key_value = data.get("original_message")
                elif typing == "received":
                    message_key_value = data.get("translated_message")
                else:
                    message_key_value = None
                message = {
                    "sender_username": sender_name,
                    "message": message_key_value,
                    "translated_message": data.get("translated_message"),
                    "original_message": data.get("original_message"),
                    "action": data.get("action"),
                    "type": data.get("type"),
                    "timestamp": timestamp
                }
                # Check if document exists for room_id and instructor_name
                existing_document = chat_history_collection.find_one({"room_id": room_id, "instructor_name": instructor_name})
                print(room_id,instructor_name,"........////////...........")
                if existing_document:
                    print("//////////////////")
                    chat_history_collection.update_one(
                        {"room_id": room_id, "instructor_name": instructor_name, "date": formatted_date},
                        {"$push": {"chat_history": message}}
                    )
                else:
                    print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,")
                    new_document = {
                        "room_id": room_id,
                        "instructor_name": instructor_name,
                        "selectedLanguage": selected_language,
                        "chat_history": [message],  # Initialize with the first message
                        "date": formatted_date,
                    }
                    chat_history_collection.insert_one(new_document)

                print("Chat history updated successfully.")
            else :
                print("He is not a Instructor",instructor_name)

    # pip install flask-socketio weasyprint pymongo

    @socketio.on("download_chat_history")
    def download_chat_history(data):
        print("Received data from client:", data)

        date_str = data.get("date")
        room_id = data.get("roomId")
        instructor_name = data.get("username")

        chatting = chat_history_collection.find({
            "room_id": room_id,
            "instructor_name": instructor_name
        })

        table_data = []
        sl_no = 1
        for chat in chatting:
            print(chat, "............DEBUG DATA.............")
            if 'chat_history' in chat:  
                for chat_item in chat['chat_history']:
                    action = chat_item.get("action")
                    o_message = chat_item.get("original_message", "")
                    t_message = chat_item.get("translated_message", "")
                    print(action,o_message,t_message,".....action, omessage, tmessage...................")
                    if action == "sent":
                        message = o_message
                    elif action == "received":
                        message = t_message
                    else:
                        message = ""
                    table_data.append({
                        "sl_no": sl_no,
                        "action": chat_item.get("action", ""),
                        "username": chat_item.get("sender_username", ""),
                        "message": message,
                        "type": chat_item.get("type", ""),
                    })
                    sl_no += 1

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Chat History - {date_str}</title>
             <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f9;
                    color: #333;
                }}
                .container {{
                    width: 90%;
                    margin: 20px auto;
                    padding: 30px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding: 10px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 8px;
                }}
                .header h1 {{
                    font-size: 36px;
                    margin: 0;
                }}
                .header p {{
                    font-size: 18px;
                    margin: 5px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                    font-size: 16px;
                }}
                td {{
                    font-size: 14px;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                tr:hover {{
                    background-color: #f1f1f1;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    font-size: 14px;
                    color: #777;
                }}
            </style>
        </head>
        <body>

        <div class="container">
            <div class="header">
                <h1>CHAT HISTORY</h1>
                <p>Date: {date_str}</p>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Sl.No</th>
                        <th>Action</th>
                        <th>Username</th>
                        <th>Message</th>
                        <th>Type</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Add rows to the table
        for row in table_data:
            html_content += f"""
            <tr>
                <td>{row["sl_no"]}</td>
                <td>{row["action"]}</td>
                <td>{row["username"]}</td>
                <td>{row["message"]}</td>
                <td>{row["type"]}</td>
            </tr>
            """

        # Close the table and HTML tags
        html_content += """
                </tbody>
            </table>

            <div class="footer">
                <p>Copyright @2024- Powered by DIT / INICAI and Pinaca Technologies</p>
            </div>
        </div>

        </body>
        </html>
        """

        # Convert HTML to PDF
        pdf_data = weasyprint.HTML(string=html_content).write_pdf()

        # Send the PDF as a file stream to the client
        pdf_filename = f"chat_history_{date_str}.pdf"

        # Create a BytesIO stream for the PDF data
        pdf_stream = BytesIO(pdf_data)
        pdf_stream.seek(0)

        # Send the PDF to the client
        emit("pdf_download_ready", {"file_name": pdf_filename,"file_data": pdf_stream.read()})

        print(f"PDF ready for download: {pdf_filename}")

        # for captions download


    @socketio.on('download_captions')
    def handle_download_captions(data):
        print("Received data for captions download:", data)
        room_id = data.get("roomId")
        instructor_name = data.get("username")
        date = data.get("date")
        print(date,"....................")
        print("room_id", room_id)

        # Fetch data from MongoDB
        trans = room_sessions_collection.find({
            "room": room_id
        })
        trans_list = list(trans)
        # Prepare table data
        table_data = []
        sl_no = 1
        for caption in trans_list:
            if 'transcriptions' in caption:  # Check if 'transcriptions' key exists
                for caption_item in caption['transcriptions']:
                    # Add transcription to the table
                    table_data.append({
                        "sl_no": sl_no,
                        "name": instructor_name,  # Use instructor_name for 'name'
                        "caption": caption_item.get("english_transcription", "N/A")  # Default to "N/A" if key is missing
                    })
                    sl_no += 1

        # Generate styled HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Captions - {date}</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    background-color: #f4f4f9;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 800px;
                    margin: auto;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    text-align: center;
                    padding: 10px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 8px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 24px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Captions</h1>
                    <p>Date: {date}</p>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Sl. No</th>
                            <th>Name</th>
                            <th>Caption</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        # Add rows dynamically to the HTML table
        for row in table_data:
            html_content += f"""
            <tr>
                <td>{row["sl_no"]}</td>
                <td>{row["name"]}</td>
                <td>{row["caption"]}</td>
            </tr>
            """

        # Close the HTML structure
        html_content += """
                    </tbody>
                </table>
                <div class="footer">
                    <p>Copyright @2024- Powered by DIT / INICAI and Pinaca Technologies</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Convert HTML to PDF
        pdf_data = weasyprint.HTML(string=html_content).write_pdf()

        # Prepare the PDF file for download
        pdf_filename = f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Send the PDF as a binary stream
        emit("pdf_download_ready", {
            "file_name": pdf_filename,
            "file_data": pdf_data
        })
        print(f"PDF ready for download: {pdf_filename}")

    @socketio.on('remove_participant')
    def handle_kick_out(data):
        user_id = data.get('user_id')
        room_id = data.get('room_id')
        role = data.get('role')
        
        print(user_id, room_id, "...................values_printing...................")
        print(rooms, "...............printing_rooms.........")

        if room_id in rooms:  # Check if the room exists
            if user_id in rooms[room_id]:  # Check if the user is in the room
                rooms[room_id].remove(user_id)  # Remove the user from the room
                print(f"User {user_id} removed from room {room_id}.")
                print(rooms, ".............after removed...............")
                
                # Notify remaining users in the room
                room_user_details = [user_session_details[uid] for uid in rooms[room_id]]
                socketio.emit(
                    "updateUsers",
                    {"users": rooms[room_id], "userDetails": room_user_details},
                    room=room_id,
                )

                # Notify the user that they were removed
                user_sid = user_sockets.get(user_id)
                if user_sid:
                    socketio.emit("kickedOut", {"message": "You have been removed by the instructor."}, to=user_sid)
                
                # Let the user leave the room
                leave_room(room_id, sid=user_sid)
                
                # Remove the room if empty
                if not rooms[room_id]:
                    del rooms[room_id]
                    print(f"Room {room_id} is now empty and has been removed.")
            else:
                print(f"User {user_id} not found in room {room_id}.")
        else:
            print(f"Room {room_id} does not exist.")


        
                