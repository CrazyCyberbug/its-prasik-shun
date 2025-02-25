from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import uvicorn
import json
import time
import threading
import torch
from typing import List, Dict
from transformers import AutoProcessor, SeamlessM4Tv2Model, VitsModel, AutoTokenizer

app = FastAPI()

# Define constants
sampling_rate = 16000
batch_size = sampling_rate * 5
keep_samples = int(sampling_rate * 0.15)
max_q_size = 50 * batch_size
is_running = True

# Initialize model and processor
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
model.to('cuda')

# TTS model for kaz and mya
tts_speech_model = None
tts_lang = None
tts_tokenizer = None

# Global dictionaries for text-only translation
text_queues: Dict[str, np.ndarray] = {}
text_response_lists: Dict[str, List] = {}
text_buffer_locks: Dict[str, threading.Lock] = {}
text_response_list_locks: Dict[str, threading.Lock] = {}
text_client_langs: Dict[str, str] = {}
text_inference_threads: Dict[str, threading.Thread] = {}
text_client_active: Dict[str, bool] = {}
text_thread_locks: Dict[str, threading.Lock] = {}

# Global dictionaries for speech translation
speech_queues: Dict[str, np.ndarray] = {}
speech_response_lists: Dict[str, List] = {}
speech_buffer_locks: Dict[str, threading.Lock] = {}
speech_response_list_locks: Dict[str, threading.Lock] = {}
speech_client_langs: Dict[str, str] = {}
speech_inference_threads: Dict[str, threading.Thread] = {}
speech_client_active: Dict[str, bool] = {}
speech_thread_locks: Dict[str, threading.Lock] = {}

class AudioData(BaseModel):
    client_id: str
    audio_data: List[float]
    sampling_rate: int
    tgt_lang: str
    reset_buffers: str

def calculate_rms_energy(audio_chunk):
    return np.sqrt(np.mean(np.square(audio_chunk)))

def is_silence(audio_chunk, silence_threshold=0.01):
    rms_energy = calculate_rms_energy(audio_chunk)
    print(f"RMS Energy: {rms_energy}")
    return rms_energy < silence_threshold

def speech2speech(audio_inputs, tgt_lang):
    speech = model.generate(**audio_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
    return speech

def load_TTS_model(tgt_lang_code):
    global tts_speech_model, tts_lang, tts_tokenizer
    
    if tts_speech_model == None or tts_lang != tgt_lang_code:        
        tts_lang = tgt_lang_code
        tts_speech_model = VitsModel.from_pretrained(f"facebook/mms-tts-{tgt_lang_code}").to('cuda')
        tts_tokenizer = AutoTokenizer.from_pretrained(f"facebook/mms-tts-{tgt_lang_code}")

    return tts_speech_model, tts_tokenizer    

def transcribe_text_only(input_array, tgt_lang):
    audio_inputs = processor(audios=input_array, sampling_rate=16000, return_tensors="pt").to('cuda')
    output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=False)
    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)    
    return translated_text_from_audio

def transcribe_speech(input_array, tgt_lang):
    audio_inputs = processor(audios=input_array, sampling_rate=16000, return_tensors="pt").to('cuda')
    output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=False)
    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

    if tgt_lang in ['mya', 'kaz']:
        speech_model, tokenizer = load_TTS_model(tgt_lang)
        inputs = tokenizer(translated_text_from_audio, return_tensors="pt").to('cuda')

        with torch.no_grad():
            output = speech_model(**inputs).waveform
        speech = output.squeeze().cpu().numpy()
    else:
        speech = speech2speech(audio_inputs=audio_inputs, tgt_lang=tgt_lang)
    
    return translated_text_from_audio, speech.tolist()


def transcribe_speech(input_array, tgt_lang):
    # get input_tokens
    audio_inputs = processor(audios=input_array, sampling_rate=16000, return_tensors="pt").to('cuda')

    if tgt_lang in ['mya', 'kaz']:
        # generate text in lang.
        output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=False)
        translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        
        #generate speech.
        speech_model, tokenizer = load_TTS_model(tgt_lang)
        inputs = tokenizer(translated_text_from_audio, return_tensors="pt").to('cuda')
        with torch.no_grad():
            output = speech_model(**inputs).waveform
        speech = output.squeeze().cpu().numpy()
        
    elif tgt_lang in ['hin', 'ben', 'arb']:
        # generate intermediate eng text.
        output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=False)
        eng_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        
        # generate text and speech in tgt_lang.
        audio_inputs = processor(text=eng_text, sampling_rate=16000, return_tensors="pt").to('cuda')
        output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, return_intermediate_token_ids=True)
        speech = output_tokens.waveform
        speech = speech.cpu().numpy()
        translated_text_from_audio = processor.decode(output_tokens.sequences.tolist()[0], skip_special_tokens=True)
        
    else:
        # generate text and speech in tgt_lang.
        output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, return_intermediate_token_ids = True)
        speech = output_tokens.waveform
        speech = speech.cpu().numpy()
        translated_text_from_audio = processor.decode(output_tokens.sequences.tolist()[0], skip_special_tokens=True)
    
    
    return translated_text_from_audio, speech.flatten().tolist()

def text_inference(client_id):
    while text_client_active.get(client_id, False):
        time.sleep(0.1)
        
        with text_thread_locks[client_id]:
            tgt_lang = text_client_langs.get(client_id)
            if not tgt_lang:
                continue
                
            queue = text_queues.get(client_id)
            if queue is None or queue.size < batch_size:
                continue
                
            audio_chunk = queue[:batch_size]
            text_queues[client_id] = queue[batch_size - keep_samples:]
            
            try:
                response = transcribe_text_only(audio_chunk, tgt_lang)
                with text_response_list_locks[client_id]:
                    text_response_lists[client_id].append(response)
            except Exception as e:
                print(f"Error processing audio for client {client_id}: {e}")

def speech_inference(client_id):
    while speech_client_active.get(client_id, False):
        time.sleep(0.1)
        
        with speech_thread_locks[client_id]:
            tgt_lang = speech_client_langs.get(client_id)
            if not tgt_lang:
                continue
                
            queue = speech_queues.get(client_id)
            if queue is None or queue.size < batch_size:
                continue
                
            audio_chunk = queue[:batch_size]
            speech_queues[client_id] = queue[batch_size - keep_samples:]
            
            try:
                response = transcribe_speech(audio_chunk, tgt_lang)
                with speech_response_list_locks[client_id]:
                    speech_response_lists[client_id].append(response)
            except Exception as e:
                print(f"Error processing audio for client {client_id}: {e}")

    # print(audio,"..................audio..............")
    # language = {
    #     "vietnamese": "vie",
    #     "kazakh": "kaz",
    #     "russian": "rus",
    #     "french": "fra",
    #     "bengali": "ben",
    #     "arabic": "arb",
    #     "english": "eng",
    #     "hindi": "hin",
    #     "telugu": "tel",
    # }
@app.post("/text-translation")
async def text_translation(audio: AudioData):
    client_id = audio.client_id
    tgt_lang = audio.tgt_lang
    print(client_id,tgt_lang,".....................................")
    # language_map = {
    #     "english": "eng",
    #     "vietnamese": "vie",
    #     "kazakh": "kaz",
    #     "russian": "rus",
    #     "french": "fra",
    #     "bengali": "ben",
    #     "arabic": "arb",
    #     "hindi": "hin",
    #     "telugu": "tel"
    # }
    # # Get the target language code or default to a placeholder if not found
    # tgt_lang = language_map.get(tgt_lang, "unknown")

    # print(f"Language '{tgt_lang}' found. Using code as tgt_lang.")


    if client_id not in text_queues:
        text_queues[client_id] = np.ndarray([], np.float32)
        text_response_lists[client_id] = []
        text_buffer_locks[client_id] = threading.Lock()
        text_response_list_locks[client_id] = threading.Lock()
        text_thread_locks[client_id] = threading.Lock()
        text_client_active[client_id] = True
        
        text_inference_threads[client_id] = threading.Thread(
            target=text_inference,
            args=(client_id,),
            daemon=True 
        )
        text_inference_threads[client_id].start()

    with text_thread_locks[client_id]:
        text_client_langs[client_id] = tgt_lang

    if audio.reset_buffers.lower() == "true":
        with text_thread_locks[client_id]:
            print(f"Resetting buffers for client: {client_id}")
            text_response_lists[client_id] = []
            text_queues[client_id] = np.ndarray([], np.float32)

    with text_buffer_locks[client_id]:
        if text_queues[client_id].size < max_q_size:
            audio_chunk = np.array(audio.audio_data)
            if is_silence(audio_chunk, silence_threshold=0.001):
                return {
                    "status": "Audio received",
                    "tgt_lang": audio.tgt_lang,
                    "processed": len(audio.audio_data),
                    "transcriptions": ""
                }
            text_queues[client_id] = np.append(text_queues[client_id], audio_chunk)

    transcriptions = ""
    with text_response_list_locks[client_id]:
        if text_response_lists[client_id]:
            transcriptions = text_response_lists[client_id].pop(0)
            print("transcription:",transcriptions)
            print("\n"*5,"remaining resposes in buffer:",  len(text_response_lists[client_id]), "\n" *2)
    return {
        "status": "Audio received",
        "processed": len(audio.audio_data),
        "transcriptions": str(transcriptions)
    }

@app.post("/speech-translation")
async def speech_translation(audio: AudioData):
    client_id = audio.client_id
    tgt_lang = audio.tgt_lang

    if client_id not in speech_queues:
        speech_queues[client_id] = np.ndarray([], np.float32)
        speech_response_lists[client_id] = []
        speech_buffer_locks[client_id] = threading.Lock()
        speech_response_list_locks[client_id] = threading.Lock()
        speech_thread_locks[client_id] = threading.Lock()
        speech_client_active[client_id] = True
        
        speech_inference_threads[client_id] = threading.Thread(
            target=speech_inference,
            args=(client_id,),
            daemon=True 
        )
        speech_inference_threads[client_id].start()

    with speech_thread_locks[client_id]:
        speech_client_langs[client_id] = tgt_lang

    if audio.reset_buffers.lower() == "true":
        with speech_thread_locks[client_id]:
            print(f"Resetting buffers for client: {client_id}")
            speech_response_lists[client_id] = []
            speech_queues[client_id] = np.ndarray([], np.float32)

    with speech_buffer_locks[client_id]:
        if speech_queues[client_id].size < max_q_size:
            audio_chunk = np.array(audio.audio_data)
            if is_silence(audio_chunk, silence_threshold=0.001):
                return {
                    "status": "Audio received",
                    "tgt_lang": audio.tgt_lang,
                    "processed": len(audio.audio_data),
                    "transcriptions": ""
                }
            speech_queues[client_id] = np.append(speech_queues[client_id], audio_chunk)

    transcriptions = ""
    with speech_response_list_locks[client_id]:
        if speech_response_lists[client_id]:
            # print("\n"*5,"remaining resposes in buffer:",  len(speech_response_lists[client_id]), "\n" *2)
            transcriptions = speech_response_lists[client_id].pop(0)

    return {
        "status": "Audio received",
        "processed": len(audio.audio_data),
        "transcriptions": str(transcriptions)
    }

@app.post("/cleanup/{client_id}")
async def cleanup_client(client_id: str):
    # Clean up text translation resources
    if client_id in text_client_active:
        text_client_active[client_id] = False
        if client_id in text_inference_threads:
            text_inference_threads[client_id].join(timeout=2.0)
        
        for dict_obj in [text_queues, text_response_lists, text_buffer_locks,
                        text_response_list_locks, text_client_langs,
                        text_inference_threads, text_client_active, text_thread_locks]:
            if client_id in dict_obj:
                del dict_obj[client_id]

    # Clean up speech translation resources
    if client_id in speech_client_active:
        speech_client_active[client_id] = False
        if client_id in speech_inference_threads:
            speech_inference_threads[client_id].join(timeout=2.0)
        
        for dict_obj in [speech_queues, speech_response_lists, speech_buffer_locks,
                        speech_response_list_locks, speech_client_langs,
                        speech_inference_threads, speech_client_active, speech_thread_locks]:
            if client_id in dict_obj:
                del dict_obj[client_id]
                
    return {"status": "success", "message": f"Cleaned up resources for client {client_id}"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)