import torch, io, asyncio, webrtcvad, struct, traceback, base64, aiohttp
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from openai import AsyncOpenAI
import soundfile as sf
import numpy as np

app = FastAPI()

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8001/v1"
)

print("Loading WebRTC VAD...")
vad = webrtcvad.Vad(3)

async def ffmpeg_read_from_buffer(buffer: io.BytesIO, sampling_rate=16000):
    """
    Reads audio from an in-memory BytesIO buffer using FFmpeg asynchronously.
    Returns a PyTorch tensor (float32, mono, 16kHz).
    """
    buffer.seek(0)
    command = [
        "ffmpeg",
        "-i", "pipe:0",
        "-ar", str(sampling_rate),
        "-ac", "1",
        "-f", "s16le",
        "-"
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        stdout, _ = await process.communicate(input=buffer.read())
        
        if process.returncode != 0:
            raise ValueError(f"FFmpeg failed with return code {process.returncode}")
        
        audio_np = np.frombuffer(stdout, dtype=np.int16)
        audio_float = audio_np.astype(np.float32) / 32768.0
        return torch.from_numpy(audio_float.copy())
    except Exception as e:
        raise ValueError(f"FFmpeg failed to read audio: {e}")

async def get_speech_segments_webrtc_from_buffer(buffer: io.BytesIO):
    """
    Performs WebRTC VAD on audio from a BytesIO buffer.
    Returns (full_tensor, speech_segments)
    
    Note: WebRTC VAD itself is CPU-bound and synchronous, so we run it in a thread pool
    """
    wav = await ffmpeg_read_from_buffer(buffer, sampling_rate=16000)
    if wav is None:
        return None, []

    # Run VAD processing in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    full_tensor, speech_segments = await loop.run_in_executor(
        None, 
        _process_vad, 
        wav
    )
    
    return full_tensor, speech_segments

def _process_vad(wav):
    """
    Synchronous VAD processing - called from executor
    """
    audio_np = (wav.numpy() * 32768).astype(np.int16)
    frame_duration_ms = 30
    frame_size = int(16000 * frame_duration_ms / 1000)

    speech_segments = []
    is_speech_active = False
    speech_start = None

    for i in range(0, len(audio_np), frame_size):
        frame = audio_np[i:i + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        frame_bytes = struct.pack(f'{len(frame)}h', *frame)
        is_speech = vad.is_speech(frame_bytes, 16000)

        if is_speech and not is_speech_active:
            speech_start = i
            is_speech_active = True
        elif not is_speech and is_speech_active:
            speech_segments.append({'start': speech_start, 'end': i})
            is_speech_active = False

    if is_speech_active:
        speech_segments.append({'start': speech_start, 'end': len(audio_np)})

    # Return normalized float32 tensor
    full_tensor = torch.from_numpy(audio_np.astype(np.float32) / 32768.0)
    return full_tensor, speech_segments

async def process_audio_with_vad(wav_tensor, speech_timestamps, sampling_rate=16000):
    """
    Async version of audio processing with VAD
    """
    if not speech_timestamps:
        return None, None

    # Prepare chunks and time mapping (fast, keep synchronous)
    audio_chunks = []
    time_mapping = []
    current_proc_time = 0.0

    for segment in speech_timestamps:
        start_sample = segment['start']
        end_sample = segment['end']
        chunk = wav_tensor[start_sample:end_sample]
        audio_chunks.append(chunk)

        duration_sec = (end_sample - start_sample) / sampling_rate
        orig_start_sec = start_sample / sampling_rate

        time_mapping.append({
            'proc_start': current_proc_time,
            'proc_end': current_proc_time + duration_sec,
            'orig_start': float(orig_start_sec)
        })
        current_proc_time += duration_sec

    final_tensor = torch.cat(audio_chunks)
    final_audio_np = final_tensor.numpy()

    # Run soundfile write in thread pool (I/O operation)
    loop = asyncio.get_event_loop()
    temp_wav = await loop.run_in_executor(
        None,
        _write_wav_to_buffer,
        final_audio_np,
        sampling_rate
    )

    # Compress to MP3 via FFmpeg asynchronously
    compressed_audio = await _compress_to_mp3(temp_wav, sampling_rate)
    
    buffer = io.BytesIO(compressed_audio)
    buffer.seek(0)

    return buffer, time_mapping

def _write_wav_to_buffer(audio_np, sampling_rate):
    """
    Synchronous WAV writing - called from executor
    """
    temp_wav = io.BytesIO()
    sf.write(temp_wav, audio_np, sampling_rate, format='WAV', subtype='PCM_16')
    temp_wav.seek(0)
    return temp_wav

async def _compress_to_mp3(wav_buffer: io.BytesIO, sampling_rate: int) -> bytes:
    """
    Async FFmpeg MP3 compression
    """
    process = await asyncio.create_subprocess_exec(
        'ffmpeg',
        '-i', 'pipe:0',
        '-ar', str(sampling_rate),
        '-ac', '1',
        '-codec:a', 'libmp3lame',
        '-b:a', '64k',
        '-f', 'mp3',
        'pipe:1',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )

    stdout, _ = await process.communicate(input=wav_buffer.read())
    return stdout

def map_timestamp(processed_time, mapping):
    """
    Maps processed timestamp back to original audio timestamp
    """
    for segment in mapping:
        if segment['proc_start'] <= processed_time <= segment['proc_end']:
            offset = processed_time - segment['proc_start']
            return segment['orig_start'] + offset

    if mapping and processed_time > mapping[-1]['proc_end']:
        last = mapping[-1]
        return last['orig_start'] + (processed_time - last['proc_start'])

    return processed_time

async def download_file_from_url(url: str) -> io.BytesIO:
    """
    Downloads file from URL asynchronously
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download audio from URL: HTTP {response.status}"
                    )
                content = await response.read()
                return io.BytesIO(content)
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {str(e)}")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=400, detail="Audio download timed out (30s)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unexpected error during download: {str(e)}")

def decode_base64_audio(b64_str: str) -> io.BytesIO:
    """
    Decodes base64 audio string to BytesIO
    """
    try:
        # Handle data URI if present
        if b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[1]
        audio_bytes = base64.b64decode(b64_str)
        return io.BytesIO(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {str(e)}")

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(None),
    base64_audio: str = Form(None),
    url: str = Form(None)
):
    """
    Transcribe audio from file upload, base64 string, or URL
    """
    # Validate exactly one input method
    provided = sum(x is not None for x in [file, base64_audio, url])
    if provided == 0:
        raise HTTPException(status_code=400, detail="Provide one of: file, base64_audio, or url.")
    if provided > 1:
        raise HTTPException(status_code=400, detail="Provide only one of: file, base64_audio, or url.")

    try:
        # Load audio into BytesIO
        if file:
            audio_buffer = io.BytesIO(await file.read())
        elif base64_audio:
            audio_buffer = decode_base64_audio(base64_audio)
        elif url:
            audio_buffer = await download_file_from_url(url)
        else:
            raise HTTPException(status_code=400, detail="No valid audio input provided.")

        # Run VAD (now async)
        wav_data, timestamps = await get_speech_segments_webrtc_from_buffer(audio_buffer)

        if not timestamps:
            return {
                "text": "",
                "segments": [],
                "message": "No speech detected."
            }

        # Process and compress (now async)
        processed_buffer, time_map = await process_audio_with_vad(wav_data, timestamps)

        if processed_buffer is None:
            return {
                "text": "",
                "segments": [],
                "message": "No valid speech segments after processing."
            }

        # Transcribe with Whisper via vLLM
        transcription = await client.audio.transcriptions.create(
            model="openai/whisper-large-v3-turbo",
            file=("audio.mp3", processed_buffer, "audio/mpeg"),
            response_format="verbose_json",
            language="pt",
            timestamp_granularities=["segment"]
        )

        def filter_and_map_segments(segments):
            """
            Filters low-quality segments and maps timestamps back to original
            """
            filtered = []
            last_text = None
            repetition_count = 0

            for segment in segments:
                orig_start = map_timestamp(segment.start, time_map)
                orig_end = map_timestamp(segment.end, time_map)

                text = segment.text.strip()
                # Skip low-quality segments
                if segment.no_speech_prob and segment.no_speech_prob >= 0.6: 
                    continue
                if segment.avg_logprob and segment.avg_logprob <= -1.0: 
                    continue
                if abs(segment.start - segment.end) < 0.05: 
                    continue

                # Avoid repetitions
                if text == last_text:
                    repetition_count += 1
                    if repetition_count > 2:
                        continue
                else:
                    repetition_count = 0

                filtered.append({
                    "start": round(orig_start, 2),
                    "end": round(orig_end, 2),
                    "text": text
                })
                last_text = text

            return filtered

        filtered_segments = filter_and_map_segments(transcription.segments)

        return {
            "segments": filtered_segments
        }

    except Exception as e:
        print("Error during transcription:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
