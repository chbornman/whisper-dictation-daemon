#!/usr/bin/env python3
"""
Unified Whisper Streaming Daemon with multiple algorithms
"""

import argparse
import io
import logging
import os
import queue
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
from faster_whisper import WhisperModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/whisper_streaming.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionTask:
    """Represents a transcription task with metadata"""

    task_id: int
    audio_data: np.ndarray
    timestamp: float
    context_length: float  # Length of audio in seconds


class LocalAgreementBuffer:
    """
    Implements the Local Agreement-n algorithm for streaming transcription.
    For sliding windows, outputs text immediately as tasks complete.
    """
    def __init__(self, n=2, window_duration=5.0, stride_duration=2.0):
        self.n = n  # Number of agreements needed (unused for now)
        self.transcriptions = {}  # task_id -> text
        self.last_output_task = -1  # Track last task we output
        self.lock = threading.Lock()
        self.window_duration = window_duration  # seconds per window (configurable)
        self.stride_duration = stride_duration  # seconds between windows (configurable)
        
    def update(self, new_text, task_id):
        """Update with new transcription and return new text to output"""
        with self.lock:
            # Store the transcription
            self.transcriptions[task_id] = new_text
            logger.info(f"Stored task {task_id}: {new_text[:50]}...")
            
            # For sliding windows with 50% overlap:
            # Task 0 (0-3s): output first 50% (0-1.5s portion)
            # Task 1 (1.5-4.5s): output first 50% (1.5-3s portion)  
            # Task 2 (3-6s): output first 50% (3-4.5s portion)
            # This way each task outputs its unique non-overlapping portion
            
            if task_id == 0:
                # First task - output first half (0-1.5s)
                words = new_text.split()
                if words:
                    # Output first half of the words
                    half_idx = len(words) // 2
                    output_words = words[:half_idx]
                    output_text = " ".join(output_words)
                    
                    if output_text:
                        self.last_output_task = 0
                        logger.info(f"Outputting task 0 first half: {output_text[:50]}...")
                        return output_text + " "
            else:
                # For subsequent tasks, also output first half
                # Since each window overlaps 50% with the previous,
                # the first half of each window is the new unique portion
                words = new_text.split()
                if words:
                    # Output first half of the words
                    half_idx = len(words) // 2
                    output_words = words[:half_idx]
                    output_text = " ".join(output_words)
                    
                    if output_text:
                        self.last_output_task = task_id
                        logger.info(f"Outputting task {task_id} first half: {output_text[:50]}...")
                        return output_text + " "
            
            return None
    
    def find_overlap_agreement(self, text1, text2, overlap_start, overlap_duration):
        """Find agreement in the overlapping portion of two sliding windows"""
        # This method is kept for potential future use but not currently called
        # with the simplified sliding window approach
        return None


class WhisperStreamingDaemon:
    def __init__(
        self, algorithm="local-agreement", model_size="large-v3", no_streaming=False,
        window_duration=None, stride_duration=None
    ):
        self.sample_rate = 16000
        self.recording = False
        self.streaming = False
        self.interrupted = False
        self.model_loaded = False
        self.algorithm = algorithm
        self.model_size = model_size
        self.no_streaming = no_streaming  # If True, wait for complete recording

        # Algorithm-specific parameters with configurable window/stride
        if algorithm == "local-agreement":
            self.window_duration = window_duration or 5.0  # Default 5 seconds for better context
            self.stride_duration = stride_duration or 2.0  # Default 2 seconds (40% overlap)
            self.chunk_duration = 1.5  # Process 1.5-second chunks (faster)
            self.max_buffer_duration = 4.0  # Maximum buffer size (reduced)
        elif algorithm == "sliding-window":
            self.chunk_duration = 3.0  # Total chunk size to process
            self.stride_duration = 1.0  # Move forward by 1 second each time
        else:  # simple
            self.chunk_duration = 5.0  # Simple 5-second chunks

        self.silence_threshold = 0.01

        # Queues for processing
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()

        # Buffer for continuous audio
        self.audio_buffer = []
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)

        # Algorithm-specific state
        if algorithm == "local-agreement":
            self.local_agreement = LocalAgreementBuffer(
                n=2, 
                window_duration=self.window_duration,
                stride_duration=self.stride_duration
            )
        elif algorithm == "sliding-window":
            self.stride_samples = int(self.stride_duration * self.sample_rate)
            self.last_chunk_text = ""

        # Socket for IPC
        if no_streaming:
            self.socket_path = (
                f"/tmp/whisper_streaming_{algorithm}_nostream_daemon.sock"
            )
        else:
            self.socket_path = f"/tmp/whisper_streaming_{algorithm}_daemon.sock"
        self.server_socket = None

        # Preload sounds into memory for instant playback
        self.start_sound_path = (
            "/home/caleb/projects/whisper-dictation-daemon/snare.wav"
        )
        self.stop_sound_path = "/home/caleb/projects/whisper-dictation-daemon/hihat.wav"
        self.preload_sounds()

        # Load model once at startup
        self.load_model()

        # Start processing threads
        self.start_processing_threads()

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def preload_sounds(self):
        """Preload sound files into memory for instant playback"""
        try:
            # Load WAV files into memory
            rate1, self.start_sound_data = wavfile.read(self.start_sound_path)
            rate2, self.stop_sound_data = wavfile.read(self.stop_sound_path)

            # Convert to float32 for sounddevice
            self.start_sound_data = self.start_sound_data.astype(np.float32) / 32768.0
            self.stop_sound_data = self.stop_sound_data.astype(np.float32) / 32768.0

            self.sound_rate = rate1  # Assuming both files have same rate
            logger.info("Sound files preloaded into memory")
        except Exception as e:
            logger.error(f"Failed to preload sounds: {e}")
            # Fallback to file-based playback
            self.start_sound_data = None
            self.stop_sound_data = None

    def play_sound_memory(self, sound_data):
        """Play preloaded sound from memory using sounddevice"""
        if sound_data is not None:
            try:
                # Non-blocking play
                sd.play(sound_data, self.sound_rate)
            except Exception as e:
                logger.error(f"Failed to play sound from memory: {e}")

    def load_model(self):
        """Load the Whisper model once at startup"""
        logger.info(f"Loading faster-whisper {self.model_size} model on CPU...")
        logger.info("CPU: 24-core Ryzen 9 9900X")

        try:
            self.model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8", num_workers=4
            )
            self.model_loaded = True
            logger.info(f"Model loaded with {self.algorithm} algorithm!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)

    def start_processing_threads(self):
        """Start background threads for audio processing"""
        # Choose processor based on algorithm
        if self.algorithm == "local-agreement":
            processor = self.process_audio_local_agreement
        elif self.algorithm == "sliding-window":
            processor = self.process_audio_sliding_window
        else:
            processor = self.process_audio_simple

        self.audio_processor = threading.Thread(target=processor, daemon=True)
        self.audio_processor.start()

        # Thread for outputting transcriptions
        self.transcription_outputter = threading.Thread(
            target=self.output_transcriptions, daemon=True
        )
        self.transcription_outputter.start()

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        logger.info("Received shutdown signal")
        self.interrupted = True
        self.recording = False
        self.streaming = False
        if self.server_socket:
            self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        sys.exit(0)

    def play_sound(self, sound_file):
        """Play a sound file using fastest available method"""
        try:
            # Try paplay first (PulseAudio - fastest)
            subprocess.Popen(
                ["paplay", sound_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            try:
                # Fallback to aplay
                subprocess.Popen(
                    ["aplay", "-q", sound_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                # Final fallback to ffplay
                try:
                    subprocess.Popen(
                        [
                            "ffplay",
                            "-nodisp",
                            "-autoexit",
                            "-loglevel",
                            "quiet",
                            sound_file,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as e:
                    logger.error(f"Failed to play sound {sound_file}: {e}")

    def start_streaming_recording(self):
        """Start recording with streaming transcription"""
        # Clear log file for new recording
        with open("/tmp/whisper_streaming.log", "w") as f:
            f.write(
                f"=== New Recording Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            )

        # Set up everything FIRST
        self.audio_buffer = []

        # Reset algorithm-specific state
        if self.algorithm == "local-agreement":
            self.local_agreement = LocalAgreementBuffer(
                n=2,
                window_duration=self.window_duration,
                stride_duration=self.stride_duration
            )
        elif self.algorithm == "sliding-window":
            self.last_chunk_text = ""

        # Prepare the audio callback
        def audio_callback(indata, frames, time, status):
            if self.recording:
                # Add to buffer
                audio_chunk = indata.copy().flatten()
                self.audio_buffer.extend(audio_chunk)

                # If no_streaming mode, just collect audio, don't process yet
                if self.no_streaming:
                    return

                # Check if we have enough for processing
                if self.algorithm == "sliding-window":
                    if len(self.audio_buffer) >= self.chunk_samples:
                        # Extract full chunk
                        chunk = np.array(self.audio_buffer[: self.chunk_samples])
                        # Move forward by stride
                        self.audio_buffer = self.audio_buffer[self.stride_samples :]
                        self.audio_queue.put(chunk)
                elif self.algorithm == "local-agreement":
                    # For local-agreement, send small chunks regularly
                    # The processing thread will handle windowing
                    chunk_size = int(0.5 * self.sample_rate)  # Send 0.5s chunks
                    while len(self.audio_buffer) >= chunk_size:
                        chunk = np.array(self.audio_buffer[:chunk_size])
                        self.audio_buffer = self.audio_buffer[chunk_size:]
                        self.audio_queue.put(chunk)
                        logger.info(f"Sent {len(chunk)/self.sample_rate:.2f}s chunk to processing queue")
                else:  # simple
                    if len(self.audio_buffer) >= self.chunk_samples:
                        # Take the buffered audio
                        chunk_to_process = self.audio_buffer.copy()
                        # Clear the buffer
                        self.audio_buffer = []
                        self.audio_queue.put(np.array(chunk_to_process))

        # Now everything is ready - play the sound and start recording
        self.play_sound_memory(self.start_sound_data)
        self.recording = True
        self.streaming = not self.no_streaming

        if self.no_streaming:
            logger.info("Recording audio (non-streaming mode)... Press Ctrl to stop")
        else:
            logger.info(
                f"Starting streaming with {self.algorithm}... Press Ctrl to stop"
            )

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            dtype=np.float32,
        ):
            while self.recording and not self.interrupted:
                sd.sleep(100)

        # Handle end of recording differently for no_streaming mode
        if self.no_streaming:
            # Process the entire recording at once
            if self.audio_buffer:
                logger.info(
                    f"Processing complete recording ({len(self.audio_buffer) / self.sample_rate:.1f}s)..."
                )
                full_audio = np.array(self.audio_buffer)

                # Transcribe the entire audio
                try:
                    segments, info = self.model.transcribe(
                        full_audio,
                        language="en",
                        beam_size=5,
                        temperature=0.0,
                        vad_filter=True,
                    )

                    # Output all text at once
                    full_text = " ".join([segment.text.strip() for segment in segments])
                    if full_text:
                        self.transcription_queue.put(full_text)
                        logger.info("Transcription complete")

                except Exception as e:
                    logger.error(f"Transcription error: {e}")

            self.transcription_queue.put(None)
        else:
            # Process any remaining audio for streaming mode
            # Send all remaining audio in small chunks
            if self.algorithm == "local-agreement":
                # Send any remaining audio as 0.5s chunks
                chunk_size = int(0.5 * self.sample_rate)
                while len(self.audio_buffer) >= chunk_size:
                    chunk = np.array(self.audio_buffer[:chunk_size])
                    self.audio_buffer = self.audio_buffer[chunk_size:]
                    self.audio_queue.put(chunk)
                    logger.info(f"Sent final {len(chunk)/self.sample_rate:.2f}s chunk")
                # Send any leftover audio that's less than 0.5s
                if len(self.audio_buffer) > 0:
                    final_chunk = np.array(self.audio_buffer)
                    self.audio_queue.put(final_chunk)
                    logger.info(f"Sent final {len(final_chunk)/self.sample_rate:.2f}s chunk")
            else:
                # For other algorithms, send remaining audio if significant
                if len(self.audio_buffer) > int(0.5 * self.sample_rate):
                    final_chunk = np.array(self.audio_buffer)
                    self.audio_queue.put(final_chunk)

            # Signal end of stream
            self.audio_queue.put(None)

        self.streaming = False

        # Stop sound already played in handle_client for faster response

        # Clean up recording flag file
        if self.no_streaming:
            flag_file = "/tmp/whisper_nostream_recording"
        else:
            flag_file = "/tmp/whisper_stream_recording"

        if os.path.exists(flag_file):
            os.remove(flag_file)

        logger.info("Recording stopped")

    def process_audio_local_agreement(self):
        """Process audio chunks with Parallel Local Agreement algorithm using fixed-size sliding windows"""
        # This thread runs continuously for the lifetime of the daemon
        while not self.interrupted:
            # Reset state for each recording session
            all_audio = []  # Accumulate all audio
            last_full_text = ""
            task_counter = 0
            
            # Use configurable window size and stride for consistent transcription
            window_duration = self.window_duration  # Use instance variable
            stride_duration = self.stride_duration  # Use instance variable
            window_samples = int(window_duration * self.sample_rate)
            stride_samples = int(stride_duration * self.sample_rate)
            
            # Track last submission position
            last_submission_pos = 0
            
            # Thread pool for parallel transcriptions
            executor = ThreadPoolExecutor(max_workers=3)  # 3 parallel transcriptions
            pending_tasks = {}  # task_id -> Future
            completed_results = {}  # task_id -> transcription text
            
            # Flag to stop completion thread
            self.stop_completion_thread = False
            
            # Separate thread for handling completed transcriptions
            def process_completed_transcriptions():
                processed_tasks = set()
                
                while not self.stop_completion_thread:
                    # Process any available results
                    for task_id in list(completed_results.keys()):
                        if task_id not in processed_tasks:
                            text = completed_results[task_id]
                            
                            # Update local agreement buffer
                            confirmed_text = self.local_agreement.update(text, task_id)
                            
                            if confirmed_text:
                                self.transcription_queue.put(confirmed_text)
                                logger.info(f"Confirmed: {confirmed_text[:50]}...")
                            
                            processed_tasks.add(task_id)
                            del completed_results[task_id]
                        
                    time.sleep(0.01)  # Small delay to avoid busy waiting
            
            # Start the completion handler thread
            completion_thread = threading.Thread(target=process_completed_transcriptions, daemon=True)
            completion_thread.start()
            
            def transcribe_audio(task):
                """Worker function for thread pool"""
                try:
                    logger.info(f"Task {task.task_id}: Transcribing {task.context_length:.1f}s")
                    segments, info = self.model.transcribe(
                        task.audio_data,
                        language="en",
                        beam_size=5,
                        temperature=0.0,
                        vad_filter=True,
                        without_timestamps=True
                    )
                    
                    full_text = " ".join([segment.text.strip() for segment in segments])
                    logger.info(f"Task {task.task_id}: Complete")
                    return task.task_id, full_text
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")
                    return task.task_id, ""
            
            while True:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)  # Shorter timeout for responsiveness
                    
                    if chunk is None:
                        # Shutdown signal - wait for all tasks to complete
                        logger.info("Waiting for all transcription tasks to complete...")
                        executor.shutdown(wait=True)
                        
                        # Wait for all results to be processed through Local Agreement
                        logger.info("Processing final results through Local Agreement...")
                        while completed_results or pending_tasks:
                            # Check for any remaining completed tasks
                            for task_id, future in list(pending_tasks.items()):
                                if future.done():
                                    try:
                                        tid, text = future.result(timeout=0)
                                        completed_results[tid] = text
                                        del pending_tasks[task_id]
                                    except:
                                        del pending_tasks[task_id]
                            
                            # Give completion thread time to process
                            time.sleep(0.1)
                        
                        # Now stop the completion thread
                        self.stop_completion_thread = True
                        time.sleep(0.1)  # Give thread time to finish
                        
                        # Signal end of stream
                        self.transcription_queue.put(None)
                        break  # Break inner loop to start fresh for next recording
                    
                    # Skip if too quiet
                    if np.max(np.abs(chunk)) < self.silence_threshold:
                        continue
                    
                    # Add to all audio
                    all_audio.extend(chunk)
                    logger.info(f"Received chunk, total audio: {len(all_audio)/self.sample_rate:.1f}s")
                    
                    # Check if we can submit new tasks with fixed-size windows
                    while last_submission_pos + window_samples <= len(all_audio):
                        # Extract fixed-size window
                        start_pos = last_submission_pos
                        end_pos = start_pos + window_samples
                        window_audio = np.array(all_audio[start_pos:end_pos])
                        
                        # Create transcription task with fixed window
                        task = TranscriptionTask(
                            task_id=task_counter,
                            audio_data=window_audio,
                            timestamp=time.time(),
                            context_length=window_duration
                        )
                        
                        # Submit task to thread pool (non-blocking)
                        future = executor.submit(transcribe_audio, task)
                        pending_tasks[task_counter] = future
                        
                        # Log with position info
                        logger.info(f"Submitted task {task.task_id}: {window_duration:.1f}s window at pos {start_pos/self.sample_rate:.1f}s-{end_pos/self.sample_rate:.1f}s")
                        
                        task_counter += 1
                        
                        # Move forward by stride
                        last_submission_pos += stride_samples
                    
                    # Clean up old audio to prevent memory issues
                    # Keep last 30 seconds plus window size
                    max_keep_samples = int(30 * self.sample_rate) + window_samples
                    if len(all_audio) > max_keep_samples:
                        # Calculate how much to remove
                        remove_samples = len(all_audio) - max_keep_samples
                        all_audio = all_audio[remove_samples:]
                        # Adjust submission position
                        last_submission_pos -= remove_samples
                        if last_submission_pos < 0:
                            last_submission_pos = 0
                    
                    # Check for completed tasks
                    completed = []
                    for task_id, future in pending_tasks.items():
                        if future.done():
                            try:
                                tid, text = future.result(timeout=0)
                                completed_results[tid] = text
                                completed.append(task_id)
                                last_full_text = text
                            except:
                                pass
                    
                    # Clean up completed tasks
                    for task_id in completed:
                        del pending_tasks[task_id]
                    
                    # Log status
                    if len(pending_tasks) > 0:
                        logger.info(f"Tasks in flight: {len(pending_tasks)}, Completed waiting: {len(completed_results)}")
                        
                except queue.Empty:
                    # Check for completed tasks even when no new audio
                    completed = []
                    for task_id, future in pending_tasks.items():
                        if future.done():
                            try:
                                tid, text = future.result(timeout=0)
                                completed_results[tid] = text
                                completed.append(task_id)
                                last_full_text = text
                            except:
                                pass
                    
                    for task_id in completed:
                        del pending_tasks[task_id]
                        
                except Exception as e:
                    logger.error(f"Audio processing error: {e}")

    def process_audio_sliding_window(self):
        """Process audio with sliding window and timestamp-based deduplication"""
        while not self.interrupted:
            try:
                chunk = self.audio_queue.get(timeout=1)

                if chunk is None:
                    self.transcription_queue.put(None)
                    continue

                # Skip if too quiet
                if np.max(np.abs(chunk)) < self.silence_threshold:
                    continue

                logger.info(f"Processing {len(chunk) / self.sample_rate:.1f}s chunk...")

                try:
                    segments, info = self.model.transcribe(
                        chunk,
                        language="en",
                        beam_size=5,
                        temperature=0.0,
                        vad_filter=True,
                        without_timestamps=False,  # Keep timestamps for alignment
                    )

                    # Only output text from the middle portion
                    for segment in segments:
                        # Check if segment is in our target window (0.5s to 1.5s)
                        if segment.start >= 0.5 and segment.start < 1.5:
                            text = segment.text.strip()
                            if text:
                                self.transcription_queue.put(text + " ")
                                logger.info(f"Streamed: {text}")

                except Exception as e:
                    logger.error(f"Transcription error: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")

    def process_audio_simple(self):
        """Simple processing - transcribe chunks as they come"""
        while not self.interrupted:
            try:
                chunk = self.audio_queue.get(timeout=1)

                if chunk is None:
                    self.transcription_queue.put(None)
                    continue

                # Skip if too quiet
                if np.max(np.abs(chunk)) < self.silence_threshold:
                    continue

                logger.info(f"Processing {len(chunk) / self.sample_rate:.1f}s chunk...")

                try:
                    segments, info = self.model.transcribe(
                        chunk,
                        language="en",
                        beam_size=5,
                        temperature=0.0,
                        vad_filter=True,
                        without_timestamps=True,
                    )

                    # Collect text
                    text = " ".join([segment.text.strip() for segment in segments])

                    if text:
                        self.transcription_queue.put(text + " ")
                        logger.info(f"Transcribed: {text}")

                except Exception as e:
                    logger.error(f"Transcription error: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")

    def output_transcriptions(self):
        """Output transcriptions as they become available"""
        while not self.interrupted:
            try:
                text = self.transcription_queue.get(timeout=1)

                if text is None:
                    logger.info("End of transcription stream")
                    continue

                # Type the text immediately
                self.type_text(text)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription output error: {e}")

    def type_text(self, text):
        """Type text using wtype"""
        if text:
            try:
                subprocess.run(["wtype", text], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to type text: {e}")

    def process_file(self, file_path):
        """Process an audio file for testing"""
        logger.info(f"Processing file: {file_path}")

        # Convert if MP3
        if file_path.endswith(".mp3"):
            logger.info("Converting MP3 to WAV...")
            wav_path = self.convert_to_wav(file_path)
        else:
            wav_path = file_path

        # Load audio
        sample_rate, audio = wavfile.read(wav_path)

        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        logger.info(f"Audio loaded: {len(audio) / sample_rate:.1f} seconds")

        # Process based on algorithm
        if self.algorithm == "local-agreement":
            self.process_file_local_agreement(audio)
        elif self.algorithm == "sliding-window":
            self.process_file_sliding_window(audio)
        else:
            self.process_file_simple(audio)

    def convert_to_wav(self, mp3_file):
        """Convert MP3 to WAV using ffmpeg"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name

        cmd = [
            "ffmpeg",
            "-i",
            mp3_file,
            "-ar",
            "16000",  # 16kHz sample rate
            "-ac",
            "1",  # Mono
            "-f",
            "wav",
            "-y",  # Overwrite
            wav_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr}")
            raise Exception(f"Failed to convert MP3: {result.stderr}")

        return wav_path

    def process_file_local_agreement(self, audio):
        """Process file with local agreement"""
        local_agreement = LocalAgreementBuffer(n=2)
        position = 0
        task_id = 0
        chunk_samples = int(2.0 * self.sample_rate)
        context_audio = []
        last_full_text = ""

        while position < len(audio):
            # Get next chunk
            end = min(position + chunk_samples, len(audio))
            chunk = audio[position:end]
            context_audio.extend(chunk)

            # Limit context
            max_samples = int(10.0 * self.sample_rate)
            if len(context_audio) > max_samples:
                context_audio = context_audio[-max_samples:]

            # Transcribe
            segments, _ = self.model.transcribe(
                np.array(context_audio),
                language="en",
                beam_size=5,
                without_timestamps=True,
            )

            full_text = " ".join([s.text.strip() for s in segments])

            if full_text:
                confirmed = local_agreement.update(full_text, task_id)
                task_id += 1
                if confirmed:
                    print(confirmed, end=" ", flush=True)
                last_full_text = full_text

            position = end

        # Output any remaining text at the end
        # Since we're using sliding windows, just output any final text
        if last_full_text:
            remaining = last_full_text
            print(remaining, end=" ", flush=True)

        print()  # Final newline

    def process_file_sliding_window(self, audio):
        """Process file with sliding window"""
        chunk_samples = int(3.0 * self.sample_rate)
        stride_samples = int(1.0 * self.sample_rate)
        position = 0

        while position + chunk_samples <= len(audio):
            chunk = audio[position : position + chunk_samples]

            segments, _ = self.model.transcribe(
                chunk, language="en", beam_size=5, without_timestamps=False
            )

            # Output middle portion
            for segment in segments:
                if segment.start >= 0.5 and segment.start < 1.5:
                    text = segment.text.strip()
                    if text:
                        print(text, end=" ", flush=True)

            position += stride_samples

        print()  # Final newline

    def process_file_simple(self, audio):
        """Simple file processing"""
        segments, _ = self.model.transcribe(audio, language="en", beam_size=5)

        for segment in segments:
            print(segment.text.strip(), end=" ", flush=True)
        print()

    def handle_client(self, client_socket):
        """Handle client requests"""
        try:
            data = client_socket.recv(1024).decode()
            if data == "STREAM_START":
                if not self.streaming and not self.recording:
                    # Send response FIRST for fastest client return
                    if self.no_streaming:
                        client_socket.send(b"RECORDING")
                    else:
                        client_socket.send(b"STREAMING")
                    # Start recording thread (it will play sound when ready)
                    threading.Thread(target=self.start_streaming_recording).start()
                else:
                    if self.no_streaming:
                        client_socket.send(b"ALREADY_RECORDING")
                    else:
                        client_socket.send(b"ALREADY_STREAMING")
            elif data == "STREAM_STOP":
                # Send response first
                client_socket.send(b"STOPPED")
                # Then play sound and stop
                self.play_sound_memory(self.stop_sound_data)
                self.recording = False
            elif data == "STATUS":
                if self.recording:
                    status = "RECORDING" if self.no_streaming else "STREAMING"
                elif self.model_loaded:
                    status = "READY"
                else:
                    status = "LOADING"
                client_socket.send(status.encode())
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            client_socket.close()

    def start_daemon(self):
        """Start the daemon server"""
        logger.info(f"Starting Whisper daemon with {self.algorithm} algorithm...")

        # Remove existing socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Create Unix socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)

        logger.info(f"Daemon listening on {self.socket_path}")
        logger.info(f"Algorithm: {self.algorithm}, Model: {self.model_size}")

        while not self.interrupted:
            try:
                client_socket, _ = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client, args=(client_socket,)
                )
                client_thread.daemon = True
                client_thread.start()
            except OSError:
                if not self.interrupted:
                    logger.error("Socket error")
                break


def main():
    parser = argparse.ArgumentParser(description="Whisper Streaming Daemon")
    parser.add_argument(
        "--algorithm",
        "-a",
        choices=["local-agreement", "sliding-window", "simple"],
        default="local-agreement",
        help="Streaming algorithm to use (default: local-agreement)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="large-v3",
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--no-streaming",
        "-n",
        action="store_true",
        help="Wait for complete recording before transcribing (non-streaming mode)",
    )
    parser.add_argument(
        "--window-duration",
        "-w",
        type=float,
        default=None,
        help="Window duration in seconds for local-agreement algorithm (default: 5.0)",
    )
    parser.add_argument(
        "--stride-duration",
        "-s",
        type=float,
        default=None,
        help="Stride duration in seconds for local-agreement algorithm (default: 2.0)",
    )
    parser.add_argument(
        "--file", "-f", help="Process an audio file instead of starting daemon"
    )
    parser.add_argument(
        "--client",
        "-c",
        choices=["start", "stop", "status", "toggle"],
        help="Send command to running daemon",
    )

    args = parser.parse_args()

    # Client mode
    if args.client:
        if args.no_streaming:
            socket_path = (
                f"/tmp/whisper_streaming_{args.algorithm}_nostream_daemon.sock"
            )
        else:
            socket_path = f"/tmp/whisper_streaming_{args.algorithm}_daemon.sock"
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(socket_path)

            if args.client == "start":
                client.send(b"STREAM_START")
            elif args.client == "stop":
                client.send(b"STREAM_STOP")
            elif args.client == "status":
                client.send(b"STATUS")
            elif args.client == "toggle":
                # First check status
                client.send(b"STATUS")
                status = client.recv(1024).decode()
                client.close()

                # Then toggle
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.connect(socket_path)
                if status in ["STREAMING", "RECORDING"]:
                    client.send(b"STREAM_STOP")
                    print(
                        f"Stopped {'recording' if status == 'RECORDING' else 'streaming'}"
                    )
                else:
                    client.send(b"STREAM_START")
                    print(
                        f"Started {'recording' if args.no_streaming else 'streaming'}"
                    )

            response = client.recv(1024).decode()
            if args.client != "toggle":
                print(f"Response: {response}")
            client.close()

        except FileNotFoundError:
            print(
                f"Daemon not running! Start with: python3 whisper_streaming.py -a {args.algorithm}"
            )
        except Exception as e:
            print(f"Error: {e}")

        return

    # Create daemon with optional window/stride configuration
    daemon = WhisperStreamingDaemon(
        algorithm=args.algorithm, 
        model_size=args.model, 
        no_streaming=args.no_streaming,
        window_duration=args.window_duration,
        stride_duration=args.stride_duration
    )

    # File processing mode
    if args.file:
        daemon.process_file(args.file)
    else:
        # Daemon mode
        daemon.start_daemon()


if __name__ == "__main__":
    main()
