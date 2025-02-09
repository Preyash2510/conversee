from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from speech_recognition import Recognizer, Microphone
from elevenlabs import play
from elevenlabs.client import ElevenLabs
import os
import sys
import time
import threading
import textwrap
import re
import uuid
import atexit
from pydub import AudioSegment
import io

class SpeechChat:
    """
    A class that implements a speech-based chat interface using LangChain, Ollama, and ElevenLabs.
    
    This class enables voice-based interaction with an AI model, converting speech to text,
    processing it through an LLM, and converting the response back to speech.
    
    Attributes:
        llm (ChatOllama): The language model instance using Ollama
        elevenLabs (ElevenLabs): Text-to-speech service client
        r (Recognizer): Speech recognition instance
        workflow (StateGraph): LangChain workflow graph
        done_loading (bool): Flag to control loading animation
        user_input (list): History of user messages
        ai_output (list): History of AI responses
        thread_id (UUID): Unique identifier for the chat session
        pause_threshold (float): Silence threshold for speech recognition
        wrap_width (int): Text wrap width for display
        speak (bool): Whether to enable text-to-speech
        output_to_file (bool): Whether to save chat transcript
        output_audio_to_file (bool): Whether to save audio recordings
        output_file (file): File handle for chat transcript
        output_audio_file_path (str): Path to save audio recordings
    """

    def __init__(self, pause_threshold=0.8, wrap_width=60, speak=True, output_path="", 
                 output_audio_path="", output_to_file=False, output_audio_to_file=False):
        """
        Initialize the SpeechChat instance.
        
        Args:
            pause_threshold (float): Silence threshold for speech recognition. Defaults to 0.8.
            wrap_width (int): Text wrap width for display. Defaults to 60.
            speak (bool): Enable text-to-speech output. Defaults to True.
            output_path (str): Custom path for chat transcript. Defaults to "".
            output_audio_path (str): Custom path for audio recordings. Defaults to "".
            output_to_file (bool): Enable chat transcript saving. Defaults to False.
            output_audio_to_file (bool): Enable audio recording saving. Defaults to False.
        """
        # Create required directories in one go
        for directory in ['recordings', 'chats', 'temp_audio']:
            os.makedirs(f"output/{directory}", exist_ok=True)
            
        # Initialize core components
        self.llm = ChatOllama(model="deepseek-r1:7b")
        self.elevenLabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        self.r = Recognizer()
        self.workflow = StateGraph(MessagesState)
        
        # Initialize state variables
        self.done_loading = False
        self.user_input = []
        self.ai_output = []
        self.thread_id = uuid.uuid4()
        self.config = {"configurable": {"thread_id": self.thread_id}}
        
        # Store configuration
        self.pause_threshold = pause_threshold
        self.wrap_width = wrap_width
        self.speak = speak
        self.output_to_file = output_to_file
        self.output_audio_to_file = output_audio_to_file

        # Setup output files
        if self.output_to_file:
            self.output_file = open(
                output_path or f"output/chats/{self.thread_id}.txt", 
                "a", 
                encoding="utf-8"
            )

        if self.output_audio_to_file:
            self.output_audio_file_path = (
                output_audio_path if output_audio_path and output_audio_path.endswith(".wav")
                else f"{output_audio_path or f'output/recordings/{self.thread_id}'}.wav"
            )
            self._initialize_wav_file()

        atexit.register(self.cleanup)


    def _initialize_wav_file(self):
        """Initialize an empty WAV file with proper headers."""
        with open(self.output_audio_file_path, "wb") as f:
            # Write WAV header structure
            headers = [
                (b'RIFF', b'\x00\x00\x00\x00', b'WAVE'),
                (b'fmt ', (16).to_bytes(4, 'little')),
                ((1).to_bytes(2, 'little'),  # audio format (PCM)
                 (1).to_bytes(2, 'little'),  # num channels (mono)
                 (44100).to_bytes(4, 'little'),  # sample rate
                 (88200).to_bytes(4, 'little'),  # byte rate
                 (2).to_bytes(2, 'little'),  # block align
                 (16).to_bytes(2, 'little')),  # bits per sample
                (b'data', b'\x00\x00\x00\x00')  # data chunk header
            ]
            for parts in headers:
                f.write(b''.join(parts))


    def __exit__(self):
        """Cleanup resources when exiting the context manager."""
        self.cleanup()


    def cleanup(self):
        """
        Clean up resources and temporary files.
        
        Closes open file handles and removes temporary audio files.
        """
        if self.output_to_file:
            self.output_file.close()
        # Clean up temp_audio directory
        if os.path.exists('output/temp_audio'):
            for file in os.listdir('output/temp_audio'):
                try:
                    os.remove(os.path.join('output/temp_audio', file))
                except:
                    pass


    def __call_model(self, state: MessagesState):
        """
        Invoke the language model with the current message state.
        
        Args:
            state (MessagesState): Current state containing message history.
            
        Returns:
            dict: Dictionary containing the model's response message.

        Example:
            >>> state = {"messages": [HumanMessage(content="Hello!")]}
            >>> result = chat._SpeechChat__call_model(state)
            >>> print(result)
            {'messages': [AIMessage(content="Hi! How can I help you today?")]}
        """
        response = self.llm.invoke(state["messages"])
        return {"messages": [response]}
    
    
    def __animate_listening(self):
        """
        Display an animated loading indicator while listening for user input.
        
        This method runs in a separate thread and shows a dots animation
        until self.done_loading is set to True.

        Example:
            >>> thread = threading.Thread(target=chat._SpeechChat__animate_listening)
            >>> thread.start()
            Listening...
            >>> chat.done_loading = True
            >>> thread.join()
        """
        dots = ''
        while not self.done_loading:
            if len(dots) < 3:
                dots += '.'
            else:
                dots = ''
            sys.stdout.write('\rListening' + dots + ' ' * (3 - len(dots)))
            sys.stdout.flush()
            time.sleep(0.5)
        sys.stdout.write('\r' + ' ' * 20 + '\r')  # Clear the line
        sys.stdout.flush()


    def __clean_output(self, text: str) -> str:
        """
        Clean and format the AI model's output text.
        
        Removes LaTeX markers, commands, markdown formatting, and normalizes whitespace.
        
        Args:
            text (str): Raw text from the AI model.
            
        Returns:
            str: Cleaned and formatted text.

        Example:
            >>> raw_text = "\\boxed{Hello} \\(x^2\\) **world**"
            >>> cleaned = chat._SpeechChat__clean_output(raw_text)
            >>> print(cleaned)
            'Hello x^2 world'
        """
        # Remove LaTeX math mode markers
        text = re.sub(r'\\\(|\\\)', '', text)
        text = re.sub(r'\\\[|\\\]', '', text)
        
        # Remove LaTeX commands
        text = re.sub(r'\\boxed\{([^}]*)\}', r'\1', text)
        
        # Remove markdown bold markers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        
        # Remove any remaining LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Clean up multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    
    def __print_conversation(self, text: str, user: bool):
        """
        Print formatted conversation messages to the console.
        
        Args:
            text (str): The message text to display.
            user (bool): True if the message is from the user, False if from AI.

        Example:
            >>> chat._SpeechChat__print_conversation("Hello!", True)
            User:
                      Hello!
            >>> chat._SpeechChat__print_conversation("Hi there!", False)
            AI:
                      Hi there!
        """
        wrapped_text = textwrap.fill(text, width=self.wrap_width)
        formatted_text = "\n          ".join(wrapped_text.split('\n'))
        
        if user:
            print_text = "\033[92mUser:\n          {}\033[0m".format(formatted_text)
            file_text = "User:\n          {}\n".format(formatted_text)
        else:
            print_text = "\033[93mAI:\n          {}\033[0m".format(formatted_text)
            file_text = "AI:\n          {}\n".format(formatted_text)

        if self.output_to_file:
            self.output_file.write(file_text)

        print(print_text)


    def __save_audio(self, audio, is_user_audio=True):
        """
        Save audio segments to the conversation recording file.
        
        Args:
            audio (bytes or AudioData): Audio data to save
            is_user_audio (bool): Whether the audio is from user (WAV) or AI (MP3)
        """
        temp_filename = os.path.join('output/temp_audio', str(uuid.uuid4()))
        temp_ext = ".wav" if is_user_audio else ".mp3"
        temp_path = temp_filename + temp_ext
        
        try:
            # Save audio to temporary file
            with open(temp_path, "wb") as f:
                if is_user_audio and not isinstance(audio, bytes):
                    audio = b''.join(audio)
                f.write(audio)
            
            # Convert to AudioSegment
            audio_segment = (AudioSegment.from_wav(temp_path) if is_user_audio 
                           else AudioSegment.from_mp3(temp_path))
            
            # Append or create output file
            if os.path.exists(self.output_audio_file_path):
                existing_audio = AudioSegment.from_wav(self.output_audio_file_path)
                audio_segment = existing_audio + audio_segment
            
            audio_segment.export(self.output_audio_file_path, format="wav")
            
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except (OSError, FileNotFoundError):
                pass

        if os.getenv("DEBUG"):
            print(f"File size: {os.path.getsize(self.output_audio_file_path)} bytes")


    def start_speech_chat(self):
        """
        Start an interactive speech chat session.
        
        Initializes the conversation workflow and begins the main interaction loop.
        Handles speech recognition, AI model interaction, and text-to-speech conversion.
        The session continues until the user says "goodbye".
        
        The conversation flow:
        1. Listen for user speech input
        2. Convert speech to text
        3. Process text through AI model
        4. Convert AI response to speech (if enabled)
        5. Save transcript and audio (if enabled)
        """
        # Initialize workflow once
        self.workflow.add_edge(START, "model")
        self.workflow.add_node('model', self.__call_model)
        app = self.workflow.compile(checkpointer=MemorySaver())

        while True:
            # Handle speech recognition
            prompt = self._get_user_input()
            if prompt is None:  # Recognition failed
                continue

            # Process through AI model
            self._process_ai_response(prompt, app)

            # Check for goodbye after processing response
            if 'goodbye' in prompt.lower():
                break


    def _get_user_input(self):
        """
        Handle speech recognition and return the recognized text.
        
        Returns:
            str: Recognized text or None if recognition fails
        """
        self.done_loading = False
        loading = threading.Thread(target=self.__animate_listening)

        with Microphone() as source:
            self.r.adjust_for_ambient_noise(source)
            loading.start()
            audio = self.r.listen(source)

        self.done_loading = True
        loading.join()

        try:
            prompt = self.r.recognize_google(audio)
            self.__save_audio(audio.get_wav_data(), is_user_audio=True)
            self.user_input.append(prompt)
            self.__print_conversation(prompt, True)
            return prompt
        except Exception as e:
            print("Could not understand audio")
            return None


    def _process_ai_response(self, prompt, app):
        """
        Process user input through AI model and handle response.
        
        Args:
            prompt (str): User input text
            app (StateGraph): Compiled application state graph
        """
        input_message = HumanMessage(content=prompt)
        res = app.invoke({"messages": [input_message]}, config=self.config)

        ai_message = res["messages"][-1].content
        ai_message = self.__clean_output(
            re.sub(r"<think>.*?</think>", "", ai_message, flags=re.DOTALL).strip()
        )
        
        self.ai_output.append(ai_message)
        self.__print_conversation(ai_message, False)

        if self.speak:
            self._handle_tts(ai_message)


    def _handle_tts(self, message):
        """
        Handle text-to-speech conversion and playback.
        
        Args:
            message (str): Text to convert to speech
        """
        audio = self.elevenLabs.text_to_speech.convert(
            text=message,
            voice_id=os.getenv("VOICE_ID"),
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        # Convert generator to bytes and play
        audio_bytes = b''.join(audio)
        play(audio_bytes)

        if self.output_audio_to_file:
            self.__save_audio(audio_bytes, is_user_audio=False)
            
