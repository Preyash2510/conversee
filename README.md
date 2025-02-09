# Conversee - AI Speech Chat

A real-time speech-based chat interface that enables natural voice conversations with an AI assistant. This application combines speech recognition, AI language processing, and text-to-speech capabilities to create an interactive voice-based experience.

## Features

- 🎤 Real-time speech recognition
- 🤖 AI-powered conversation using LangChain and Ollama
- 🔊 Text-to-speech responses using ElevenLabs
- 📝 Optional conversation transcription
- 🎵 Optional audio recording of entire conversations
- ⚙️ Configurable speech recognition parameters
- 🎯 Clean exit handling and resource management

## Prerequisites

### Required Software
- Python 3.11
- Ollama (for AI model)
- FFmpeg (for audio processing)

### API Keys
You'll need to set up the following environment variables:
- `ELEVENLABS_API_KEY`: Your ElevenLabs API key
- `VOICE_ID`: Your chosen ElevenLabs voice ID

Create a `.env` file in the project root:
```
ELEVENLABS_API_KEY=your_elevenlabs_api_key
VOICE_ID=your_voice_id
```

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/conversee.git
cd conversee
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```
ELEVENLABS_API_KEY=your_elevenlabs_api_key
VOICE_ID=your_voice_id
```

## Usage

```
python main.py
```

## Command Line Arguments

The application supports the following command line arguments:

- `--output_to_file`: Save chat transcript to file
- `--output_path`: Path for saving chat transcript
- `--output_audio_to_file`: Save conversation audio to file
- `--output_audio_path`: Path for saving audio recording


## Examples

### Basic Usage

```
python main.py
```

### Save Transcript and Audio

```
python main.py --output_to_file --output_audio_to_file
```

### Silent Mode with Saved Transcript

```
python main.py --no-speak --output_to_file --output_path "chat_log.txt"
```

### Custom Pause Threshold

```
python main.py --pause_threshold 1.2
```

### Disable Text-to-Speech for AI responses

```
python main.py --no-speak
```

### Save Transcript and Audio with Custom Paths

```
python main.py --output_to_file --output_audio_to_file --output_path "chat_log.txt" --output_audio_path "recordings/conversation.wav"
```

## Directory Structure

```
conversee/
├── main.py
├── requirements.txt
├── .env
├── SpeechChat.py
├── .venv/
├── output/
│ ├── chats/
│ │ └── chat_id.txt
│ ├── recordings/
│ │ └── conversation.wav
│ └── temp_audio/
│ └── temp_audio/
```