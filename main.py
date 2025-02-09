"""Speech-based Chat Interface

This module provides a command-line interface for interacting with an AI assistant through speech.
It supports both real-time microphone input and audio file processing.

The interface leverages the SpeechChat class to handle:
- Speech recognition
- AI model interaction
- Text-to-speech conversion
- Audio file management
- Conversation history tracking

Usage:
    python main.py [options]

Options:
    --output_to_file          Save chat transcript to file
    --output_path            Path for saving chat transcript
    --output_audio_to_file   Save conversation audio to file
    --output_audio_path      Path for saving audio recording
    --pause_threshold        Silence duration to end recording (default: 0.8s)
    --wrap_width            Text display width (default: 60)
    --speak                 Enable AI voice response (default: True)
    --no-speak              Disable AI voice response

Examples:
    # Start basic voice chat session
    python main.py

    # Save transcript and audio, custom pause threshold
    python main.py --output_to_file --output_audio_to_file --pause_threshold 1.2

    # Silent mode with saved transcript
    python main.py --no-speak --output_to_file --output_path "chat_log.txt"
"""

import argparse
import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from SpeechChat import SpeechChat

# Load environment variables at module level
load_dotenv()

def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Configure and return the argument parser for command line options.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Speech-based chat interface with AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Output configuration
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        "--output_to_file", 
        action='store_true',
        help="Save chat transcript to file"
    )
    output_group.add_argument(
        "--output_path", 
        type=str,
        help="Path for saving chat transcript"
    )
    output_group.add_argument(
        "--output_audio_to_file", 
        action='store_true',
        help="Save conversation audio to file"
    )
    output_group.add_argument(
        "--output_audio_path", 
        type=str,
        help="Path for saving audio recording"
    )

    # Recognition configuration
    recog_group = parser.add_argument_group('Recognition Options')
    recog_group.add_argument(
        "--pause_threshold", 
        type=float, 
        default=0.8,
        help="Silence duration to end recording (default: 0.8s)"
    )
    recog_group.add_argument(
        "--wrap_width", 
        type=int, 
        default=60,
        help="Text display width (default: 60)"
    )

    # TTS configuration
    tts_group = parser.add_argument_group('Text-to-Speech Options')
    tts_group.add_argument(
        "--speak", 
        action='store_true', 
        default=True,
        help="Enable AI voice response (default: True)"
    )
    tts_group.add_argument(
        "--no-speak", 
        action='store_false', 
        dest='speak',
        help="Disable AI voice response"
    )

    return parser


def main(args: argparse.Namespace) -> None:
    """
    Initialize and start the speech chat interface.

    Args:
        args (argparse.Namespace): Validated command line arguments

    The function creates a SpeechChat instance with the specified configuration
    and starts the interactive chat session.
    """
    try:
        chat = SpeechChat(
            pause_threshold=args.pause_threshold,
            wrap_width=args.wrap_width,
            speak=args.speak,
            output_path=args.output_path,
            output_audio_path=args.output_audio_path,
            output_to_file=args.output_to_file,
            output_audio_to_file=args.output_audio_to_file
        )
        chat.start_speech_chat()
    except KeyboardInterrupt:
        print("\nExiting chat session...")
    except Exception as e:
        print(f"Error during chat session: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        main(args)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)

        sys.exit(1)
