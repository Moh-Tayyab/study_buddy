# 📚 Study Buddy - Your Smart Study Assistant 🤖
> Stay focused and alert during your study sessions with AI-powered attention monitoring!
Study Buddy is an innovative Python application that leverages computer vision technology to enhance your study experience. Using advanced facial recognition, it monitors your attention levels in real-time and helps you maintain peak focus. 🎯
## ✨ Key Features
- 👁️ **Eye Closure Detection**: Smart alerts when eyes remain closed (>2 seconds)
- 🥱 **Yawn Detection**: Intelligent fatigue monitoring system
- 📊 **Real-time Analytics**: Live display of attention metrics
- 🔔 **Smart Alerts**: Customizable audio notifications
- 📝 **Detailed Logging**: Comprehensive study session tracking
## 🛠️ System Requirements
- 🐍 Python 3.9 or higher
- 📷 Working webcam
- 📦 Dependencies (see `requirements.txt`)
## 🚀 Getting Started
### Installation
- Webcam
- Packages listed in `requirements.txt`

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application with default settings:

```bash
python study_buddy.py
```

### Command-line Options

Study Buddy accepts several command-line arguments to customize its behavior:

```
usage: study_buddy.py [-h] [--camera CAMERA] [--eye-threshold EYE_THRESHOLD] [--yawn-threshold YAWN_THRESHOLD] [--log LOG] [--sound SOUND]

Study Buddy - Attention Monitoring System

optional arguments:
  -h, --help            show this help message and exit
  --camera CAMERA       Camera device ID (default: 0)
  --eye-threshold EYE_THRESHOLD
                        Eye closure threshold in seconds (default: 2.0)
  --yawn-threshold YAWN_THRESHOLD
                        Yawn detection threshold (ratio, default: 0.5)
  --log LOG             Log file path (default: study_buddy_log.txt)
  --sound SOUND         Path to custom alert sound file (.wav)
```

Examples:

```bash
# Use camera ID 1 (external webcam)
python study_buddy.py --camera 1

# Increase eye closure threshold to 3 seconds
python study_buddy.py --eye-threshold 3.0

# Change log file location
python study_buddy.py --log my_study_session.txt

# Use custom alert sound
python study_buddy.py --sound my_alert.wav
```

## How It Works

Study Buddy uses MediaPipe's Face Mesh to detect facial landmarks in real-time. From these landmarks, it calculates:

1. **Eye Aspect Ratio (EAR)**: The ratio between the height and width of the eyes. A low EAR indicates closed eyes.
2. **Mouth Aspect Ratio (MAR)**: The ratio between the height and width of the mouth. A high MAR indicates an open mouth (potential yawn).

When the application detects that your eyes have been closed for too long or that you're yawning, it triggers visual and audio alerts to help you regain focus.

## Controls

- Press `q` to quit the application

## Log File Format

The application logs all alert events to a CSV-like text file with the following format:

```
Timestamp, Event Type, Duration (s)
2023-06-12 14:32:45, eye_closure, 2.34s
2023-06-12 14:40:12, yawn, 3.56s
```

## Troubleshooting

- **Camera not working**: Make sure your webcam is properly connected and not being used by another application.
- **Slow performance**: Try reducing your webcam resolution or closing other CPU-intensive applications.
- **False positives/negatives**: Adjust the threshold values using the command-line options.

## Calibration Tips

For optimal performance, consider these calibration tips:

1. **Lighting**: Ensure good, even lighting on your face
2. **Camera position**: Position your webcam at eye level
3. **Threshold adjustment**: If you're getting too many false alerts, try increasing the threshold values

## License

MIT License