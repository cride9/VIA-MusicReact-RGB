# Music-Reactive Keyboard LED Controller - Requirements

## Python Dependencies
- numpy
- scipy
- pyaudio
- hidapi

## Installation Instructions
1. Install Python 3.8 or newer from https://www.python.org/downloads/
2. Install the required Python packages:
   ```
   pip install numpy scipy pyaudio hidapi
   ```
3. Enable "Stereo Mix" in Windows Sound settings:
   - Right-click the speaker icon in the system tray
   - Select "Sound settings"
   - Click "Sound Control Panel"
   - Go to the "Recording" tab
   - Right-click in the empty area and select "Show Disabled Devices"
   - Right-click on "Stereo Mix" and select "Enable"
   - If you don't see "Stereo Mix", you may need to install a virtual audio cable software

## Usage
Run the program with:
```
python music_reactive_keyboard.py
```

### Command Line Options
- `--simulate`: Use simulated audio instead of capturing system audio
- `--sensitivity FLOAT`: Bass detection sensitivity (default: 5.0)
- `--min-brightness FLOAT`: Minimum brightness level (default: 0.1)
- `--max-brightness FLOAT`: Maximum brightness level (default: 1.0)
- `--smoothing FLOAT`: Smoothing factor for brightness changes (default: 0.3)

Example:
```
python music_reactive_keyboard.py --sensitivity 3.0 --min-brightness 0.2 --max-brightness 0.9
```

## Troubleshooting
- If the keyboard is not detected, make sure it's connected and in QMK mode
- If audio capture fails, check that "Stereo Mix" is enabled in your sound settings
- If the LED brightness doesn't change with music, try adjusting the sensitivity parameter
- If the program crashes, check the error message and ensure all dependencies are installed

## How It Works
1. The program captures your system's audio output using Windows WASAPI loopback
2. It analyzes the audio in real-time to detect bass frequencies (20-250 Hz)
3. The bass level is used to control the LED brightness of your Keychron V2 keyboard
4. Smoothing is applied to prevent flickering and create a more pleasing visual effect

## Notes
- The program requires administrator privileges to access the keyboard's HID interface
- Some antivirus software may block HID access; you may need to add an exception
- The program has been tested with Keychron V2 keyboards running QMK firmware
