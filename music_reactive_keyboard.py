#!/usr/bin/env python3
"""
Enhanced Music-Reactive Keyboard LED Controller with GUI
This program provides a graphical interface to control LED colors based on music.
Features:
- System tray icon for minimizing to taskbar
- RGB color control based on audio levels
- Support for any QMK RGB keyboard
- Enhanced configuration saving
"""

import sys
import os
import numpy as np
import pyaudio
import time
from scipy import signal
import hid
import threading
import queue
import json
import colorsys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QPushButton, QCheckBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QTabWidget, QProgressBar, QMessageBox,
    QFileDialog, QTextEdit, QColorDialog, QSystemTrayIcon, QMenu,
    QStyle, QGridLayout, QFrame, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QAction, QPixmap, QPainter, QLinearGradient

# Constants for audio processing
SAMPLE_RATE = 44100  # Hz
BLOCK_SIZE = 1024    # Number of samples per block
BASS_MIN_FREQ = 20   # Hz - lower bound of bass frequencies
BASS_MAX_FREQ = 250  # Hz - upper bound of bass frequencies

# Constants for LED control
MIN_BRIGHTNESS = 0.1  # Minimum brightness level (0.0-1.0)
MAX_BRIGHTNESS = 1.0  # Maximum brightness level (0.0-1.0)
SMOOTHING_FACTOR = 0.3  # Smoothing factor for brightness changes (0.0-1.0)

# Default colors
DEFAULT_QUIET_COLOR = QColor(255, 255, 255)  # White
DEFAULT_LOUD_COLOR = QColor(255, 0, 0)       # Red

def rgb_to_hsv_packet(r, g, b):
    """
    Convert r,g,b ∈ [0…255] to a HID packet [hue_lo, hue_hi, sat, val].
    QMK expects:
      - hue as a uint16_t (0…0xFFFF) little‑endian
      - saturation as uint8_t (0…0xFF)
      - value   as uint8_t (0…0xFF)
    """
    # 1) normalize to [0…1]
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0

    # 2) use Python’s colorsys to get h, s, v in [0…1]
    h_f, s_f, v_f = colorsys.rgb_to_hsv(rf, gf, bf)

    # 3) scale h to full 16‑bit range, s,v to 8‑bit
    hue_16 = int(h_f * 0xFFFF)            # 0…65535
    sat_8  = int(s_f * 0xFF)              # 0…255
    val_8  = int(v_f * 0xFF)              # 0…255

    # 4) split hue into two bytes, little‑endian
    hue_lo = hue_16 & 0xFF
    hue_hi = (hue_16 >> 8) & 0xFF

    return hue_lo, hue_hi, sat_8, val_8

class AudioCaptureThread(QThread):
    """Thread for capturing and processing audio"""
    
    # Signal to update the UI with new audio data
    audio_data_ready = pyqtSignal(float, float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, device_index=None, use_loopback=True):
        super().__init__()
        self.device_index = device_index
        self.use_loopback = use_loopback
        self.running = False
        self.p = None
        self.stream = None
        self.sensitivity = 5.0
        self.bass_min_freq = BASS_MIN_FREQ
        self.bass_max_freq = BASS_MAX_FREQ
        self.simulate = False
        self.simulation_index = 0
        self.simulation_pattern = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
            0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
            0.1, 0.5, 1.0, 0.5, 0.1, 0.5, 1.0, 0.5, 0.1
        ]
    
    def find_wasapi_loopback_device(self):
        """Find the WASAPI loopback device for capturing system audio"""
        p = pyaudio.PyAudio()
        wasapi_info = None
        
        # Find the WASAPI loopback device
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                
                # Check if this is a WASAPI device (hostApi = 1 on Windows)
                if device_info['hostApi'] == 1:
                    # Look for Stereo Mix or similar loopback devices
                    if ('Stereo Mix' in device_info['name'] or 
                        'What U Hear' in device_info['name'] or
                        'Loopback' in device_info['name']):
                        
                        wasapi_info = device_info
                        break
            except Exception:
                pass
        
        # If no Stereo Mix, try to find any output device that supports loopback
        if wasapi_info is None:
            try:
                default_output_device = p.get_default_output_device_info()
                
                # In some WASAPI implementations, we can use the output device index + 1 for loopback
                loopback_index = default_output_device['index'] + 1
                if loopback_index < p.get_device_count():
                    loopback_device = p.get_device_info_by_index(loopback_index)
                    wasapi_info = loopback_device
            except Exception:
                pass
        
        p.terminate()
        return wasapi_info
    
    def get_audio_devices(self):
        """Get a list of available audio devices"""
        devices = []
        p = pyaudio.PyAudio()
        
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                
                # Get host API name
                host_api_info = p.get_host_api_info_by_index(device_info['hostApi'])
                host_api_name = host_api_info['name']
                
                # Only include devices with input channels
                if device_info['maxInputChannels'] > 0:
                    device_name = f"{device_info['name']} ({host_api_name})"
                    devices.append((i, device_name))
            except Exception:
                pass
        
        p.terminate()
        return devices
    
    def start_capture(self):
        """Start capturing audio"""
        if self.simulate:
            self.running = True
            return True
            
        self.p = pyaudio.PyAudio()
        
        # If device index is specified, use it
        if self.device_index is not None:
            try:
                device_info = self.p.get_device_info_by_index(self.device_index)
            except Exception as e:
                self.error_occurred.emit(f"Error getting info for device {self.device_index}: {e}")
                self.device_index = None
        
        # If no device index specified and loopback is enabled, try to find WASAPI loopback device
        if self.device_index is None and self.use_loopback:
            wasapi_info = self.find_wasapi_loopback_device()
            
            if wasapi_info is not None:
                self.device_index = wasapi_info['index']
            else:
                self.error_occurred.emit("Could not find a suitable loopback audio capture device.")
        
        # If still no device index, use default input device
        if self.device_index is None:
            try:
                default_input_device = self.p.get_default_input_device_info()
                self.device_index = default_input_device['index']
            except Exception as e:
                self.error_occurred.emit(f"Error getting default input device: {e}")
                return False
        
        # Get device info
        try:
            device_info = self.p.get_device_info_by_index(self.device_index)
            channels = 2 if device_info['maxInputChannels'] >= 2 else 1
            
            # Open stream in callback mode
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=BLOCK_SIZE
            )
            
            self.running = True
            return True
        except Exception as e:
            self.error_occurred.emit(f"Error starting audio capture: {e}")
            return False
    
    def stop_capture(self):
        """Stop capturing audio"""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
    
    def get_bass_level(self, audio_data):
        """
        Extract bass level from audio data using FFT
        
        Args:
            audio_data: numpy array of audio samples
        
        Returns:
            bass_level: normalized bass level (0.0 to 1.0)
        """
        # If stereo, convert to mono by averaging channels
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Apply window function to reduce spectral leakage
        windowed_data = audio_data * signal.windows.hann(len(audio_data))
        
        # Compute FFT
        fft_data = np.abs(np.fft.rfft(windowed_data))
        
        # Get frequency bins
        freq_bins = np.fft.rfftfreq(len(windowed_data), 1/SAMPLE_RATE)
        
        # Find indices corresponding to bass frequency range
        bass_indices = np.where((freq_bins >= self.bass_min_freq) & (freq_bins <= self.bass_max_freq))[0]
        
        # Calculate average magnitude in bass range
        if len(bass_indices) > 0:
            bass_magnitude = np.mean(fft_data[bass_indices])
            # Normalize with adjustable factor
            bass_level = min(1.0, bass_magnitude / self.sensitivity)
        else:
            bass_level = 0.0
        
        return bass_level
    
    def get_audio_level(self, audio_data):
        """
        Calculate overall audio level (volume)
        
        Args:
            audio_data: numpy array of audio samples
        
        Returns:
            audio_level: normalized audio level (0.0 to 1.0)
        """
        # If stereo, convert to mono by averaging channels
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Calculate RMS (root mean square) of audio data
        rms = np.sqrt(np.mean(np.square(audio_data)))
        
        # Normalize to 0.0-1.0 range with sensitivity adjustment
        audio_level = min(1.0, rms * 5.0 / self.sensitivity)
        
        return audio_level
    
    def simulate_audio(self):
        """Simulate audio with varying bass levels for testing"""
        bass_level = self.simulation_pattern[self.simulation_index]
        self.simulation_index = (self.simulation_index + 1) % len(self.simulation_pattern)
        return bass_level
    
    def run(self):
        """Main thread function"""
        if not self.start_capture():
            return
        
        try:
            while self.running:
                if self.simulate:
                    # Use simulated audio
                    bass_level = self.simulate_audio()
                    audio_level = bass_level  # For simplicity, use same value for audio level
                    self.audio_data_ready.emit(bass_level, audio_level)
                    time.sleep(0.1)  # Simulate processing time
                else:
                    # Read audio data
                    try:
                        audio_data = np.frombuffer(self.stream.read(BLOCK_SIZE), dtype=np.float32)
                        
                        # Process audio data
                        bass_level = self.get_bass_level(audio_data)
                        audio_level = self.get_audio_level(audio_data)
                        
                        # Emit signal with processed data
                        self.audio_data_ready.emit(bass_level, audio_level)
                    except Exception as e:
                        self.error_occurred.emit(f"Error processing audio: {e}")
                        time.sleep(0.1)  # Avoid tight loop on error
        finally:
            self.stop_capture()

class KeyboardControlThread(QThread):
    """Thread for controlling keyboard LEDs"""
    
    # Signal to update the UI with status
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    connection_status = pyqtSignal(bool)
    keyboard_list_updated = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.device = None
        self.raw_hid_path = None
        self.connected = False
        self.running = False
        self.current_brightness = MAX_BRIGHTNESS
        self.target_brightness = MAX_BRIGHTNESS
        self.smoothing_factor = SMOOTHING_FACTOR
        self.min_brightness = MIN_BRIGHTNESS
        self.max_brightness = MAX_BRIGHTNESS
        self.quiet_color = DEFAULT_QUIET_COLOR
        self.loud_color = DEFAULT_LOUD_COLOR
        self.use_rgb = False
        self.current_color = self.quiet_color
        self.vendor_id = None
        self.product_id = None
        self.interface_number = 1  # Default to 1 for QMK Raw HID
        self.scan_interval = 5  # Seconds between keyboard scans
        self.last_scan_time = 0
        self.available_keyboards = []
    
    def find_qmk_keyboards(self):
        """Find all available QMK keyboards"""
        keyboards = []
        
        try:
            # Get all HID devices
            all_devices = hid.enumerate()
            
            # Filter for potential QMK keyboards
            for device in all_devices:
                # Check if this might be a QMK keyboard
                # QMK keyboards typically have usage page 0xFF60 or interface 1
                if (device.get("usage_page") == 0xFF60 or 
                    device.get("interface_number") == 1):
                    
                    vendor_id = device.get("vendor_id")
                    product_id = device.get("product_id")
                    interface_number = device.get("interface_number")
                    
                    # Try to get product name
                    product = device.get("product_string", "Unknown Keyboard")
                    
                    # Add to list if not already there
                    keyboard_id = (vendor_id, product_id, interface_number)
                    if keyboard_id not in [k[0] for k in keyboards]:
                        keyboards.append((keyboard_id, product))
            
            # Update available keyboards list
            self.available_keyboards = keyboards
            
            # Emit signal with keyboard list
            self.keyboard_list_updated.emit(keyboards)
            
            return keyboards
        except Exception as e:
            self.error_occurred.emit(f"Error finding keyboards: {e}")
            return []
    
    def connect_to_keyboard(self, vendor_id=None, product_id=None, interface_number=1):
        """Connect to a specific keyboard"""
        self.status_update.emit(f"Connecting to keyboard (VID: {vendor_id:04x}, PID: {product_id:04x}, Interface: {interface_number})...")
        
        try:
            # Store keyboard identifiers
            self.vendor_id = vendor_id
            self.product_id = product_id
            self.interface_number = interface_number
            
            # Find the Raw HID interface
            devices = hid.enumerate(vendor_id, product_id)
            
            if not devices:
                self.error_occurred.emit(f"No keyboard found with VID: {vendor_id:04x}, PID: {product_id:04x}")
                return False
            
            self.status_update.emit(f"Found {len(devices)} HID interfaces")
            
            # Find the specified interface
            for i, dev in enumerate(devices):
                dev_interface = dev.get("interface_number")
                
                if dev_interface == interface_number:
                    self.raw_hid_path = dev["path"]
                    self.status_update.emit(f"Found interface {interface_number}")
                    break
            
            if not self.raw_hid_path:
                self.error_occurred.emit(f"Interface {interface_number} not found")
                return False
            
            # Open the Raw HID device
            self.device = hid.device()
            self.device.open_path(self.raw_hid_path)
            
            self.status_update.emit(f"Connected to keyboard")
            self.connected = True
            self.connection_status.emit(True)
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error connecting to keyboard: {e}")
            self.connection_status.emit(False)
            return False
    
    def set_brightness(self, brightness):
        """
        Set the LED brightness
        
        Args:
            brightness: Brightness level (0.0 to 1.0)
        
        Returns:
            Success status
        """
        if not self.connected:
            return False
        
        # Store target brightness
        self.target_brightness = brightness
        
        # Apply smoothing to brightness changes
        smoothed_brightness = (self.smoothing_factor * brightness + 
                              (1 - self.smoothing_factor) * self.current_brightness)
        
        # Ensure brightness is within range
        smoothed_brightness = max(self.min_brightness, min(self.max_brightness, smoothed_brightness))
        
        # Only update if brightness has changed significantly
        if abs(smoothed_brightness - self.current_brightness) < 0.01 and not self.use_rgb:
            return True
        
        # Convert brightness to QMK value (0-255)
        qmk_brightness = int(smoothed_brightness * 255)
        
        try:
            if self.use_rgb:
                # Interpolate between quiet and loud colors based on brightness
                r1, g1, b1 = self.quiet_color.red(), self.quiet_color.green(), self.quiet_color.blue()
                r2, g2, b2 = self.loud_color.red(), self.loud_color.green(), self.loud_color.blue()
                
                r = int(r1 + (r2 - r1) * smoothed_brightness)
                g = int(g1 + (g2 - g1) * smoothed_brightness)
                b = int(b1 + (b2 - b1) * smoothed_brightness)
                
                # Update current color
                self.current_color = QColor(r, g, b)
                rf, gf, bf = r/255.0, g/255.0, b/255.0
                h_f, s_f, v_f = colorsys.rgb_to_hsv(rf, gf, bf)
                h8 = int(h_f * 255)   # hue
                s8 = int(s_f * 255)   # saturation
                v8 = int(v_f * 255)   # brightness/value

                command = [
                    0x00,       # report ID
                    0x07,       # id_custom_set_value
                    0x03,       # id_qmk_rgb_matrix_channel
                    0x04,       # id_qmk_rgb_matrix_color
                    h8,         # hue  (0–255)
                    s8,         # sat  (0–255)
                ]
                self.device.write(command)

                command = [
                    0,      # report ID - always 0
                    0x07,   # command id - 0x07 (custom set value)
                    0x03,   # channel id - 0x03 (rgbmatrix)
                    0x01,   # value id - 0x01 (brightness)
                    255  # value data - brightness level (0-255)
                ]
            else:
                # QMK/VIA Raw HID command format for brightness
                command = [
                    0,      # report ID - always 0
                    0x07,   # command id - 0x07 (custom set value)
                    0x03,   # channel id - 0x03 (rgbmatrix)
                    0x01,   # value id - 0x01 (brightness)
                    qmk_brightness  # value data - brightness level (0-255)
                ]
            
            self.device.write(command)
            self.current_brightness = smoothed_brightness
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error setting {'color' if self.use_rgb else 'brightness'}: {e}")
            return False
    
    def run(self):
        """Main thread function"""
        # Initial keyboard scan
        if not self.find_qmk_keyboards():
            self.error_occurred.emit("No QMK keyboards found. Please connect a compatible keyboard.")
        
        self.running = True
        
        try:
            while self.running:
                # Periodically scan for keyboards
                current_time = time.time()
                if current_time - self.last_scan_time > self.scan_interval:
                    self.find_qmk_keyboards()
                    self.last_scan_time = current_time
                
                # If not connected but we have vendor_id and product_id, try to connect
                if not self.connected and self.vendor_id and self.product_id:
                    self.connect_to_keyboard(self.vendor_id, self.product_id, self.interface_number)
                
                # If connected, set brightness to target value
                if self.connected:
                    self.set_brightness(self.target_brightness)
                
                # Sleep to avoid tight loop
                time.sleep(0.01)
        finally:
            # Set max brightness before exiting
            if self.connected:
                self.use_rgb = False  # Revert to brightness control
                self.set_brightness(1.0)
                
                if self.device:
                    self.device.close()
                    self.connected = False
                    self.connection_status.emit(False)
                    self.status_update.emit("Disconnected from keyboard")

class ColorPreviewWidget(QWidget):
    """Widget to preview color transitions"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.quiet_color = DEFAULT_QUIET_COLOR
        self.loud_color = DEFAULT_LOUD_COLOR
        self.current_level = 0.0
        self.setMinimumHeight(30)
        self.setMinimumWidth(100)
    
    def set_colors(self, quiet_color, loud_color):
        """Set the quiet and loud colors"""
        self.quiet_color = quiet_color
        self.loud_color = loud_color
        self.update()
    
    def set_level(self, level):
        """Set the current level (0.0 to 1.0)"""
        self.current_level = max(0.0, min(1.0, level))
        self.update()
    
    def paintEvent(self, event):
        """Paint the widget"""
        painter = QPainter(self)
        
        # Draw gradient from quiet to loud color
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, self.quiet_color)
        gradient.setColorAt(1, self.loud_color)
        
        # Fill background with gradient
        painter.fillRect(0, 0, self.width(), self.height(), gradient)
        
        # Draw current level indicator
        x = int(self.current_level * self.width())
        painter.setPen(Qt.GlobalColor.black)
        painter.drawLine(x, 0, x, self.height())
        
        # Draw border
        painter.setPen(Qt.GlobalColor.black)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Set up the UI
        self.setWindowTitle("Music-Reactive Keyboard LED Controller")
        self.setMinimumSize(800, 600)
        
        # Initialize threads
        self.audio_thread = AudioCaptureThread()
        self.keyboard_thread = KeyboardControlThread()
        
        # Connect signals
        self.audio_thread.audio_data_ready.connect(self.update_audio_data)
        self.audio_thread.error_occurred.connect(self.show_error)
        self.keyboard_thread.status_update.connect(self.update_status)
        self.keyboard_thread.error_occurred.connect(self.show_error)
        self.keyboard_thread.connection_status.connect(self.update_connection_status)
        self.keyboard_thread.keyboard_list_updated.connect(self.update_keyboard_list)
        
        # Set up the central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create main control tab
        self.control_tab = QWidget()
        self.tabs.addTab(self.control_tab, "Controls")
        
        # Create RGB tab
        self.rgb_tab = QWidget()
        self.tabs.addTab(self.rgb_tab, "RGB Control")
        
        # Create settings tab
        self.settings_tab = QWidget()
        self.tabs.addTab(self.settings_tab, "Settings")
        
        # Create about tab
        self.about_tab = QWidget()
        self.tabs.addTab(self.about_tab, "About")
        
        # Set up control tab
        self.setup_control_tab()
        
        # Set up RGB tab
        self.setup_rgb_tab()
        
        # Set up settings tab
        self.setup_settings_tab()
        
        # Set up about tab
        self.setup_about_tab()
        
        # Status bar at the bottom
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Set up system tray icon
        self.setup_system_tray()
        
        # Start keyboard thread
        self.keyboard_thread.start()
        
        # Update timer for UI
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(50)  # 50ms = 20fps
        
        # Load default settings
        self.load_default_settings()
    
    def setup_control_tab(self):
        """Set up the main control tab"""
        layout = QVBoxLayout(self.control_tab)
        
        # Connection status
        connection_group = QGroupBox("Connection Status")
        connection_layout = QHBoxLayout(connection_group)
        
        self.keyboard_status_label = QLabel("Keyboard: Disconnected")
        self.keyboard_status_label.setStyleSheet("color: red;")
        connection_layout.addWidget(self.keyboard_status_label)
        
        self.audio_status_label = QLabel("Audio: Stopped")
        self.audio_status_label.setStyleSheet("color: red;")
        connection_layout.addWidget(self.audio_status_label)
        
        layout.addWidget(connection_group)
        
        # Visualization
        viz_group = QGroupBox("Audio Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Bass level
        bass_layout = QHBoxLayout()
        bass_layout.addWidget(QLabel("Bass Level:"))
        self.bass_level_bar = QProgressBar()
        self.bass_level_bar.setRange(0, 100)
        self.bass_level_bar.setValue(0)
        bass_layout.addWidget(self.bass_level_bar)
        viz_layout.addLayout(bass_layout)
        
        # Audio level
        audio_layout = QHBoxLayout()
        audio_layout.addWidget(QLabel("Audio Level:"))
        self.audio_level_bar = QProgressBar()
        self.audio_level_bar.setRange(0, 100)
        self.audio_level_bar.setValue(0)
        audio_layout.addWidget(self.audio_level_bar)
        viz_layout.addLayout(audio_layout)
        
        # Brightness level
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("LED Brightness:"))
        self.brightness_level_bar = QProgressBar()
        self.brightness_level_bar.setRange(0, 100)
        self.brightness_level_bar.setValue(0)
        brightness_layout.addWidget(self.brightness_level_bar)
        viz_layout.addLayout(brightness_layout)
        
        layout.addWidget(viz_group)
        
        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Sensitivity slider
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(QLabel("Bass Sensitivity:"))
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(1, 200)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sensitivity_slider.setTickInterval(10)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        self.sensitivity_value_label = QLabel("5.0")
        sensitivity_layout.addWidget(self.sensitivity_value_label)
        controls_layout.addLayout(sensitivity_layout)
        
        # Min brightness slider
        min_brightness_layout = QHBoxLayout()
        min_brightness_layout.addWidget(QLabel("Min Brightness:"))
        self.min_brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_brightness_slider.setRange(0, 100)
        self.min_brightness_slider.setValue(int(MIN_BRIGHTNESS * 100))
        self.min_brightness_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.min_brightness_slider.setTickInterval(10)
        self.min_brightness_slider.valueChanged.connect(self.update_min_brightness)
        min_brightness_layout.addWidget(self.min_brightness_slider)
        self.min_brightness_value_label = QLabel(f"{MIN_BRIGHTNESS:.2f}")
        min_brightness_layout.addWidget(self.min_brightness_value_label)
        controls_layout.addLayout(min_brightness_layout)
        
        # Max brightness slider
        max_brightness_layout = QHBoxLayout()
        max_brightness_layout.addWidget(QLabel("Max Brightness:"))
        self.max_brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_brightness_slider.setRange(0, 100)
        self.max_brightness_slider.setValue(int(MAX_BRIGHTNESS * 100))
        self.max_brightness_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.max_brightness_slider.setTickInterval(10)
        self.max_brightness_slider.valueChanged.connect(self.update_max_brightness)
        max_brightness_layout.addWidget(self.max_brightness_slider)
        self.max_brightness_value_label = QLabel(f"{MAX_BRIGHTNESS:.2f}")
        max_brightness_layout.addWidget(self.max_brightness_value_label)
        controls_layout.addLayout(max_brightness_layout)
        
        # Smoothing slider
        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(QLabel("Smoothing:"))
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(0, 100)
        self.smoothing_slider.setValue(int(SMOOTHING_FACTOR * 100))
        self.smoothing_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.smoothing_slider.setTickInterval(10)
        self.smoothing_slider.valueChanged.connect(self.update_smoothing)
        smoothing_layout.addWidget(self.smoothing_slider)
        self.smoothing_value_label = QLabel(f"{SMOOTHING_FACTOR:.2f}")
        smoothing_layout.addWidget(self.smoothing_value_label)
        controls_layout.addLayout(smoothing_layout)
        
        # Manual brightness control
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(QLabel("Manual Brightness:"))
        self.manual_brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.manual_brightness_slider.setRange(0, 100)
        self.manual_brightness_slider.setValue(100)
        self.manual_brightness_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.manual_brightness_slider.setTickInterval(10)
        self.manual_brightness_slider.valueChanged.connect(self.update_manual_brightness)
        manual_layout.addWidget(self.manual_brightness_slider)
        self.manual_brightness_value_label = QLabel("1.00")
        manual_layout.addWidget(self.manual_brightness_value_label)
        controls_layout.addLayout(manual_layout)
        
        # Manual/Auto mode
        mode_layout = QHBoxLayout()
        self.auto_mode_checkbox = QCheckBox("Auto Mode (React to Music)")
        self.auto_mode_checkbox.setChecked(True)
        self.auto_mode_checkbox.stateChanged.connect(self.toggle_auto_mode)
        mode_layout.addWidget(self.auto_mode_checkbox)
        
        # RGB mode
        self.rgb_mode_checkbox = QCheckBox("RGB Mode (Color Changes)")
        self.rgb_mode_checkbox.setChecked(False)
        self.rgb_mode_checkbox.stateChanged.connect(self.toggle_rgb_mode)
        mode_layout.addWidget(self.rgb_mode_checkbox)
        
        controls_layout.addLayout(mode_layout)
        
        layout.addWidget(controls_group)
        
        # Start/Stop buttons
        buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Audio Capture")
        self.start_button.clicked.connect(self.start_audio)
        buttons_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Audio Capture")
        self.stop_button.clicked.connect(self.stop_audio)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        self.simulate_checkbox = QCheckBox("Simulate Audio")
        self.simulate_checkbox.stateChanged.connect(self.toggle_simulation)
        buttons_layout.addWidget(self.simulate_checkbox)
        
        layout.addLayout(buttons_layout)
    
    def setup_rgb_tab(self):
        """Set up the RGB control tab"""
        layout = QVBoxLayout(self.rgb_tab)
        
        # RGB mode checkbox
        rgb_mode_layout = QHBoxLayout()
        rgb_mode_checkbox = QCheckBox("Enable RGB Color Mode")
        rgb_mode_checkbox.setChecked(self.keyboard_thread.use_rgb)
        rgb_mode_checkbox.stateChanged.connect(self.toggle_rgb_mode)
        rgb_mode_layout.addWidget(rgb_mode_checkbox)
        layout.addLayout(rgb_mode_layout)
        
        # Color selection
        color_group = QGroupBox("Color Selection")
        color_layout = QGridLayout(color_group)
        
        # Quiet color
        color_layout.addWidget(QLabel("Quiet Color:"), 0, 0)
        self.quiet_color_button = QPushButton()
        self.quiet_color_button.setStyleSheet(f"background-color: {self.keyboard_thread.quiet_color.name()}")
        self.quiet_color_button.clicked.connect(self.select_quiet_color)
        color_layout.addWidget(self.quiet_color_button, 0, 1)
        
        # Loud color
        color_layout.addWidget(QLabel("Loud Color:"), 1, 0)
        self.loud_color_button = QPushButton()
        self.loud_color_button.setStyleSheet(f"background-color: {self.keyboard_thread.loud_color.name()}")
        self.loud_color_button.clicked.connect(self.select_loud_color)
        color_layout.addWidget(self.loud_color_button, 1, 1)
        
        layout.addWidget(color_group)
        
        # Color preview
        preview_group = QGroupBox("Color Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.color_preview = ColorPreviewWidget()
        self.color_preview.set_colors(self.keyboard_thread.quiet_color, self.keyboard_thread.loud_color)
        preview_layout.addWidget(self.color_preview)
        
        # Preview slider
        preview_slider_layout = QHBoxLayout()
        preview_slider_layout.addWidget(QLabel("Preview Level:"))
        self.preview_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_slider.setRange(0, 100)
        self.preview_slider.setValue(0)
        self.preview_slider.valueChanged.connect(self.update_preview)
        preview_slider_layout.addWidget(self.preview_slider)
        preview_layout.addLayout(preview_slider_layout)
        
        layout.addWidget(preview_group)
        
        # Preset colors
        presets_group = QGroupBox("Color Presets")
        presets_layout = QGridLayout(presets_group)
        
        # Define presets
        presets = [
            ("White → Red", QColor(255, 255, 255), QColor(255, 0, 0)),
            ("Blue → Red", QColor(0, 0, 255), QColor(255, 0, 0)),
            ("Green → Red", QColor(0, 255, 0), QColor(255, 0, 0)),
            ("Cyan → Purple", QColor(0, 255, 255), QColor(128, 0, 128)),
            ("Blue → Yellow", QColor(0, 0, 255), QColor(255, 255, 0)),
            ("Green → Blue", QColor(0, 255, 0), QColor(0, 0, 255))
        ]
        
        # Create preset buttons
        row, col = 0, 0
        for name, quiet, loud in presets:
            button = QPushButton(name)
            button.clicked.connect(lambda checked, q=quiet, l=loud: self.apply_color_preset(q, l))
            
            # Create gradient background
            gradient_style = f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5,
                        stop:0 {quiet.name()}, stop:1 {loud.name()});
                    color: black;
                    font-weight: bold;
                }}
            """
            button.setStyleSheet(gradient_style)
            
            presets_layout.addWidget(button, row, col)
            col += 1
            if col > 2:  # 3 columns
                col = 0
                row += 1
        
        layout.addWidget(presets_group)
        
        # Add spacer
        layout.addStretch()
    
    def setup_settings_tab(self):
        """Set up the settings tab"""
        layout = QVBoxLayout(self.settings_tab)
        
        # Keyboard selection
        keyboard_group = QGroupBox("Keyboard Selection")
        keyboard_layout = QVBoxLayout(keyboard_group)
        
        # Keyboard list
        self.keyboard_combo = QComboBox()
        self.keyboard_combo.addItem("Select a keyboard...", None)
        keyboard_layout.addWidget(self.keyboard_combo)
        
        # Refresh and connect buttons
        keyboard_buttons_layout = QHBoxLayout()
        
        self.refresh_keyboards_button = QPushButton("Refresh Keyboards")
        self.refresh_keyboards_button.clicked.connect(self.refresh_keyboards)
        keyboard_buttons_layout.addWidget(self.refresh_keyboards_button)
        
        self.connect_keyboard_button = QPushButton("Connect")
        self.connect_keyboard_button.clicked.connect(self.connect_selected_keyboard)
        keyboard_buttons_layout.addWidget(self.connect_keyboard_button)
        
        keyboard_layout.addLayout(keyboard_buttons_layout)
        
        # Manual keyboard ID entry
        manual_id_group = QGroupBox("Manual Keyboard ID")
        manual_id_layout = QGridLayout(manual_id_group)
        
        manual_id_layout.addWidget(QLabel("Vendor ID (hex):"), 0, 0)
        self.vendor_id_input = QLineEdit("3434")
        self.vendor_id_input.setInputMask("HHHH")
        manual_id_layout.addWidget(self.vendor_id_input, 0, 1)
        
        manual_id_layout.addWidget(QLabel("Product ID (hex):"), 1, 0)
        self.product_id_input = QLineEdit("0320")
        self.product_id_input.setInputMask("HHHH")
        manual_id_layout.addWidget(self.product_id_input, 1, 1)
        
        manual_id_layout.addWidget(QLabel("Interface Number:"), 2, 0)
        self.interface_number_input = QSpinBox()
        self.interface_number_input.setRange(0, 10)
        self.interface_number_input.setValue(1)
        manual_id_layout.addWidget(self.interface_number_input, 2, 1)
        
        self.manual_connect_button = QPushButton("Connect with Manual ID")
        self.manual_connect_button.clicked.connect(self.connect_manual_keyboard)
        manual_id_layout.addWidget(self.manual_connect_button, 3, 0, 1, 2)
        
        keyboard_layout.addWidget(manual_id_group)
        layout.addWidget(keyboard_group)
        
        # Audio settings
        audio_group = QGroupBox("Audio Settings")
        audio_layout = QVBoxLayout(audio_group)
        
        # Audio device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Audio Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto-detect", None)
        
        # Add available devices
        for device_id, device_name in self.audio_thread.get_audio_devices():
            self.device_combo.addItem(device_name, device_id)
        
        device_layout.addWidget(self.device_combo)
        self.refresh_devices_button = QPushButton("Refresh")
        self.refresh_devices_button.clicked.connect(self.refresh_devices)
        device_layout.addWidget(self.refresh_devices_button)
        audio_layout.addLayout(device_layout)
        
        # Use loopback checkbox
        loopback_layout = QHBoxLayout()
        self.loopback_checkbox = QCheckBox("Use WASAPI Loopback (for system audio)")
        self.loopback_checkbox.setChecked(True)
        loopback_layout.addWidget(self.loopback_checkbox)
        audio_layout.addLayout(loopback_layout)
        
        # Bass frequency range
        freq_group = QGroupBox("Bass Frequency Range")
        freq_layout = QVBoxLayout(freq_group)
        
        min_freq_layout = QHBoxLayout()
        min_freq_layout.addWidget(QLabel("Min Frequency (Hz):"))
        self.min_freq_spin = QSpinBox()
        self.min_freq_spin.setRange(10, 100)
        self.min_freq_spin.setValue(BASS_MIN_FREQ)
        self.min_freq_spin.valueChanged.connect(self.update_bass_range)
        min_freq_layout.addWidget(self.min_freq_spin)
        freq_layout.addLayout(min_freq_layout)
        
        max_freq_layout = QHBoxLayout()
        max_freq_layout.addWidget(QLabel("Max Frequency (Hz):"))
        self.max_freq_spin = QSpinBox()
        self.max_freq_spin.setRange(100, 500)
        self.max_freq_spin.setValue(BASS_MAX_FREQ)
        self.max_freq_spin.valueChanged.connect(self.update_bass_range)
        max_freq_layout.addWidget(self.max_freq_spin)
        freq_layout.addLayout(max_freq_layout)
        
        audio_layout.addWidget(freq_group)
        layout.addWidget(audio_group)
        
        # System tray settings
        tray_group = QGroupBox("System Tray Settings")
        tray_layout = QVBoxLayout(tray_group)
        
        self.minimize_to_tray_checkbox = QCheckBox("Minimize to System Tray on Close")
        self.minimize_to_tray_checkbox.setChecked(True)
        tray_layout.addWidget(self.minimize_to_tray_checkbox)
        
        self.start_minimized_checkbox = QCheckBox("Start Minimized to System Tray")
        self.start_minimized_checkbox.setChecked(False)
        tray_layout.addWidget(self.start_minimized_checkbox)
        
        layout.addWidget(tray_group)
        
        # Preset settings
        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout(preset_group)
        
        # Music genre selection
        genre_layout = QHBoxLayout()
        genre_layout.addWidget(QLabel("Music Genre:"))
        self.genre_combo = QComboBox()
        self.genre_combo.addItems(["Electronic", "Rock", "Classical", "Hip-hop"])
        genre_layout.addWidget(self.genre_combo)
        preset_layout.addLayout(genre_layout)
        
        # Effect preference
        effect_layout = QHBoxLayout()
        effect_layout.addWidget(QLabel("Effect Preference:"))
        self.effect_combo = QComboBox()
        self.effect_combo.addItems(["Responsive", "Stable", "Dynamic"])
        effect_layout.addWidget(self.effect_combo)
        preset_layout.addLayout(effect_layout)
        
        # Apply preset button
        self.apply_preset_button = QPushButton("Apply Preset")
        self.apply_preset_button.clicked.connect(self.apply_preset)
        preset_layout.addWidget(self.apply_preset_button)
        
        layout.addWidget(preset_group)
        
        # Save/Load settings
        save_load_group = QGroupBox("Save/Load Settings")
        save_load_layout = QHBoxLayout(save_load_group)
        
        self.save_settings_button = QPushButton("Save Settings")
        self.save_settings_button.clicked.connect(self.save_settings)
        save_load_layout.addWidget(self.save_settings_button)
        
        self.load_settings_button = QPushButton("Load Settings")
        self.load_settings_button.clicked.connect(self.load_settings)
        save_load_layout.addWidget(self.load_settings_button)
        
        layout.addWidget(save_load_group)
        
        # Add spacer
        layout.addStretch()
    
    def setup_about_tab(self):
        """Set up the about tab"""
        layout = QVBoxLayout(self.about_tab)
        
        # Title
        title_label = QLabel("Music-Reactive Keyboard LED Controller")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Description
        desc_text = QTextEdit()
        desc_text.setReadOnly(True)
        desc_text.setHtml("""
        <p>This application controls the LED colors of your QMK keyboard based on the music you're listening to.</p>
        
        <h3>Features:</h3>
        <ul>
            <li>Real-time audio analysis to detect bass levels</li>
            <li>Dynamic LED brightness and color control</li>
            <li>Support for any QMK RGB keyboard</li>
            <li>Customizable settings for different music genres</li>
            <li>Manual or automatic brightness/color control</li>
            <li>System tray support for background operation</li>
            <li>Simulation mode for testing</li>
        </ul>
        
        <h3>How to Use:</h3>
        <ol>
            <li>Connect your QMK keyboard</li>
            <li>Enable "Stereo Mix" in your Windows sound settings</li>
            <li>Select your keyboard from the Settings tab</li>
            <li>Choose between brightness or RGB color mode</li>
            <li>Click "Start Audio Capture" to begin</li>
            <li>Adjust settings as needed for your preferences</li>
        </ol>
        
        <p>Created with PyQt6 and Python 3.8+</p>
        """)
        layout.addWidget(desc_text)
    
    def setup_system_tray(self):
        """Set up system tray icon and menu"""
        # Create system tray icon
        self.tray_icon = QSystemTrayIcon(self)
        
        # Try to use app icon or fallback to a system icon
        try:
            app_icon = QIcon(QPixmap(32, 32))
            app_icon.addPixmap(self.style().standardPixmap(QStyle.StandardPixmap.SP_ComputerIcon))
            self.tray_icon.setIcon(app_icon)
            self.setWindowIcon(app_icon)
        except:
            # Fallback to system icon
            self.tray_icon.setIcon(self.style().standardPixmap(QStyle.StandardPixmap.SP_ComputerIcon))
        
        # Create tray menu
        tray_menu = QMenu()
        
        # Show/hide action
        self.show_hide_action = QAction("Hide", self)
        self.show_hide_action.triggered.connect(self.toggle_window)
        tray_menu.addAction(self.show_hide_action)
        
        # Start/stop audio action
        self.start_stop_action = QAction("Start Audio", self)
        self.start_stop_action.triggered.connect(self.toggle_audio)
        tray_menu.addAction(self.start_stop_action)
        
        tray_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.exit_application)
        tray_menu.addAction(exit_action)
        
        # Set the menu
        self.tray_icon.setContextMenu(tray_menu)
        
        # Connect signals
        self.tray_icon.activated.connect(self.tray_icon_activated)
        
        # Show the tray icon
        self.tray_icon.show()
    
    def tray_icon_activated(self, reason):
        """Handle tray icon activation"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.toggle_window()
    
    def toggle_window(self):
        """Toggle window visibility"""
        if self.isVisible():
            self.hide()
            self.show_hide_action.setText("Show")
        else:
            self.show()
            self.show_hide_action.setText("Hide")
    
    def toggle_audio(self):
        """Toggle audio capture from tray menu"""
        if self.audio_thread.running:
            self.stop_audio()
            self.start_stop_action.setText("Start Audio")
        else:
            self.start_audio()
            self.start_stop_action.setText("Stop Audio")
    
    def update_audio_data(self, bass_level, audio_level):
        """Update UI with new audio data"""
        # Update progress bars
        self.bass_level_bar.setValue(int(bass_level * 100))
        self.audio_level_bar.setValue(int(audio_level * 100))
        
        # Update color preview
        self.color_preview.set_level(bass_level)
        
        # Update keyboard brightness if in auto mode
        if self.auto_mode_checkbox.isChecked():
            self.keyboard_thread.target_brightness = bass_level
    
    def update_ui(self):
        """Update UI elements periodically"""
        # Update brightness level bar
        self.brightness_level_bar.setValue(int(self.keyboard_thread.current_brightness * 100))
    
    def update_status(self, message):
        """Update status bar with message"""
        self.status_bar.showMessage(message)
    
    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.status_bar.showMessage(f"Error: {message}")
    
    def update_connection_status(self, connected):
        """Update connection status label"""
        if connected:
            self.keyboard_status_label.setText("Keyboard: Connected")
            self.keyboard_status_label.setStyleSheet("color: green;")
        else:
            self.keyboard_status_label.setText("Keyboard: Disconnected")
            self.keyboard_status_label.setStyleSheet("color: red;")
    
    def update_keyboard_list(self, keyboards):
        """Update the keyboard selection dropdown"""
        # Save current selection
        current_data = self.keyboard_combo.currentData()
        
        # Clear and repopulate
        self.keyboard_combo.clear()
        self.keyboard_combo.addItem("Select a keyboard...", None)
        
        for keyboard_id, name in keyboards:
            vendor_id, product_id, interface_number = keyboard_id
            display_name = f"{name} (VID: {vendor_id:04x}, PID: {product_id:04x}, Interface: {interface_number})"
            self.keyboard_combo.addItem(display_name, keyboard_id)
        
        # Restore selection if possible
        if current_data:
            index = self.keyboard_combo.findData(current_data)
            if index >= 0:
                self.keyboard_combo.setCurrentIndex(index)
    
    def refresh_keyboards(self):
        """Refresh the list of available keyboards"""
        self.keyboard_thread.find_qmk_keyboards()
        self.status_bar.showMessage("Keyboard list refreshed")
    
    def connect_selected_keyboard(self):
        """Connect to the selected keyboard"""
        keyboard_id = self.keyboard_combo.currentData()
        
        if not keyboard_id:
            self.show_error("Please select a keyboard")
            return
        
        vendor_id, product_id, interface_number = keyboard_id
        
        # Update manual input fields
        self.vendor_id_input.setText(f"{vendor_id:04x}")
        self.product_id_input.setText(f"{product_id:04x}")
        self.interface_number_input.setValue(interface_number)
        
        # Connect to keyboard
        self.keyboard_thread.connect_to_keyboard(vendor_id, product_id, interface_number)
    
    def connect_manual_keyboard(self):
        """Connect to keyboard using manually entered IDs"""
        try:
            vendor_id = int(self.vendor_id_input.text(), 16)
            product_id = int(self.product_id_input.text(), 16)
            interface_number = self.interface_number_input.value()
            
            self.keyboard_thread.connect_to_keyboard(vendor_id, product_id, interface_number)
        except ValueError:
            self.show_error("Invalid vendor or product ID. Please enter valid hexadecimal values.")
    
    def update_sensitivity(self):
        """Update sensitivity value"""
        value = self.sensitivity_slider.value() / 10.0
        self.sensitivity_value_label.setText(f"{value:.1f}")
        self.audio_thread.sensitivity = value
    
    def update_min_brightness(self):
        """Update minimum brightness value"""
        value = self.min_brightness_slider.value() / 100.0
        self.min_brightness_value_label.setText(f"{value:.2f}")
        self.keyboard_thread.min_brightness = value
    
    def update_max_brightness(self):
        """Update maximum brightness value"""
        value = self.max_brightness_slider.value() / 100.0
        self.max_brightness_value_label.setText(f"{value:.2f}")
        self.keyboard_thread.max_brightness = value
    
    def update_smoothing(self):
        """Update smoothing factor"""
        value = self.smoothing_slider.value() / 100.0
        self.smoothing_value_label.setText(f"{value:.2f}")
        self.keyboard_thread.smoothing_factor = value
    
    def update_manual_brightness(self):
        """Update manual brightness value"""
        value = self.manual_brightness_slider.value() / 100.0
        self.manual_brightness_value_label.setText(f"{value:.2f}")
        
        # Update keyboard brightness if in manual mode
        if not self.auto_mode_checkbox.isChecked():
            self.keyboard_thread.target_brightness = value
    
    def toggle_auto_mode(self):
        """Toggle between auto and manual mode"""
        if self.auto_mode_checkbox.isChecked():
            self.status_bar.showMessage("Auto mode: Brightness reacts to music")
        else:
            self.status_bar.showMessage("Manual mode: Use slider to set brightness")
            # Set brightness to current manual slider value
            value = self.manual_brightness_slider.value() / 100.0
            self.keyboard_thread.target_brightness = value
    
    def toggle_rgb_mode(self):
        """Toggle RGB color mode"""
        use_rgb = self.rgb_mode_checkbox.isChecked()
        self.keyboard_thread.use_rgb = use_rgb
        
        if use_rgb:
            self.status_bar.showMessage("RGB mode: Colors change based on music")
            # Switch to RGB tab
            self.tabs.setCurrentWidget(self.rgb_tab)
        else:
            self.status_bar.showMessage("Brightness mode: Only brightness changes")
    
    def toggle_simulation(self):
        """Toggle simulation mode"""
        self.audio_thread.simulate = self.simulate_checkbox.isChecked()
        
        if self.audio_thread.simulate:
            self.status_bar.showMessage("Simulation mode: Using generated audio patterns")
        else:
            self.status_bar.showMessage("Live mode: Capturing real audio")
    
    def update_bass_range(self):
        """Update bass frequency range"""
        min_freq = self.min_freq_spin.value()
        max_freq = self.max_freq_spin.value()
        
        # Ensure min < max
        if min_freq >= max_freq:
            self.max_freq_spin.setValue(min_freq + 10)
            max_freq = min_freq + 10
        
        self.audio_thread.bass_min_freq = min_freq
        self.audio_thread.bass_max_freq = max_freq
        self.status_bar.showMessage(f"Bass range updated: {min_freq}-{max_freq} Hz")
    
    def refresh_devices(self):
        """Refresh the list of audio devices"""
        # Clear current items
        self.device_combo.clear()
        self.device_combo.addItem("Auto-detect", None)
        
        # Add available devices
        for device_id, device_name in self.audio_thread.get_audio_devices():
            self.device_combo.addItem(device_name, device_id)
        
        self.status_bar.showMessage("Audio devices refreshed")
    
    def select_quiet_color(self):
        """Open color dialog to select quiet color"""
        color = QColorDialog.getColor(self.keyboard_thread.quiet_color, self, "Select Quiet Color")
        
        if color.isValid():
            self.keyboard_thread.quiet_color = color
            self.quiet_color_button.setStyleSheet(f"background-color: {color.name()}")
            self.color_preview.set_colors(self.keyboard_thread.quiet_color, self.keyboard_thread.loud_color)
    
    def select_loud_color(self):
        """Open color dialog to select loud color"""
        color = QColorDialog.getColor(self.keyboard_thread.loud_color, self, "Select Loud Color")
        
        if color.isValid():
            self.keyboard_thread.loud_color = color
            self.loud_color_button.setStyleSheet(f"background-color: {color.name()}")
            self.color_preview.set_colors(self.keyboard_thread.quiet_color, self.keyboard_thread.loud_color)
    
    def update_preview(self):
        """Update color preview based on slider value"""
        value = self.preview_slider.value() / 100.0
        self.color_preview.set_level(value)
    
    def apply_color_preset(self, quiet_color, loud_color):
        """Apply a color preset"""
        self.keyboard_thread.quiet_color = quiet_color
        self.keyboard_thread.loud_color = loud_color
        
        # Update UI
        self.quiet_color_button.setStyleSheet(f"background-color: {quiet_color.name()}")
        self.loud_color_button.setStyleSheet(f"background-color: {loud_color.name()}")
        self.color_preview.set_colors(quiet_color, loud_color)
        
        self.status_bar.showMessage(f"Applied color preset: {quiet_color.name()} → {loud_color.name()}")
    
    def apply_preset(self):
        """Apply preset settings based on music genre and effect preference"""
        genre = self.genre_combo.currentText().lower()
        effect = self.effect_combo.currentText().lower()
        
        # Default settings
        sensitivity = 5.0
        smoothing = 0.3
        
        # Settings based on our performance testing
        settings = {
            "electronic": {
                "responsive": {"sensitivity": 10.0, "smoothing": 0.9},
                "stable": {"sensitivity": 10.0, "smoothing": 0.1},
                "dynamic": {"sensitivity": 1.0, "smoothing": 0.1}
            },
            "rock": {
                "responsive": {"sensitivity": 1.0, "smoothing": 0.1},
                "stable": {"sensitivity": 1.0, "smoothing": 0.9},
                "dynamic": {"sensitivity": 1.0, "smoothing": 0.1}
            },
            "classical": {
                "responsive": {"sensitivity": 10.0, "smoothing": 0.9},
                "stable": {"sensitivity": 1.0, "smoothing": 0.1},
                "dynamic": {"sensitivity": 10.0, "smoothing": 0.5}
            },
            "hip-hop": {
                "responsive": {"sensitivity": 1.0, "smoothing": 0.1},
                "stable": {"sensitivity": 1.0, "smoothing": 0.9},
                "dynamic": {"sensitivity": 1.0, "smoothing": 0.1}
            }
        }
        
        # Get settings if available
        if genre in settings and effect in settings[genre]:
            sensitivity = settings[genre][effect]["sensitivity"]
            smoothing = settings[genre][effect]["smoothing"]
        
        # Update UI and settings
        self.sensitivity_slider.setValue(int(sensitivity * 10))
        self.smoothing_slider.setValue(int(smoothing * 100))
        
        self.status_bar.showMessage(f"Applied preset for {genre} music with {effect} effect")
    
    def get_settings_dict(self):
        """Get current settings as a dictionary"""
        settings = {
            # Audio settings
            "sensitivity": self.sensitivity_slider.value() / 10.0,
            "bass_min_freq": self.min_freq_spin.value(),
            "bass_max_freq": self.max_freq_spin.value(),
            "device_index": self.device_combo.currentData(),
            "use_loopback": self.loopback_checkbox.isChecked(),
            "simulate": self.simulate_checkbox.isChecked(),
            
            # Brightness settings
            "min_brightness": self.min_brightness_slider.value() / 100.0,
            "max_brightness": self.max_brightness_slider.value() / 100.0,
            "smoothing": self.smoothing_slider.value() / 100.0,
            "manual_brightness": self.manual_brightness_slider.value() / 100.0,
            
            # Mode settings
            "auto_mode": self.auto_mode_checkbox.isChecked(),
            "rgb_mode": self.rgb_mode_checkbox.isChecked(),
            
            # Color settings
            "quiet_color": {
                "r": self.keyboard_thread.quiet_color.red(),
                "g": self.keyboard_thread.quiet_color.green(),
                "b": self.keyboard_thread.quiet_color.blue()
            },
            "loud_color": {
                "r": self.keyboard_thread.loud_color.red(),
                "g": self.keyboard_thread.loud_color.green(),
                "b": self.keyboard_thread.loud_color.blue()
            },
            
            # Keyboard settings
            "vendor_id": self.vendor_id_input.text(),
            "product_id": self.product_id_input.text(),
            "interface_number": self.interface_number_input.value(),
            
            # Preset settings
            "genre": self.genre_combo.currentText(),
            "effect": self.effect_combo.currentText(),
            
            # System tray settings
            "minimize_to_tray": self.minimize_to_tray_checkbox.isChecked(),
            "start_minimized": self.start_minimized_checkbox.isChecked()
        }
        
        return settings
    
    def apply_settings_dict(self, settings):
        """Apply settings from a dictionary"""
        # Audio settings
        if "sensitivity" in settings:
            self.sensitivity_slider.setValue(int(settings["sensitivity"] * 10))
        
        if "bass_min_freq" in settings:
            self.min_freq_spin.setValue(settings["bass_min_freq"])
        
        if "bass_max_freq" in settings:
            self.max_freq_spin.setValue(settings["bass_max_freq"])
        
        if "device_index" in settings and settings["device_index"] is not None:
            index = self.device_combo.findData(settings["device_index"])
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
        
        if "use_loopback" in settings:
            self.loopback_checkbox.setChecked(settings["use_loopback"])
        
        if "simulate" in settings:
            self.simulate_checkbox.setChecked(settings["simulate"])
        
        # Brightness settings
        if "min_brightness" in settings:
            self.min_brightness_slider.setValue(int(settings["min_brightness"] * 100))
        
        if "max_brightness" in settings:
            self.max_brightness_slider.setValue(int(settings["max_brightness"] * 100))
        
        if "smoothing" in settings:
            self.smoothing_slider.setValue(int(settings["smoothing"] * 100))
        
        if "manual_brightness" in settings:
            self.manual_brightness_slider.setValue(int(settings["manual_brightness"] * 100))
        
        # Mode settings
        if "auto_mode" in settings:
            self.auto_mode_checkbox.setChecked(settings["auto_mode"])
        
        if "rgb_mode" in settings:
            self.rgb_mode_checkbox.setChecked(settings["rgb_mode"])
            self.keyboard_thread.use_rgb = settings["rgb_mode"]
        
        # Color settings
        if "quiet_color" in settings:
            color = settings["quiet_color"]
            quiet_color = QColor(color["r"], color["g"], color["b"])
            self.keyboard_thread.quiet_color = quiet_color
            self.quiet_color_button.setStyleSheet(f"background-color: {quiet_color.name()}")
        
        if "loud_color" in settings:
            color = settings["loud_color"]
            loud_color = QColor(color["r"], color["g"], color["b"])
            self.keyboard_thread.loud_color = loud_color
            self.loud_color_button.setStyleSheet(f"background-color: {loud_color.name()}")
        
        # Update color preview
        self.color_preview.set_colors(self.keyboard_thread.quiet_color, self.keyboard_thread.loud_color)
        
        # Keyboard settings
        if "vendor_id" in settings:
            self.vendor_id_input.setText(settings["vendor_id"])
        
        if "product_id" in settings:
            self.product_id_input.setText(settings["product_id"])
        
        if "interface_number" in settings:
            self.interface_number_input.setValue(settings["interface_number"])
        
        # Preset settings
        if "genre" in settings:
            index = self.genre_combo.findText(settings["genre"])
            if index >= 0:
                self.genre_combo.setCurrentIndex(index)
        
        if "effect" in settings:
            index = self.effect_combo.findText(settings["effect"])
            if index >= 0:
                self.effect_combo.setCurrentIndex(index)
        
        # System tray settings
        if "minimize_to_tray" in settings:
            self.minimize_to_tray_checkbox.setChecked(settings["minimize_to_tray"])
        
        if "start_minimized" in settings:
            self.start_minimized_checkbox.setChecked(settings["start_minimized"])
    
    def save_settings(self):
        """Save current settings to a file"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "", "Settings Files (*.json);;All Files (*)"
        )
        
        if not file_name:
            return
        
        # Add .json extension if not present
        if not file_name.endswith(".json"):
            file_name += ".json"
        
        settings = self.get_settings_dict()
        
        try:
            with open(file_name, "w") as f:
                json.dump(settings, f, indent=4)
            
            self.status_bar.showMessage(f"Settings saved to {file_name}")
        except Exception as e:
            self.show_error(f"Error saving settings: {e}")
    
    def load_settings(self):
        """Load settings from a file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "Settings Files (*.json);;All Files (*)"
        )
        
        if not file_name:
            return
        
        try:
            with open(file_name, "r") as f:
                settings = json.load(f)
            
            self.apply_settings_dict(settings)
            
            self.status_bar.showMessage(f"Settings loaded from {file_name}")
        except Exception as e:
            self.show_error(f"Error loading settings: {e}")
    
    def load_default_settings(self):
        """Load default settings from config file if it exists"""
        config_dir = os.path.join(os.path.expanduser("~"), ".music_reactive_keyboard")
        config_file = os.path.join(config_dir, "config.json")
        
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    settings = json.load(f)
                
                self.apply_settings_dict(settings)
                self.status_bar.showMessage("Loaded saved configuration")
            except Exception as e:
                self.status_bar.showMessage(f"Error loading configuration: {e}")
    
    def save_default_settings(self):
        """Save current settings as default"""
        config_dir = os.path.join(os.path.expanduser("~"), ".music_reactive_keyboard")
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, "config.json")
        
        settings = self.get_settings_dict()
        
        try:
            with open(config_file, "w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.status_bar.showMessage(f"Error saving configuration: {e}")
    
    def start_audio(self):
        """Start audio capture"""
        # Update audio thread settings
        device_index = self.device_combo.currentData()
        self.audio_thread.device_index = device_index
        self.audio_thread.use_loopback = self.loopback_checkbox.isChecked()
        
        # Start thread
        self.audio_thread.start()
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.audio_status_label.setText("Audio: Running")
        self.audio_status_label.setStyleSheet("color: green;")
        self.status_bar.showMessage("Audio capture started")
        
        # Update tray menu
        self.start_stop_action.setText("Stop Audio")
    
    def stop_audio(self):
        """Stop audio capture"""
        # Stop thread
        self.audio_thread.running = False
        self.audio_thread.wait()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.audio_status_label.setText("Audio: Stopped")
        self.audio_status_label.setStyleSheet("color: red;")
        self.status_bar.showMessage("Audio capture stopped")
        
        # Update tray menu
        self.start_stop_action.setText("Start Audio")
    
    def exit_application(self):
        """Exit the application"""
        # Save current settings
        self.save_default_settings()
        
        # Stop threads
        self.audio_thread.running = False
        self.keyboard_thread.running = False
        
        # Wait for threads to finish
        self.audio_thread.wait()
        self.keyboard_thread.wait()
        
        # Exit
        QApplication.quit()
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.minimize_to_tray_checkbox.isChecked():
            event.ignore()
            self.hide()
            self.show_hide_action.setText("Show")
            
            # Show balloon message
            self.tray_icon.showMessage(
                "Music-Reactive Keyboard LED Controller",
                "Application minimized to system tray. Click the tray icon to restore.",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
        else:
            self.exit_application()
            event.accept()

def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    window = MainWindow()
    
    # Check if should start minimized
    if window.start_minimized_checkbox.isChecked():
        window.hide()
        window.show_hide_action.setText("Show")
    else:
        window.show()
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
