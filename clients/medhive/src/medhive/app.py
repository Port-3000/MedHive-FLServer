import importlib.metadata
import sys
import threading
import random

from PySide6 import QtWidgets, QtCore
from .client_logic import start_fl_client


class MedHive(QtWidgets.QMainWindow):
    log_signal = QtCore.Signal(str)
    status_signal = QtCore.Signal(str)
    
    def __init__(self):
        super().__init__()
        self.fl_thread = None
        self.stop_fl_event = threading.Event()
        self.init_ui()
        
        # Connect signals
        self.log_signal.connect(self._update_log_area)
        self.status_signal.connect(self._update_status_label)

    def init_ui(self):
        # Main widget and layout
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Server address input
        server_layout = QtWidgets.QHBoxLayout()
        server_label = QtWidgets.QLabel("FL Server Address:")
        server_label.setMinimumWidth(150)
        self.server_input = QtWidgets.QLineEdit("localhost:8089")
        server_layout.addWidget(server_label)
        server_layout.addWidget(self.server_input)
        
        # Client ID input
        client_id_layout = QtWidgets.QHBoxLayout()
        client_id_label = QtWidgets.QLabel("Client ID:")
        client_id_label.setMinimumWidth(150)
        default_client_id = str(random.randint(1000, 9999))
        self.client_id_input = QtWidgets.QLineEdit(default_client_id)
        client_id_layout.addWidget(client_id_label)
        client_id_layout.addWidget(self.client_id_input)
        
        # Buttons
        self.connect_button = QtWidgets.QPushButton("Connect & Start Training")
        self.connect_button.clicked.connect(self.handle_connect)
        
        self.disconnect_button = QtWidgets.QPushButton("Disconnect (Stop Training)")
        self.disconnect_button.clicked.connect(self.handle_disconnect)
        self.disconnect_button.setEnabled(False)
        
        # Status label
        self.status_label = QtWidgets.QLabel("Status: Disconnected")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # Log area
        self.log_area = QtWidgets.QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(200)
        
        # Add all widgets to main layout
        main_layout.addLayout(server_layout)
        main_layout.addLayout(client_id_layout)
        main_layout.addWidget(self.connect_button)
        main_layout.addWidget(self.disconnect_button)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.log_area)
        
        # Set the central widget
        self.setCentralWidget(main_widget)
        
        # Set window title
        self.setWindowTitle("MedHive FL Client")
        self.resize(600, 400)  # Set a reasonable initial size
        self.show()

    def log(self, message):
        """Thread-safe logging to the UI"""
        self.log_signal.emit(message)

    def _update_log_area(self, message):
        """Update log area with new message"""
        self.log_area.append(message)
        # Scroll to bottom
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_status(self, message):
        """Thread-safe status update"""
        self.status_signal.emit(message)

    def _update_status_label(self, message):
        """Update the status label text"""
        self.status_label.setText(f"Status: {message}")

    def set_controls_state(self, is_connected):
        """Enable/disable UI controls based on connection state"""
        self.connect_button.setEnabled(not is_connected)
        self.disconnect_button.setEnabled(is_connected)
        self.server_input.setReadOnly(is_connected)
        self.client_id_input.setReadOnly(is_connected)

    def fl_runner(self, server_address, client_id):
        """Runs the Flower client in a separate thread."""
        self.log(f"Thread started for client {client_id}.")
        self.set_status(f"Connecting to {server_address}...")
        try:
            # Redirect stdout/stderr or use a custom logger to capture Flower output
            start_fl_client(server_address=server_address, client_id=client_id)
            if not self.stop_fl_event.is_set():
                self.log(f"Client {client_id} task completed normally.")
                self.set_status("Task Completed")
        except Exception as e:
            self.log(f"!!! Client {client_id} Error: {e}")
            if not self.stop_fl_event.is_set():
                self.set_status("Error Occurred")
        finally:
            # Ensure state is updated even if thread crashes or finishes
            if not self.stop_fl_event.is_set():  # Don't reset state if disconnect was pressed
                # Use invoke_later to update UI from non-UI thread
                QtCore.QMetaObject.invokeMethod(
                    self, 
                    "set_controls_state", 
                    QtCore.Qt.QueuedConnection, 
                    QtCore.Q_ARG(bool, False)
                )
            self.log(f"Thread finished for client {client_id}.")

    def handle_connect(self):
        """Handle connect button press"""
        server_address = self.server_input.text()
        client_id = self.client_id_input.text()
        if not server_address or not client_id:
            QtWidgets.QMessageBox.warning(
                self, 
                "Input Error", 
                "Please enter Server Address and Client ID."
            )
            return

        self.log(f"Attempting connection to {server_address} as client {client_id}...")
        self.set_controls_state(is_connected=True)
        self.set_status("Initializing...")
        self.log_area.clear()  # Clear log
        self.stop_fl_event.clear()  # Reset stop event

        self.fl_thread = threading.Thread(
            target=self.fl_runner,
            args=(server_address, client_id),
            daemon=True  # Allows app to exit even if thread is running
        )
        self.fl_thread.start()

    def handle_disconnect(self):
        """Handle disconnect button press"""
        self.log("Disconnect requested by user.")
        self.set_status("Disconnecting...")
        self.stop_fl_event.set()  # Signal the thread to stop

        # Update UI immediately
        self.set_controls_state(is_connected=False)
        self.set_status("Disconnected")


def main():
    # Linux desktop environments use an app's .desktop file to integrate the app
    # in to their application menus. The .desktop file of this app will include
    # the StartupWMClass key, set to app's formal name. This helps associate the
    # app's windows to its menu item.
    #
    # For association to work, any windows of the app must have WMCLASS property
    # set to match the value set in app's desktop file. For PySide6, this is set
    # with setApplicationName().

    # Find the name of the module that was used to start the app
    app_module = sys.modules["__main__"].__package__
    # Retrieve the app's metadata
    metadata = importlib.metadata.metadata(app_module)

    QtWidgets.QApplication.setApplicationName(metadata["Formal-Name"])

    app = QtWidgets.QApplication(sys.argv)
    main_window = MedHive()
    sys.exit(app.exec())
