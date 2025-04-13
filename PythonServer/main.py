#!/usr/bin/env python3
"""
Tempest AI Main Entry Point
Coordinates the socket server, metrics display, and keyboard handling.
"""

import os
import time
import threading
from datetime import datetime
import traceback

from aimodel import (
    DQNAgent, KeyboardHandler
)
from config import (
    RL_CONFIG, MODEL_DIR, LATEST_MODEL_PATH, IS_INTERACTIVE, metrics, SERVER_CONFIG
)
from socket_server import SocketServer
from metrics_display import display_metrics_header, display_metrics_row

def stats_reporter(agent, kb_handler):
    """Thread function to report stats periodically"""
    print("Starting stats reporter thread...")
    last_report = time.time()
    report_interval = 10.0  # Report every 10 seconds
    
    # Display the header once at the beginning
    display_metrics_header()
    
    while True:
        try:
            current_time = time.time()
            if current_time - last_report >= report_interval:
                display_metrics_row(agent, kb_handler)
                last_report = current_time
            
            # Check if server is still running
            if metrics.global_server is None or not metrics.global_server.running:
                print("Server stopped running, exiting stats reporter")
                break
                
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in stats reporter: {e}")
            traceback.print_exc()
            break

def keyboard_input_handler(agent, keyboard_handler):
    """Thread function to handle keyboard input"""
    print("Starting keyboard input handler thread...")
    
    while True:
        try:
            # Check for keyboard input
            key = keyboard_handler.check_key()
            
            if key:
                # Handle different keys
                if key == 'q':
                    print("Quit command received, shutting down...")
                    metrics.global_server.running = False
                    break
                elif key == 's':
                    print("Save command received, saving model...")
                    agent.save(LATEST_MODEL_PATH)
                    print(f"Model saved to {LATEST_MODEL_PATH}")
                elif key == 'o':
                    metrics.toggle_override(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key == 'e':
                    metrics.toggle_expert_mode(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
            
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in keyboard input handler: {e}")
            break

def main():
    """Main function to run the Tempest AI application"""
    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Initialize the DQN agent
    agent = DQNAgent(
        state_size=RL_CONFIG.state_size,
        action_size=RL_CONFIG.action_size,
        learning_rate=RL_CONFIG.learning_rate,
        gamma=RL_CONFIG.gamma,
        epsilon=RL_CONFIG.epsilon,
        epsilon_min=RL_CONFIG.epsilon_min,
        epsilon_decay=RL_CONFIG.epsilon_decay,
        memory_size=RL_CONFIG.memory_size,
        batch_size=RL_CONFIG.batch_size
    )
    
    # Load the model if it exists
    if os.path.exists(LATEST_MODEL_PATH):
        agent.load(LATEST_MODEL_PATH)
    
    # Initialize the socket server
    server = SocketServer(SERVER_CONFIG.host, SERVER_CONFIG.port, agent, metrics)
    
    # Set the global server reference in metrics
    metrics.global_server = server
    
    # Initialize client_count in metrics
    metrics.client_count = 0
    
    # Start the server in a separate thread
    server_thread = threading.Thread(target=server.start)
    server_thread.daemon = True
    server_thread.start()
    
    # Set up keyboard handler for interactive mode
    keyboard_handler = None
    if IS_INTERACTIVE:
        keyboard_handler = KeyboardHandler()
        keyboard_handler.setup_terminal()
        keyboard_thread = threading.Thread(target=keyboard_input_handler, args=(agent, keyboard_handler))
        keyboard_thread.daemon = True
        keyboard_thread.start()
    
    # Start the stats reporter in a separate thread
    stats_thread = threading.Thread(target=stats_reporter, args=(agent, keyboard_handler))
    stats_thread.daemon = True
    stats_thread.start()
    
    # Track last save time
    last_save_time = time.time()
    save_interval = 300  # 5 minutes in seconds
    
    try:
        # Keep the main thread alive
        while server.running:
            current_time = time.time()
            # Save model every 5 minutes
            if current_time - last_save_time >= save_interval:
                agent.save(LATEST_MODEL_PATH)
                last_save_time = current_time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, saving and shutting down...")
    finally:
        # Save the model before exiting
        agent.save(LATEST_MODEL_PATH)
        print("Final model state saved")
        
        # Restore terminal settings
        if IS_INTERACTIVE and keyboard_handler:
            keyboard_handler.restore_terminal()
        
        print("Application shutdown complete")

if __name__ == "__main__":
    main() 