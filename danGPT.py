#!/usr/bin/env python3


###############################################################################
#                                                                             #
#                            DANIEL J. MUELLER                                #
#                          GALACTO CORPORATION                                #
#                                                                             #
#                         Date: November 24, 2024                             #
#                                                                             #
# --------------------------------------------------------------------------- #
#                               DAN-GPT PIPELINE                              #
#                                                                             #
# Purpose:                                                                    #
# This pipeline is designed for generating, processing, and synthesizing      #
# thoughts in an o1-like fashion using state-of-the-art language models.      #
# It leverages REST API integrations, GPU acceleration, and a modular design  #
# for efficient and scalable AI-driven analysis.                              #
#                                                                             #
# Key Features:                                                               #
# - Distributed Thought Generation using multiple GPUs                        #
# - Realization Processing to consolidate and refine generated ideas          #
# - Final Output Synthesis for actionable insights and decisions              #
#                                                                             #
# Technologies:                                                               #
# - REST API for model inference                                              #
# - NVIDIA A100 GPUs                                                          #
# - Python multiprocessing for parallelism                                    #
# - Advanced logging and error handling                                       #
#                                                                             #
# Built with precision and forward-thinking innovation by Daniel J. Mueller,  #
# inspired by a vision to bridge technology with human ingenuity.             #
#                                                                             #
###############################################################################

import os
import subprocess
import logging
from datetime import datetime
import math
import requests
import json
import time
import threading
import signal
import sys

# ----------------------------- Configuration -----------------------------

# Directories
THOUGHTS_DIR = "/home/daniel/Desktop/DanGPT/Thoughts"
REALIZATIONS_DIR = os.path.join(THOUGHTS_DIR, "Realizations")
LOGS_DIR = "/home/daniel/Desktop/DanGPT/Logs"
OUTPUTS_DIR = "/home/daniel/Desktop/DanGPT/Outputs"

# Logging Configuration
LOG_FILE = os.path.join(LOGS_DIR, "pipeline.log")

# Temperature Settings
MAX_TEMP = 1.0
MIN_TEMP = 0.5

# Batch Size
BATCH_SIZE = 3  # Modify this value as needed

# Number of Thoughts
NUM_THOUGHTS = 9  # Total number of thoughts to generate

# GPU Configuration
# GPU index to use (assuming one GPU)
GPU_ID = 2  # You can change this to the GPU you want to use

# Prompt Templates
with open("prompt.txt", "r") as file:
    ORIGINAL_PROMPT = file.read()

THOUGHT_PROMPT = "{original} Analyze and list considerations which must be made while analyzing the problem, focusing on specific methods of achieving the goal from meaningful angles. Consider and list off all possibilities, and list off features which may improve the implementation."

REALIZATION_PROMPT = "Combine all of the following thoughts into a comprehensive set of considerations and code which comprehensively factor in what you've learned from the generated information. You should generate all the components necessary to satisfy all the conditions presented.: {batch}"

FINAL_OUTPUT_PROMPT = "{original} Utilize the following realized information to accomplish the original prompt in its entirety. Irregardless of what comes next, remember what the objective is. You should take care to recognize and account for any error which may be present in your output. Incorporate all the features from the included information, cohesively. Output 1 singular codeblock. It is okay to outline details beforehand.: {realizations}"

# REST API Configuration
REST_API_URL = "http://localhost:11434/api/generate"

# Model Name
MODEL_NAME = "mistral-large:123b"

# Temperature Throttling Configuration
GPU_TEMP_THRESHOLD = 75  # Temperature threshold in Celsius
GPU_TEMP_CHECK_INTERVAL = 2  # Interval to check GPU temperature in seconds

# -------------------------------------------------------------------------

class Logger:
    """Sets up logging for the pipeline."""
    def __init__(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger()

    def info(self, message):
        print(message)
        self.logger.info(message)

    def warning(self, message):
        print(f"WARNING: {message}")
        self.logger.warning(message)

    def error(self, message):
        print(f"ERROR: {message}")
        self.logger.error(message)

class GPUMonitor(threading.Thread):
    """Monitors GPU temperature and controls processing based on threshold."""
    def __init__(self, temp_threshold, check_interval, logger, pause_event):
        super().__init__()
        self.temp_threshold = temp_threshold
        self.check_interval = check_interval
        self.logger = logger
        self.stop_event = threading.Event()
        self.pause_event = pause_event
        self.is_throttled = False

    def run(self):
        self.logger.info(f"Starting GPU monitor (threshold: {self.temp_threshold}°C, interval: {self.check_interval}s)")
        while not self.stop_event.is_set():
            gpu_temps = self.get_gpu_temperatures()
            if gpu_temps:
                temps_info = ', '.join([f"GPU{index}: {temp}°C" for index, temp in gpu_temps])
                self.logger.info(f"Current GPU Temperatures: {temps_info}")
                temps_only = [temp for index, temp in gpu_temps]
                if any(temp >= self.temp_threshold for temp in temps_only) and not self.is_throttled:
                    self.logger.warning("GPU temperature threshold exceeded. Throttling processing...")
                    self.pause_event.set()
                    self.is_throttled = True
                elif all(temp < (self.temp_threshold - 5) for temp in temps_only) and self.is_throttled:
                    self.logger.info("GPU temperatures normalized. Resuming processing...")
                    self.pause_event.clear()
                    self.is_throttled = False
            else:
                self.logger.error("No GPU temperatures retrieved. Skipping monitoring cycle.")
            time.sleep(self.check_interval)

    def get_gpu_temperatures(self):
        try:
            # Get temperatures of all GPUs, ignoring CUDA_VISIBLE_DEVICES
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                env={}  # Empty environment to avoid inheriting CUDA_VISIBLE_DEVICES
            )
            if result.returncode != 0:
                self.logger.error(f"Failed to get GPU temperatures: {result.stderr}")
                return []
            lines = result.stdout.strip().split('\n')
            gpu_temps = []
            for line in lines:
                if line.strip() == '':
                    continue
                index, temp = line.strip().split(',')
                gpu_temps.append((int(index.strip()), float(temp.strip())))
            return gpu_temps
        except Exception as e:
            self.logger.error(f"Error getting GPU temperatures: {str(e)}")
            return []

    def stop(self):
        self.stop_event.set()

def generate_thought(gpu_id, thought_id, temperature, thought_prompt, original_prompt, thoughts_dir, model_name, rest_api_url, pause_event):
    """Generates a single thought using the REST API."""
    thought_file = os.path.join(thoughts_dir, f"thought_{thought_id}.txt")
    prompt = thought_prompt.format(original=original_prompt)
    logger = logging.getLogger()
    logger.info(f"Generating thought {thought_id} on GPU {gpu_id} with temperature {temperature}")

    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "max_tokens": 80000,
                "num_ctx": 32768
            }
        }
        headers = {
            'Content-Type': 'application/json',
            'X-CUDA-DEVICE': str(gpu_id)  # Optional: If your REST API supports specifying GPU via headers
        }

        # Set environment variables for the REST API call
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Introduce temperature throttling check
        while pause_event.is_set():
            logger.info(f"Pausing thought generation {thought_id} due to high GPU temperature...")
            time.sleep(1)

        response = requests.post(rest_api_url, headers=headers, data=json.dumps(payload), stream=True)

        if response.status_code != 200:
            logger.error(f"Thought {thought_id} generation failed: {response.text}")
            return None

        thought = ''
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                content = data.get('response', '')
                thought += content

        thought = thought.strip()
        if not thought:
            logger.error(f"Thought {thought_id} generation failed: Empty response")
            return None

        with open(thought_file, 'w') as f:
            f.write(thought)
        logger.info(f"Thought {thought_id} saved to {thought_file}")
        return thought
    except Exception as e:
        logger.error(f"Exception in generating thought {thought_id}: {str(e)}")
        return None

def generate_realization(batch, realization_id, gpu_id, realization_prompt, realizations_dir, model_name, rest_api_url, pause_event):
    """Generates a single realization."""
    combined_thoughts = " ".join(batch)
    realization_file = os.path.join(realizations_dir, f"realization_{realization_id}.txt")
    prompt = realization_prompt.format(batch=combined_thoughts)
    logger = logging.getLogger()
    logger.info(f"Processing realization batch {realization_id} on GPU {gpu_id}")

    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "options": {
                "temperature": 0.7,  # Set a default temperature or adjust as needed
                "max_tokens": 80000,
                "num_ctx": 32768
            }
        }
        headers = {
            'Content-Type': 'application/json',
            'X-CUDA-DEVICE': str(gpu_id)  # Optional: If your REST API supports specifying GPU via headers
        }

        # Set environment variables for the REST API call
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Introduce temperature throttling check
        while pause_event.is_set():
            logger.info(f"Pausing realization processing {realization_id} due to high GPU temperature...")
            time.sleep(1)

        response = requests.post(rest_api_url, headers=headers, data=json.dumps(payload), stream=True)

        if response.status_code != 200:
            logger.error(f"Realization batch {realization_id} failed: {response.text}")
            return None

        realization = ''
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                content = data.get('response', '')
                realization += content

        realization = realization.strip()
        if not realization:
            logger.error(f"Realization batch {realization_id} failed: Empty response")
            return None

        with open(realization_file, 'w') as f:
            f.write(realization)
        logger.info(f"Realization {realization_id} saved to {realization_file}")
        return realization
    except Exception as e:
        logger.error(f"Exception in processing realization {realization_id}: {str(e)}")
        return None

def generate_final_output(original_prompt, realizations, output_file, gpu_id, final_output_prompt, model_name, rest_api_url, pause_event):
    """Generates the final output."""
    logger = logging.getLogger()
    logger.info("Starting final output generation...")
    combined_realizations = " ".join(realizations)
    prompt = final_output_prompt.format(original=original_prompt, realizations=combined_realizations)
    logger.info(f"Generating final output on GPU {gpu_id}")

    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "options": {
                "temperature": 0.7,  # Set a default temperature or adjust as needed
                "max_tokens": 80000,
                "num_ctx": 32768
            }
        }
        headers = {
            'Content-Type': 'application/json',
            'X-CUDA-DEVICE': str(gpu_id)  # Optional: If your REST API supports specifying GPU via headers
        }

        # Set environment variables for the REST API call
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Introduce temperature throttling check
        while pause_event.is_set():
            logger.info("Pausing final output generation due to high GPU temperature...")
            time.sleep(1)

        response = requests.post(rest_api_url, headers=headers, data=json.dumps(payload), stream=True)

        if response.status_code != 200:
            logger.error(f"Final output generation failed: {response.text}")
            return None

        final_output = ''
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                content = data.get('response', '')
                final_output += content

        final_output = final_output.strip()
        if not final_output:
            logger.error("Final output generation failed: Empty response")
            return None

        with open(output_file, 'w') as f:
            f.write(final_output)
        logger.info(f"Final output saved to {output_file}")
        return final_output
    except Exception as e:
        logger.error(f"Exception in generating final output: {str(e)}")
        return None

class DanGPTPipeline:
    """Orchestrates the entire pipeline."""
    def __init__(self, original_prompt, thought_prompt, realization_prompt, final_output_prompt, batch_size=3, num_thoughts=9):
        self.original_prompt = original_prompt
        self.batch_size = batch_size
        self.num_thoughts = num_thoughts
        self.logger = Logger(LOG_FILE)
        self.logger.info("Initialized DanGPT Pipeline.")

        # Create a shared pause_event
        self.pause_event = threading.Event()

        # Initialize GPU Monitor
        self.gpu_monitor = GPUMonitor(
            temp_threshold=GPU_TEMP_THRESHOLD,
            check_interval=GPU_TEMP_CHECK_INTERVAL,
            logger=self.logger,
            pause_event=self.pause_event
        )

        # Directories
        self.thoughts_dir = THOUGHTS_DIR
        self.realizations_dir = REALIZATIONS_DIR
        self.outputs_dir = OUTPUTS_DIR
        os.makedirs(self.thoughts_dir, exist_ok=True)
        os.makedirs(self.realizations_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Prompts
        self.thought_prompt = thought_prompt
        self.realization_prompt = realization_prompt
        self.final_output_prompt = final_output_prompt

        # Temperature values
        self.temperature_values = self.calculate_temperatures()

    def calculate_temperatures(self):
        """Calculates evenly spaced temperatures between max_temp and min_temp."""
        temps = []
        for i in range(self.num_thoughts):
            temp = MAX_TEMP - (i * (MAX_TEMP - MIN_TEMP) / (self.num_thoughts - 1))
            temps.append(round(temp, 4))
        self.logger.info(f"Calculated temperatures: {temps}")
        return temps

    def run(self):
        """Runs the pipeline."""
        self.logger.info("Pipeline execution started.")

        # Start GPU Monitor thread
        self.gpu_monitor.start()

        # Handle shutdown signals
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        try:
            # Step 1: Generate Thoughts
            thoughts = []
            for i in range(self.num_thoughts):
                temp = self.temperature_values[i]
                thought = generate_thought(
                    gpu_id=GPU_ID,
                    thought_id=i+1,
                    temperature=temp,
                    thought_prompt=self.thought_prompt,
                    original_prompt=self.original_prompt,
                    thoughts_dir=self.thoughts_dir,
                    model_name=MODEL_NAME,
                    rest_api_url=REST_API_URL,
                    pause_event=self.pause_event
                )
                if thought is not None:
                    thoughts.append(thought)
                else:
                    self.logger.error(f"Thought {i+1} generation failed.")

            if not thoughts:
                self.logger.error("No thoughts were generated. Exiting pipeline.")
                return

            # Step 2: Process Realizations
            realizations = []
            num_batches = math.ceil(len(thoughts) / self.batch_size)
            for i in range(num_batches):
                batch = thoughts[i*self.batch_size : (i+1)*self.batch_size]
                realization = generate_realization(
                    batch=batch,
                    realization_id=i+1,
                    gpu_id=GPU_ID,
                    realization_prompt=self.realization_prompt,
                    realizations_dir=self.realizations_dir,
                    model_name=MODEL_NAME,
                    rest_api_url=REST_API_URL,
                    pause_event=self.pause_event
                )
                if realization is not None:
                    realizations.append(realization)
                else:
                    self.logger.error(f"Realization {i+1} processing failed.")

            if not realizations:
                self.logger.error("No realizations were processed. Exiting pipeline.")
                return

            # Step 3: Generate Final Output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.outputs_dir, f"output_{timestamp}.txt")
            final_output = generate_final_output(
                original_prompt=self.original_prompt,
                realizations=realizations,
                output_file=output_file,
                gpu_id=GPU_ID,
                final_output_prompt=self.final_output_prompt,
                model_name=MODEL_NAME,
                rest_api_url=REST_API_URL,
                pause_event=self.pause_event
            )
            if final_output:
                self.logger.info("Pipeline execution completed successfully.")
            else:
                self.logger.error("Pipeline execution failed during final output generation.")
        finally:
            # Ensure GPU monitor is stopped
            self.gpu_monitor.stop()
            self.gpu_monitor.join()

    def handle_shutdown(self, signum, frame):
        """Handles graceful shutdown."""
        self.logger.info("Shutting down pipeline...")
        self.gpu_monitor.stop()
        sys.exit(0)

# ----------------------------- Execution -----------------------------

if __name__ == "__main__":
    # Initialize and run the pipeline
    pipeline = DanGPTPipeline(
        original_prompt=ORIGINAL_PROMPT,
        thought_prompt=THOUGHT_PROMPT,
        realization_prompt=REALIZATION_PROMPT,
        final_output_prompt=FINAL_OUTPUT_PROMPT,
        batch_size=BATCH_SIZE,
        num_thoughts=NUM_THOUGHTS
    )
    pipeline.run()
