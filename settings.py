import os
from dotenv import load_dotenv
import tensorflow as tf

# --- begin cuBLAS bug workaround ---
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("Executing cuBLAS bug workaround")
    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    )
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
# --- end cuBLAS bug workaround ---

# Load .env file environment variables
load_dotenv()

# Folders
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
LOGS_FOLDER = os.getenv("LOGS_FOLDER", os.path.join(PROJECT_FOLDER, "logs/"))
