import os
from dotenv import load_dotenv
import tensorflow as tf

# cuBLAS bug workaround
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("Executing cuBLAS bug workaround")
    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    )
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)


PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Study env variables
load_dotenv()  # Load .env file
STUDY_EXPERIMENT_VERSION = os.getenv("STUDY_EXPERIMENT_VERSION", "v4")
LOGS_FOLDER = os.getenv("LOGS_FOLDER", os.path.join(PROJECT_FOLDER, f"logs/logs_{STUDY_EXPERIMENT_VERSION}"))
