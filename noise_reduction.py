import os
import time
import librosa
import soundfile as sf
import noisereduce as nr
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# The noise reduction problem was failing after transforming some files
# As we could not solve the error problem, we included in the code a retry section, which every time the code fails, it reruns it again
# In that way we were able to process all the files 

input_root = 'augmented_audio'
output_root = 'augmented_audio_denoised'

Path(output_root).mkdir(parents=True, exist_ok=True)

def process_audio_file(input_output_pair):
    input_path, output_path = input_output_pair
    if not Path(output_path).exists():
        try:
            audio, sr = librosa.load(input_path, sr=None)
            reduced_noise_audio = nr.reduce_noise(y=audio, sr=sr, thresh_n_mult_nonstationary=2, stationary=False)
            sf.write(output_path, reduced_noise_audio, sr)
            print(f"Processed and saved {output_path}")
            return output_path, None
        except Exception as e:
            return input_path, str(e)
    else:
        print(f"Skipping {output_path}, already exists.")
        return output_path, None

def collect_files(input_dir, output_dir):
    file_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.ogg'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, root[len(input_dir):].lstrip('/\\'), file)
                file_paths.append((input_path, output_path))
                Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    return file_paths

def process_chunk(chunk):
    errors = []
    for input_output_pair in chunk:
        path, error = process_audio_file(input_output_pair)
        if error:
            errors.append(path)
    return errors

def process_files_in_batches(file_paths):
    errors = []
    chunk_size = 10  
    chunks = [file_paths[i:i + chunk_size] for i in range(0, len(file_paths), chunk_size)]
    try:
        with ProcessPoolExecutor(max_workers=4) as executor:  
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                chunk_errors = future.result()
                errors.extend(chunk_errors)
                gc.collect()  
        return errors
    except Exception as e:
        print(f"Caught an exception: {e}")
        return file_paths  

def retry_failed_processing(input_root, output_root):
    try:
        file_paths = collect_files(input_root, output_root)
        while file_paths:
            print(f"Starting or retrying processing of {len(file_paths)} files...")
            file_paths = process_files_in_batches(file_paths)
            if not file_paths:
                print("All files processed successfully.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Unhandled exception: {e}")
        print("Will attempt to restart processing in 10 seconds...")
        time.sleep(10)
        retry_failed_processing(input_root, output_root)  


retry_failed_processing(input_root, output_root)
