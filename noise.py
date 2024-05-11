import os
import librosa
import noisereduce as nr
import soundfile as sf

def reduce_noise_in_files(input_folder, output_folder):
    # Walk through the input directory
    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.ogg'):
                # Prepare paths
                input_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Load audio file
                audio, sr = librosa.load(input_path, sr=None)

                # Select a noise profile (first second of audio)
                noise_clip = audio[0:int(1 * sr)]

                # Apply noise reduction
                reduced_noise_audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip)

                # Save the processed audio
                sf.write(output_path, reduced_noise_audio, sr, format='OGG')
                print(f"Processed {input_path} and saved to {output_path}")

# Set your directories here
source_directory = '/home/muhammad.sheikh/Downloads/ai701_project_final_new/birdclef-2023/augmented_audio/'
target_directory = '/home/muhammad.sheikh/Downloads/ai701_project_final_new/birdclef-2023/noise-rem/'
reduce_noise_in_files(source_directory, target_directory)
