import os
import subprocess
import time

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    return process.returncode

def main():
    start_time = time.time()
    
    print("Step 1: Training model on WESAD (2 epochs, all subjects)...")
    train_code = run_command("python cross_eval/train_and_save.py")
    
    if train_code != 0:
        print("Training failed!")
        return

    print("\nStep 2: Running inference on Affective Road...")
    infer_code = run_command("python cross_eval/run_inference.py")
    
    if infer_code != 0:
        print("Inference failed!")
        return

    end_time = time.time()
    duration_min = (end_time - start_time) / 60
    print(f"\nTotal execution time: {duration_min:.2f} minutes")

if __name__ == "__main__":
    main()
