#!/usr/bin/env python3
"""
Automated Evaluation Script for All Project Parts
==================================================

This script automatically runs evaluation for all trained models across:
- LM Part A: 4 models (RNN, LSTM, LSTM+Dropout, LSTM+Dropout+AdamW)
- LM Part B: 3 models (BASE, VARDROP, FULL)
- NLU Part A: 3 models (IAS_BASELINE, IAS_BIDIR, IAS_BIDIR_DROPOUT)
- NLU Part B: 2 models (bert-base, bert-large)

Usage: python run_all_evaluations.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_section(text):
    """Print a formatted section header"""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{'-'*60}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{'-'*60}{Colors.ENDC}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️ {text}{Colors.ENDC}")

def check_file_exists(file_path):
    """Check if a file exists"""
    return Path(file_path).exists()

def modify_main_file(file_path, model_config, evaluation_path, backup_suffix=".backup"):
    """
    Modify a main.py file to set evaluation mode and model path
    Returns the original content for restoration
    """
    try:
        # Read original content
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        # Create backup
        backup_path = file_path + backup_suffix
        with open(backup_path, 'w') as f:
            f.write(original_content)
        
        # Modify content
        modified_content = original_content
        
        # Set evaluation mode to True
        if 'EVALUATION_MODE = False' in modified_content:
            modified_content = modified_content.replace('EVALUATION_MODE = False', 'EVALUATION_MODE = True')
        elif 'EVALUATION_MODE = True' not in modified_content:
            # Find where to insert EVALUATION_MODE
            if 'MODEL_CONFIG' in modified_content:
                modified_content = modified_content.replace(
                    'MODEL_CONFIG =', 
                    'EVALUATION_MODE = True  # Auto-set by evaluation script\nMODEL_CONFIG ='
                )
        
        # Set model config if provided
        if model_config:
            # Find and replace MODEL_CONFIG line
            lines = modified_content.split('\n')
            for i, line in enumerate(lines):
                if 'MODEL_CONFIG =' in line and not line.strip().startswith('#'):
                    lines[i] = f'MODEL_CONFIG = "{model_config}"  # Auto-set by evaluation script'
                    break
            modified_content = '\n'.join(lines)
        
        # Set evaluation model path if provided
        if evaluation_path:
            lines = modified_content.split('\n')
            for i, line in enumerate(lines):
                if 'EVALUATION_MODEL_PATH =' in line and not line.strip().startswith('#'):
                    lines[i] = f'EVALUATION_MODEL_PATH = "{evaluation_path}"  # Auto-set by evaluation script'
                    break
            modified_content = '\n'.join(lines)
        
        # Write modified content
        with open(file_path, 'w') as f:
            f.write(modified_content)
        
        return original_content
    
    except Exception as e:
        print_error(f"Failed to modify {file_path}: {str(e)}")
        return None

def restore_main_file(file_path, backup_suffix=".backup"):
    """Restore original main.py file from backup"""
    try:
        backup_path = file_path + backup_suffix
        if Path(backup_path).exists():
            with open(backup_path, 'r') as f:
                original_content = f.read()
            
            with open(file_path, 'w') as f:
                f.write(original_content)
            
            # Remove backup
            os.remove(backup_path)
            return True
    except Exception as e:
        print_error(f"Failed to restore {file_path}: {str(e)}")
        return False

def run_evaluation(script_path, model_name, timeout=300):
    """
    Run evaluation script and capture output using conda environment
    Returns (success, output, error)
    """
    try:
        print(f"   🔄 Running {model_name}...")
        
        # Get the project root directory (where this script is located)
        project_root = "/Users/marcoprosperi/Desktop/257857_Marco_Prosperi"
        original_dir = os.getcwd()
        
        # Change to project root directory (scripts expect to be run from here)
        os.chdir(project_root)
        
        # Get the conda environment name
        conda_env_name = "nlu25"
        
        # Method 1: Try to use conda run (recommended approach)
        try:
            result = subprocess.run(
                ["conda", "run", "-n", conda_env_name, "python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except FileNotFoundError:
            # Method 2: Fallback - try to find conda python directly
            conda_base = os.environ.get('CONDA_PREFIX_1', os.environ.get('CONDA_PREFIX', ''))
            if conda_base:
                # Try different possible paths for conda environment
                possible_paths = [
                    os.path.join(os.path.dirname(conda_base), 'envs', conda_env_name, 'bin', 'python'),
                    os.path.join(conda_base, 'envs', conda_env_name, 'bin', 'python'),
                    os.path.join(os.path.expanduser('~'), 'anaconda3', 'envs', conda_env_name, 'bin', 'python'),
                    os.path.join(os.path.expanduser('~'), 'miniconda3', 'envs', conda_env_name, 'bin', 'python')
                ]
                
                conda_python = None
                for path in possible_paths:
                    if os.path.exists(path):
                        conda_python = path
                        break
                
                if conda_python:
                    result = subprocess.run(
                        [conda_python, script_path],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                else:
                    raise FileNotFoundError(f"Could not find conda environment '{conda_env_name}' python interpreter")
            else:
                # Method 3: Last resort - use system python but try to activate conda env first
                activate_script = f"source activate {conda_env_name} && python {script_path}"
                result = subprocess.run(
                    ["/bin/bash", "-c", activate_script],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
        
        # Return to original directory
        os.chdir(original_dir)
        
        if result.returncode == 0:
            return True, result.stdout, result.stderr
        else:
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        os.chdir(original_dir)
        return False, "", f"Timeout after {timeout} seconds"
    except Exception as e:
        os.chdir(original_dir)
        return False, "", str(e)

def extract_results(output, task_type):
    """Extract key results from output based on task type"""
    lines = output.split('\n')
    results = {}
    
    if task_type == "LM":
        # Look for perplexity results
        for line in lines:
            if "Test Perplexity:" in line:
                try:
                    ppl = float(line.split("Test Perplexity:")[-1].strip())
                    results["perplexity"] = ppl
                except:
                    pass
            elif "Test ppl:" in line:
                try:
                    ppl = float(line.split("Test ppl:")[-1].strip())
                    results["perplexity"] = ppl
                except:
                    pass
    
    elif task_type == "NLU":
        # Look for F1 and accuracy results
        for line in lines:
            if "Test Slot F1 Score:" in line:
                try:
                    f1 = float(line.split("Test Slot F1 Score:")[-1].strip())
                    results["slot_f1"] = f1
                except:
                    pass
            elif "Test Intent Accuracy:" in line:
                try:
                    acc = float(line.split("Test Intent Accuracy:")[-1].strip())
                    results["intent_accuracy"] = acc
                except:
                    pass
            elif "Slot F1" in line and "+-" in line:
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        f1 = float(parts[2])
                        results["slot_f1"] = f1
                except:
                    pass
            elif "Intent Acc" in line and "+-" in line:
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        acc = float(parts[2])
                        results["intent_accuracy"] = acc
                except:
                    pass
    
    return results

def run_lm_part_a():
    """Run all LM Part A evaluations"""
    print_section("🔤 LANGUAGE MODELING - PART A")
    
    base_path = "/Users/marcoprosperi/Desktop/257857_Marco_Prosperi/LM/part_A"
    main_file = os.path.join(base_path, "main.py")
    
    models = [
        ("RNN", "LM/part_A/bin/RNN_baseline/weights.pt"),
        ("LSTM", "LM/part_A/bin/LSTM/weights.pt"),
        ("LSTM_DROPOUT", "LM/part_A/bin/LSTM_Drop/weights.pt"),
        ("LSTM_DROPOUT_ADAMW", "LM/part_A/bin/LSTM_Drop_AdamW/weights.pt")
    ]
    
    results = {}
    
    for model_config, model_path in models:
        print(f"\n📊 Evaluating {model_config}...")
        
        # Check if model file exists
        full_model_path = f"/Users/marcoprosperi/Desktop/257857_Marco_Prosperi/{model_path}"
        if not check_file_exists(full_model_path):
            print_warning(f"Model file not found: {model_path}")
            continue
        
        # Modify main.py
        original_content = modify_main_file(main_file, model_config, model_path)
        if original_content is None:
            continue
        
        try:
            # Run evaluation
            success, output, error = run_evaluation(main_file, model_config)
            
            if success:
                model_results = extract_results(output, "LM")
                if "perplexity" in model_results:
                    results[model_config] = model_results["perplexity"]
                    print_success(f"{model_config}: Perplexity = {model_results['perplexity']:.4f}")
                else:
                    print_warning(f"{model_config}: Could not extract perplexity from output")
            else:
                print_error(f"{model_config}: Evaluation failed")
                if error:
                    print(f"   Error: {error}")
        
        finally:
            # Restore original file
            restore_main_file(main_file)
    
    return results

def run_lm_part_b():
    """Run all LM Part B evaluations"""
    print_section("🔤 LANGUAGE MODELING - PART B")
    
    base_path = "/Users/marcoprosperi/Desktop/257857_Marco_Prosperi/LM/part_B"
    main_file = os.path.join(base_path, "main.py")
    
    models = [
        ("BASE", "LM/part_B/bin/LSTM_WT/weights.pt"),
        ("VARDROP", "LM/part_B/bin/LSTM_WT_VD/weights.pt"),
        ("FULL", "LM/part_B/bin/LSTM_WT_VD_avSGD/weights.pt")
    ]
    
    results = {}
    
    for model_config, model_path in models:
        print(f"\n📊 Evaluating {model_config}...")
        
        # Check if model file exists
        full_model_path = f"/Users/marcoprosperi/Desktop/257857_Marco_Prosperi/{model_path}"
        if not check_file_exists(full_model_path):
            print_warning(f"Model file not found: {model_path}")
            continue
        
        # Modify main.py
        original_content = modify_main_file(main_file, model_config, model_path)
        if original_content is None:
            continue
        
        try:
            # Run evaluation
            success, output, error = run_evaluation(main_file, model_config)
            
            if success:
                model_results = extract_results(output, "LM")
                if "perplexity" in model_results:
                    results[model_config] = model_results["perplexity"]
                    print_success(f"{model_config}: Perplexity = {model_results['perplexity']:.4f}")
                else:
                    print_warning(f"{model_config}: Could not extract perplexity from output")
            else:
                print_error(f"{model_config}: Evaluation failed")
                if error:
                    print(f"   Error: {error}")
        
        finally:
            # Restore original file
            restore_main_file(main_file)
    
    return results

def run_nlu_part_a():
    """Run all NLU Part A evaluations"""
    print_section("🗣️ NATURAL LANGUAGE UNDERSTANDING - PART A")
    
    base_path = "/Users/marcoprosperi/Desktop/257857_Marco_Prosperi/NLU/part_A"
    main_file = os.path.join(base_path, "main.py")
    
    models = [
        ("IAS_BASELINE", "NLU/part_A/bin/IAS_BASELINE/weights_1.pt"),
        ("IAS_BIDIR", "NLU/part_A/bin/IAS_BIDIR/weights_1.pt"),
        ("IAS_BIDIR_DROPOUT", "NLU/part_A/bin/IAS_BIDIR_DROPOUT/weights_1.pt")
    ]
    
    results = {}
    
    for model_config, model_path in models:
        print(f"\n📊 Evaluating {model_config}...")
        
        # Check if model file exists
        full_model_path = f"/Users/marcoprosperi/Desktop/257857_Marco_Prosperi/{model_path}"
        if not check_file_exists(full_model_path):
            print_warning(f"Model file not found: {model_path}")
            continue
        
        # Modify main.py
        original_content = modify_main_file(main_file, model_config, model_path)
        if original_content is None:
            continue
        
        try:
            # Run evaluation
            success, output, error = run_evaluation(main_file, model_config, timeout=180)
            
            if success:
                model_results = extract_results(output, "NLU")
                if "slot_f1" in model_results and "intent_accuracy" in model_results:
                    results[model_config] = {
                        "slot_f1": model_results["slot_f1"],
                        "intent_accuracy": model_results["intent_accuracy"]
                    }
                    print_success(f"{model_config}: Slot F1 = {model_results['slot_f1']:.4f}, Intent Acc = {model_results['intent_accuracy']:.4f}")
                else:
                    print_warning(f"{model_config}: Could not extract metrics from output")
            else:
                print_error(f"{model_config}: Evaluation failed")
                if error:
                    print(f"   Error: {error}")
        
        finally:
            # Restore original file
            restore_main_file(main_file)
    
    return results

def run_nlu_part_b():
    """Run all NLU Part B evaluations"""
    print_section("🗣️ NATURAL LANGUAGE UNDERSTANDING - PART B")
    
    base_path = "/Users/marcoprosperi/Desktop/257857_Marco_Prosperi/NLU/part_B"
    main_file = os.path.join(base_path, "main.py")
    
    models = [
        ("bert-base-uncased", "NLU/part_B/bin/bert-base/weights.pt"),
        ("bert-large-uncased", "NLU/part_B/bin/bert-large/weights.pt")
    ]
    
    results = {}
    
    for model_config, model_path in models:
        print(f"\n📊 Evaluating {model_config}...")
        
        # Check if model file exists
        full_model_path = f"/Users/marcoprosperi/Desktop/257857_Marco_Prosperi/{model_path}"
        if not check_file_exists(full_model_path):
            print_warning(f"Model file not found: {model_path}")
            continue
        
        # For NLU Part B, we need to modify both MODEL_CONFIG and bert_model_name
        original_content = modify_main_file(main_file, None, model_path)
        if original_content is None:
            continue
        
        # Additional modification for bert_model_name
        try:
            with open(main_file, 'r') as f:
                content = f.read()
            
            # Replace bert_model_name
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'bert_model_name =' in line and not line.strip().startswith('#'):
                    lines[i] = f'bert_model_name = "{model_config}"  # Auto-set by evaluation script'
                    break
            
            with open(main_file, 'w') as f:
                f.write('\n'.join(lines))
                
        except Exception as e:
            print_error(f"Failed to modify bert_model_name: {str(e)}")
            restore_main_file(main_file)
            continue
        
        try:
            # Run evaluation (longer timeout for BERT models)
            success, output, error = run_evaluation(main_file, model_config, timeout=600)
            
            if success:
                model_results = extract_results(output, "NLU")
                if "slot_f1" in model_results and "intent_accuracy" in model_results:
                    results[model_config] = {
                        "slot_f1": model_results["slot_f1"],
                        "intent_accuracy": model_results["intent_accuracy"]
                    }
                    print_success(f"{model_config}: Slot F1 = {model_results['slot_f1']:.4f}, Intent Acc = {model_results['intent_accuracy']:.4f}")
                else:
                    print_warning(f"{model_config}: Could not extract metrics from output")
            else:
                print_error(f"{model_config}: Evaluation failed")
                if error:
                    print(f"   Error: {error}")
        
        finally:
            # Restore original file
            restore_main_file(main_file)
    
    return results

def print_summary(lm_a_results, lm_b_results, nlu_a_results, nlu_b_results):
    """Print a comprehensive summary of all results"""
    print_header("📊 COMPREHENSIVE EVALUATION SUMMARY")
    
    # LM Part A Summary
    if lm_a_results:
        print(f"{Colors.OKBLUE}{Colors.BOLD}🔤 Language Modeling - Part A (Perplexity ↓){Colors.ENDC}")
        for model, ppl in lm_a_results.items():
            print(f"   {model:<20}: {ppl:>8.4f}")
        
        # Find best model
        best_model = min(lm_a_results.items(), key=lambda x: x[1])
        print(f"   {Colors.OKGREEN}🏆 Best: {best_model[0]} (PPL: {best_model[1]:.4f}){Colors.ENDC}")
    
    # LM Part B Summary
    if lm_b_results:
        print(f"\n{Colors.OKBLUE}{Colors.BOLD}🔤 Language Modeling - Part B (Perplexity ↓){Colors.ENDC}")
        for model, ppl in lm_b_results.items():
            print(f"   {model:<20}: {ppl:>8.4f}")
        
        # Find best model
        best_model = min(lm_b_results.items(), key=lambda x: x[1])
        print(f"   {Colors.OKGREEN}🏆 Best: {best_model[0]} (PPL: {best_model[1]:.4f}){Colors.ENDC}")
    
    # NLU Part A Summary
    if nlu_a_results:
        print(f"\n{Colors.OKBLUE}{Colors.BOLD}🗣️ NLU - Part A (F1 Score ↑ | Intent Acc ↑){Colors.ENDC}")
        for model, metrics in nlu_a_results.items():
            print(f"   {model:<20}: F1={metrics['slot_f1']:>6.4f} | Acc={metrics['intent_accuracy']:>6.4f}")
        
        # Find best model by F1
        best_model = max(nlu_a_results.items(), key=lambda x: x[1]['slot_f1'])
        print(f"   {Colors.OKGREEN}🏆 Best F1: {best_model[0]} (F1: {best_model[1]['slot_f1']:.4f}){Colors.ENDC}")
    
    # NLU Part B Summary
    if nlu_b_results:
        print(f"\n{Colors.OKBLUE}{Colors.BOLD}🗣️ NLU - Part B (F1 Score ↑ | Intent Acc ↑){Colors.ENDC}")
        for model, metrics in nlu_b_results.items():
            print(f"   {model:<20}: F1={metrics['slot_f1']:>6.4f} | Acc={metrics['intent_accuracy']:>6.4f}")
        
        # Find best model by F1
        best_model = max(nlu_b_results.items(), key=lambda x: x[1]['slot_f1'])
        print(f"   {Colors.OKGREEN}🏆 Best F1: {best_model[0]} (F1: {best_model[1]['slot_f1']:.4f}){Colors.ENDC}")
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")

def main():
    """Main execution function"""
    start_time = time.time()
    
    print_header("🚀 AUTOMATED EVALUATION SUITE")
    print(f"{Colors.OKCYAN}Starting comprehensive evaluation of all trained models...{Colors.ENDC}")
    print(f"{Colors.OKCYAN}This may take several minutes to complete.{Colors.ENDC}")
    
    # Check if we're in the right directory
    expected_path = "/Users/marcoprosperi/Desktop/257857_Marco_Prosperi"
    if not os.path.exists(expected_path):
        print_error(f"Expected project directory not found: {expected_path}")
        print("Please make sure you're running this script from the correct location.")
        sys.exit(1)
    
    # Initialize results
    lm_a_results = {}
    lm_b_results = {}
    nlu_a_results = {}
    nlu_b_results = {}
    
    try:
        # Run all evaluations
        lm_a_results = run_lm_part_a()
        lm_b_results = run_lm_part_b()
        nlu_a_results = run_nlu_part_a()
        nlu_b_results = run_nlu_part_b()
        
        # Print comprehensive summary
        print_summary(lm_a_results, lm_b_results, nlu_a_results, nlu_b_results)
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"\n{Colors.OKGREEN}✅ Evaluation completed in {total_time:.1f} seconds{Colors.ENDC}")
        
    except KeyboardInterrupt:
        print_warning("\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
