import pandas as pd
from pathlib import Path

def load_qrels_file(file_path: str) -> dict:
    """
    Load relevance judgments (qrels) file into a dictionary.
    
    Args:
        file_path: Path to the qrels file
        
    Returns:
        Dictionary with (query_id, doc_id) tuple as key and relevance score as value
    """
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:  # qrels files have 4 fields
                continue
            query_id = parts[0]
            doc_id = parts[2]
            relevance = int(parts[3])
            data[(query_id, doc_id)] = relevance
    return data

def load_run_file(file_path: str) -> dict:
    """
    Load system output (run) file into a dictionary.
    
    Args:
        file_path: Path to the run file
        
    Returns:
        Dictionary with (query_id, doc_id) tuple as key and score as value
    """
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:  # run files have 6 fields
                continue
            query_id = parts[0]
            doc_id = parts[2]
            score = float(parts[4])
            data[(query_id, doc_id)] = score
    return data

def merge_runs_with_qrels(run_files: list, qrels_file: str, output_excel: str):
    """
    Merge multiple run files with relevance judgments and save to Excel.
    
    Args:
        run_files: List of paths to run files
        qrels_file: Path to qrels file
        output_excel: Output Excel file path
    """
    # Load all run files
    run_data = [load_run_file(run_file) for run_file in run_files]
    
    # Load qrels
    qrels_data = load_qrels_file(qrels_file)
    
    # Find common documents across all runs and qrels
    common_keys = set(run_data[0].keys())
    for data in run_data[1:]:
        common_keys.intersection_update(data.keys())
    common_keys.intersection_update(qrels_data.keys())
    
    # Prepare data for DataFrame
    rows = []
    for key in common_keys:
        query_id, doc_id = key
        scores = [data[key] for data in run_data]
        relevance = qrels_data[key]
        rows.append([query_id, doc_id, *scores, relevance])
    
    # Create DataFrame and save to Excel
    column_names = ['Query ID', 'Doc ID'] + [f'Score {i+1}' for i in range(len(run_files))] + ['Relevance']
    df = pd.DataFrame(rows, columns=column_names)
    df.to_excel(output_excel, index=False)
    print(f"Merged data saved to {output_excel}")



# File paths for test set
run_files = [
    'runs/bm25-stemmed441.run',
    'runs/laplace-stemmed441.run',
    'runs/jm-stemmed441.run'
]
qrels_file = '../data/qrels.441-450.txt'  # Updated to test set
output_excel = 'test_data.xlsx'  # Changed filename to reflect test set

# Create merged test set
merge_runs_with_qrels(run_files, qrels_file, output_excel)



run_files = [
    'runs/bm25-stemmed401.run',
    'runs/laplace-stemmed401.run',
    'runs/jm-stemmed401.run'
]
qrels_file = '../data/qrels.401-440.txt'  # Updated to test set
output_excel = 'training_data.xlsx'  # Changed filename to reflect test set

# Create merged test set
merge_runs_with_qrels(run_files, qrels_file, output_excel)


