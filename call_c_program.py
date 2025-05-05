import subprocess
import os
import shutil

def run_c_program_wsl(c_program_path, input_file, output_file, method=2, knn=1, accuracy=0.90):
    """
    Esegue un programma C attraverso WSL
    - input_file: percorso del file di input nella directory del progetto
    - output_file: percorso del file di output nella directory del progetto
    - method: metodo da utilizzare (default: 2)
    - knn: numero di vicini per KNN (default: 1)
    - accuracy: soglia di accuratezza (default: 0.90)
    """
    # Ottieni il nome del file di output
    output_filename = os.path.basename(output_file)
    
    # Percorso temporaneo in WSL per l'output
    wsl_output_temp = f"/tmp/{output_filename}"
    
    # Converti il percorso di input in formato WSL
    wsl_input = input_file.replace('\\', '/').replace('C:', '/mnt/c')
    
    # Costruisci il comando WSL
    wsl_command = ['wsl', 
                    c_program_path, 
                    wsl_input, 
                    wsl_output_temp, 
                    '-method', str(method),
                    '-knn', str(knn),
                    '-accuracy', str(accuracy)]
    
    try:
        # Esegui il comando
        result = subprocess.run(
            wsl_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Sposta il file da WSL alla directory del progetto
        wsl_copy_command = ['wsl', 'cp', wsl_output_temp, f"/mnt/c/{output_file.replace('\\', '/').replace('C:', '')}"]
        subprocess.run(wsl_copy_command, check=True)
        
        # Pulisci il file temporaneo in WSL
        wsl_clean_command = ['wsl', 'rm', wsl_output_temp]
        subprocess.run(wsl_clean_command, check=True)
        
        # Verifica che il file di output sia stato creato
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Il file di output {output_file} non è stato creato")
            
        print(f"Programma C eseguito con successo")
        print(f"File di input: {input_file}")
        print(f"File di output: {output_file}")
        
        return output_file
        
    except subprocess.CalledProcessError as e:
        print(f"Errore nell'esecuzione del programma C:")
        print(f"Stderr: {e.stderr}")
        raise
    except FileNotFoundError as e:
        print(f"Errore: {e}")
        raise

# Esempio di utilizzo
if __name__ == "__main__":
    # Percorsi dei file (percorsi Windows)
    input_file = r"C:\Users\aless\Cursor project\Experiments-SurveyGreenAI\collision.ds3"
    output_file = r"C:\Users\aless\Cursor project\Experiments-SurveyGreenAI\condensed.ds3"
    
    # Percorso del programma C in WSL
    c_program = "/mnt/c/Users/aless/Desktop/FCNN Fabrizio/fcnn"

    # Esegui il programma
    run_c_program_wsl(c_program, input_file, output_file, method=2, knn=1, accuracy=0.90) 