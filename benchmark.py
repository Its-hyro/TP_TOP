import subprocess
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path
from statistics import mean, stdev
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark de multiplication matricielle')
    parser.add_argument('--layout', choices=['right', 'left'], required=True,
                     help='Layout à tester (right ou left)')
    parser.add_argument('--threads', type=int, nargs='+', required=True,
                      help='Nombre(s) de threads à tester (ex: 1 2 4 8)')
    return parser.parse_args()

def run_benchmark(m, n, k, layout, num_threads=None, num_runs=3):
    """Exécute le programme avec les paramètres donnés et retourne les résultats moyens"""
    times_ms = []
    gflops_list = []
    
    for _ in range(num_runs):
        # Chemin correct vers l'exécutable
        cmd = ["./build/src/top.matrix_product", str(m), str(n), str(k), layout]
        
        # Création de l'environnement avec les variables OpenMP
        env = dict(os.environ)
        if num_threads is not None:
            env["OMP_NUM_THREADS"] = str(num_threads)
            # Configuration optimale pour OpenMP 4.0+
            env["OMP_PROC_BIND"] = "spread"
            env["OMP_PLACES"] = "threads"
            # Configuration supplémentaire pour optimiser les performances
            env["OMP_SCHEDULE"] = "dynamic"
            env["OMP_DYNAMIC"] = "false"
            env["OMP_NESTED"] = "false"
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        # Extraction des résultats
        time_ms = None
        gflops = None
        
        for line in result.stdout.split('\n'):
            if "Temps d'exécution:" in line:
                time_ms = float(line.split(':')[1].strip().split()[0])
            elif "Performance:" in line:
                gflops = float(line.split(':')[1].strip().split()[0])
        
        if time_ms is not None and gflops is not None:
            times_ms.append(time_ms)
            gflops_list.append(gflops)
    
    return {
        'time_ms_mean': mean(times_ms),
        'time_ms_std': stdev(times_ms) if len(times_ms) > 1 else 0,
        'gflops_mean': mean(gflops_list),
        'gflops_std': stdev(gflops_list) if len(gflops_list) > 1 else 0
    }

def benchmark_sizes(layout, num_threads=None):
    """Effectue le benchmark pour différentes tailles de matrices"""
    # Tailles en puissance de 2 : de 2^1 (2) à 2^11 (2048)
    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    results = []
    
    for size in sizes:
        print(f"Benchmarking size {size}x{size} with {layout} layout")
        stats = run_benchmark(size, size, size, layout, num_threads)
        results.append({
            'size': size,
            'layout': layout,
            'threads': num_threads if num_threads is not None else 1,
            'time_ms_mean': stats['time_ms_mean'],
            'time_ms_std': stats['time_ms_std'],
            'gflops_mean': stats['gflops_mean'],
            'gflops_std': stats['gflops_std']
        })
    
    return results

def benchmark_threads(layout, threads):
    """Effectue le benchmark pour différents nombres de threads"""
    size = 1024  # Taille fixe pour la comparaison des threads
    
    results = []
    
    for num_threads in threads:
        print(f"Benchmarking with {num_threads} threads and {layout} layout")
        stats = run_benchmark(size, size, size, layout, num_threads)
        results.append({
            'threads': num_threads,
            'layout': layout,
            'time_ms_mean': stats['time_ms_mean'],
            'time_ms_std': stats['time_ms_std'],
            'gflops_mean': stats['gflops_mean'],
            'gflops_std': stats['gflops_std']
        })
    
    return results

def save_results(results, filename):
    """Sauvegarde les résultats dans un fichier CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

def plot_size_results(results):
    """Génère les graphiques pour les différentes tailles"""
    sizes = sorted(set(r['size'] for r in results))
    layout = results[0]['layout']
    threads = results[0]['threads']
    
    plt.figure(figsize=(12, 5))
    
    # Graphique des performances (GFLOP/s)
    plt.subplot(1, 2, 1)
    gflops_means = [r['gflops_mean'] for r in results]
    gflops_stds = [r['gflops_std'] for r in results]
    plt.errorbar(sizes, gflops_means, yerr=gflops_stds, fmt='o-', 
                capsize=5, label=f'Layout {layout} ({threads} thread(s))')
    plt.xlabel('Taille de la matrice')
    plt.ylabel('Performance (GFLOP/s)')
    plt.title(f'Performance en fonction de la taille\nLayout: {layout}, Threads: {threads}')
    plt.legend()
    plt.grid(True)
    
    # Graphique du temps d'exécution
    plt.subplot(1, 2, 2)
    time_means = [r['time_ms_mean'] for r in results]
    time_stds = [r['time_ms_std'] for r in results]
    plt.errorbar(sizes, time_means, yerr=time_stds, fmt='o-', 
                capsize=5, label=f'Layout {layout} ({threads} thread(s))')
    plt.xlabel('Taille de la matrice')
    plt.ylabel('Temps d\'exécution (ms)')
    plt.title(f'Temps d\'exécution en fonction de la taille\nLayout: {layout}, Threads: {threads}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'benchmark_sizes_{layout}_{threads}threads.png')
    plt.close()

def plot_thread_results(results):
    """Génère les graphiques pour les différents nombres de threads"""
    threads = sorted(set(r['threads'] for r in results))
    layout = results[0]['layout']
    
    plt.figure(figsize=(12, 5))
    
    # Graphique des performances (GFLOP/s)
    plt.subplot(1, 2, 1)
    gflops_means = [r['gflops_mean'] for r in results]
    gflops_stds = [r['gflops_std'] for r in results]
    plt.errorbar(threads, gflops_means, yerr=gflops_stds, fmt='o-', 
                capsize=5, label=f'Layout {layout}')
    plt.xlabel('Nombre de threads')
    plt.ylabel('Performance (GFLOP/s)')
    plt.title(f'Performance en fonction du nombre de threads\nLayout: {layout}')
    plt.legend()
    plt.grid(True)
    
    # Graphique du temps d'exécution
    plt.subplot(1, 2, 2)
    time_means = [r['time_ms_mean'] for r in results]
    time_stds = [r['time_ms_std'] for r in results]
    plt.errorbar(threads, time_means, yerr=time_stds, fmt='o-', 
                capsize=5, label=f'Layout {layout}')
    plt.xlabel('Nombre de threads')
    plt.ylabel('Temps d\'exécution (ms)')
    plt.title(f'Temps d\'exécution en fonction du nombre de threads\nLayout: {layout}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'benchmark_threads_{layout}.png')
    plt.close()

def main():
    args = parse_args()
    
    # Benchmark des tailles pour chaque nombre de threads
    for num_threads in args.threads:
        print(f"\nDébut du benchmark des tailles avec {num_threads} thread(s)...")
        size_results = benchmark_sizes(args.layout, num_threads)
        save_results(size_results, f'benchmark_sizes_{args.layout}_{num_threads}threads.csv')
        plot_size_results(size_results)
    
    # Benchmark des threads
    print("\nDébut du benchmark des threads...")
    thread_results = benchmark_threads(args.layout, args.threads)
    save_results(thread_results, f'benchmark_threads_{args.layout}.csv')
    plot_thread_results(thread_results)

if __name__ == "__main__":
    main() 