import matplotlib.pyplot as plt
import csv
import os
import sys
import numpy as np

def charger_resultats_csv(chemin_fichier):
    """Charge les données depuis un fichier CSV"""
    resultats = []
    with open(chemin_fichier, 'r') as f:
        lecteur = csv.DictReader(f)
        for ligne in lecteur:
            # Conversion des chaînes en nombres
            for cle in ['size', 'threads', 'time_ms_mean', 'time_ms_std', 'gflops_mean', 'gflops_std']:
                if cle in ligne:
                    ligne[cle] = float(ligne[cle])
            resultats.append(ligne)
    return resultats

def detecter_layouts(*dossiers):
    """Détecte les layouts utilisés dans les dossiers"""
    layouts = []
    
    for dossier in dossiers:
        layout = None
        for fichier in os.listdir(dossier):
            if fichier.startswith("benchmark_threads_"):
                layout = fichier.split("benchmark_threads_")[1].split(".")[0]
                break
        if layout is None:
            raise FileNotFoundError(f"Impossible de trouver les fichiers de benchmark dans le dossier {dossier}")
        layouts.append(layout)
    
    return layouts

def calcul_ratio_et_erreur(a, b, da, db):
    """
    Calcule le ratio a/b et son erreur avec gestion des cas particuliers
    
    Parameters:
    -----------
    a, b : float
        Valeurs moyennes
    da, db : float
        Écarts-types correspondants
    
    Returns:
    --------
    ratio, ratio_error : float
        Ratio et son erreur associée
    """
    if abs(b) < 1e-10 or abs(a) < 1e-10:  # Éviter division par zéro
        return 0, 0
    
    ratio = a/b
    
    # Calcul de l'erreur relative avec vérification des divisions par zéro
    rel_error_a = da/a if abs(a) > 1e-10 else 0
    rel_error_b = db/b if abs(b) > 1e-10 else 0
    
    # Propagation d'erreur pour la division
    ratio_error = abs(ratio) * np.sqrt(rel_error_a**2 + rel_error_b**2)
    
    return ratio, ratio_error

def calculer_speedup(donnees):
    """
    Calcule le speedup pour chaque configuration
    """
    speedups = []
    speedup_errors = []
    
    for data in donnees:
        # Trouver les données pour 1 thread (référence)
        t1 = next(r for r in data if r['threads'] == 1.0)
        
        config_speedups = []
        config_errors = []
        
        for r in data:
            # Calcul du speedup T(1)/T(p)
            speedup, speedup_error = calcul_ratio_et_erreur(
                t1['time_ms_mean'], r['time_ms_mean'],
                t1['time_ms_std'], r['time_ms_std']
            )
            config_speedups.append(speedup)
            config_errors.append(speedup_error)
            
        speedups.append(config_speedups)
        speedup_errors.append(config_errors)
    
    return speedups, speedup_errors

def comparer_performances_threads(*args):
    """Compare les performances des configurations en fonction du nombre de threads"""
    dossier_sortie = "comparaison_resultats/barre d'errur"
    os.makedirs(dossier_sortie, exist_ok=True)

    dossiers = args[::3]
    noms = args[1::3]
    layouts = args[2::3]

    donnees = []
    for dossier, layout in zip(dossiers, layouts):
        donnees.append(charger_resultats_csv(f"{dossier}/benchmark_threads_{layout}.csv"))

    # Création de la figure avec trois sous-graphiques
    plt.figure(figsize=(20, 6))

    # 1. Performance (GFLOP/s)
    plt.subplot(1, 3, 1)
    for data, nom in zip(donnees, noms):
        plt.errorbar([r['threads'] for r in data], 
                    [r['gflops_mean'] for r in data],
                    yerr=[r['gflops_std'] for r in data],
                    fmt='o-', label=nom, capsize=5, markersize=8)
    plt.xlabel('Nombre de threads')
    plt.ylabel('Performance (GFLOP/s)')
    plt.title('Comparaison des performances\nentre les configurations')
    plt.legend()
    plt.grid(True)
    plt.xticks([1, 2, 4, 8])

    # 2. Temps d'exécution
    plt.subplot(1, 3, 2)
    for data, nom in zip(donnees, noms):
        plt.errorbar([r['threads'] for r in data], 
                    [r['time_ms_mean'] for r in data],
                    yerr=[r['time_ms_std'] for r in data],
                    fmt='o-', label=nom, capsize=5, markersize=8)
    plt.xlabel('Nombre de threads')
    plt.ylabel('Temps d\'exécution (ms)')
    plt.title('Comparaison des temps d\'exécution\nentre les configurations')
    plt.legend()
    plt.grid(True)
    plt.xticks([1, 2, 4, 8])
    plt.yscale('log')

    # 3. Speedup
    plt.subplot(1, 3, 3)
    speedups, speedup_errors = calculer_speedup(donnees)
    threads = [r['threads'] for r in donnees[0]]
    
    # Ligne de speedup idéal
    plt.plot(threads, threads, 'k--', label='Speedup idéal')
    
    for (speedup, error), nom in zip(zip(speedups, speedup_errors), noms):
        plt.errorbar(threads, speedup, 
                    yerr=error,
                    fmt='o-', label=nom, capsize=5, markersize=8)
    
    plt.xlabel('Nombre de threads')
    plt.ylabel('Speedup')
    plt.title('Speedup en fonction du nombre de threads')
    plt.legend()
    plt.grid(True)
    plt.xticks([1, 2, 4, 8])

    plt.tight_layout()
    plt.savefig(f'{dossier_sortie}/comparaison_configs_threads.png', dpi=300, bbox_inches='tight')
    plt.close()

def comparer_performances(*args):
    """Compare les performances des configurations avec barres d'erreur améliorées"""
    dossier_sortie = "comparaison_resultats/barre d'errur"
    os.makedirs(dossier_sortie, exist_ok=True)

    dossiers = args[::3]
    noms = args[1::3]
    layouts = args[2::3]

    for threads in [1, 2, 4, 8]:
        donnees = []
        for dossier, layout in zip(dossiers, layouts):
            donnees.append(charger_resultats_csv(f"{dossier}/benchmark_sizes_{layout}_{threads}threads.csv"))

        plt.figure(figsize=(15, 5))

        # Performance (GFLOP/s)
        plt.subplot(1, 3, 1)
        for data, nom in zip(donnees, noms):
            plt.errorbar([r['size'] for r in data], 
                        [r['gflops_mean'] for r in data],
                        yerr=[r['gflops_std'] for r in data],
                        fmt='o-', label=f'{nom} ({threads} threads)',
                        capsize=5, markersize=8)
        plt.xlabel('Taille de la matrice')
        plt.ylabel('Performance (GFLOP/s)')
        plt.title(f'Comparaison des performances\n{threads} thread(s)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log', base=2)

        # Temps d'exécution
        plt.subplot(1, 3, 2)
        for data, nom in zip(donnees, noms):
            plt.errorbar([r['size'] for r in data], 
                        [r['time_ms_mean'] for r in data],
                        yerr=[r['time_ms_std'] for r in data],
                        fmt='o-', label=f'{nom} ({threads} threads)',
                        capsize=5, markersize=8)
        plt.xlabel('Taille de la matrice')
        plt.ylabel('Temps d\'exécution (ms)')
        plt.title(f'Comparaison des temps d\'exécution\n{threads} thread(s)')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.xscale('log', base=2)

        # Ratios avec propagation d'erreur améliorée
        plt.subplot(1, 3, 3)
        sizes = [r['size'] for r in donnees[0]]
        for i in range(1, len(donnees)):
            ratios = []
            ratio_errors = []
            for d1, d2 in zip(donnees[0], donnees[i]):
                ratio, ratio_error = calcul_ratio_et_erreur(
                    d2['gflops_mean'], d1['gflops_mean'],
                    d2['gflops_std'], d1['gflops_std']
                )
                ratios.append(ratio)
                ratio_errors.append(ratio_error)
            
            plt.errorbar(sizes, ratios, 
                        yerr=ratio_errors,
                        fmt='o-', label=f'Ratio {noms[i]}/{noms[0]}',
                        capsize=5, markersize=8)
        
        plt.axhline(y=1, color='r', linestyle='--', label='Égal')
        plt.xlabel('Taille de la matrice')
        plt.ylabel('Ratios de performance')
        plt.title(f'Ratios des performances\n{threads} thread(s)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log', base=2)

        plt.tight_layout()
        plt.savefig(f'{dossier_sortie}/comparaison_{threads}threads.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_layouts.py <dossier_config1> <dossier_config2> [dossier_config3]")
        print("Exemple: python compare_layouts.py result/right_1_8_threads result/right_cb_1_8 result/right_cb_2_8")
        sys.exit(1)

    dossiers = sys.argv[1:]
    noms = [os.path.basename(dossier) for dossier in dossiers]
    layouts = detecter_layouts(*dossiers)
    
    args = []
    for dossier, nom, layout in zip(dossiers, noms, layouts):
        args.extend([dossier, nom, layout])
    
    comparer_performances(*args)
    comparer_performances_threads(*args) 