#!/bin/bash

# Configuration
MATRIX_SIZE=1024
THREADS=8
LAYOUT="right"

# Fonction pour exécuter le programme avec profilage
run_with_profiling() {
    echo "Profilage avec $2 threads pour la taille $1 x $1"
    
    # Lancer powermetrics en arrière-plan
    sudo powermetrics --samplers cpu_power,cpu_counters -i 100 --show-process-energy > cache_profile_${1}_${2}.txt &
    METRICS_PID=$!
    
    # Lancer le programme
    ./build/src/top.matrix_product $1 $1 $1 $3
    
    # Arrêter powermetrics
    sudo kill $METRICS_PID
    
    echo "Profilage terminé. Résultats dans cache_profile_${1}_${2}.txt"
}

# Créer le répertoire build si nécessaire
mkdir -p build
cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
cd ..

# Tests avec différentes configurations
echo "=== Début du profilage ==="

# Version non bloquée
run_with_profiling $MATRIX_SIZE $THREADS $LAYOUT

# Version avec cache blocking
run_with_profiling $MATRIX_SIZE $THREADS "${LAYOUT}_blocked"

echo "=== Profilage terminé ==="

# Analyse des résultats
echo "=== Analyse des résultats ==="
for file in cache_profile_*.txt; do
    echo "Analyse de $file :"
    grep -A 5 "CPU cache misses" "$file"
done 