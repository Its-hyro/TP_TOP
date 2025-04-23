#!/bin/bash

# Configuration
MATRIX_SIZE=1024
THREADS=8
RUNS=3  # Nombre d'exécutions par configuration

# Configurations à tester
LAYOUTS=("right" "left")
BLOCK_SIZES=(
    "16 16 16"    # Petit bloc (L1)
    "32 32 32"    # Bloc moyen (L1-L2)
    "64 64 64"    # Grand bloc (L2)
    "128 128 128" # Très grand bloc (L3)
)

# Configuration OpenMP
export OMP_NUM_THREADS=$THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active
export OMP_DYNAMIC=false

# Fonction pour calculer les statistiques
calculate_stats() {
    local sum=0
    local sum_sq=0
    local count=0
    while read -r value; do
        sum=$(echo "$sum + $value" | bc -l)
        sum_sq=$(echo "$sum_sq + ($value * $value)" | bc -l)
        count=$((count + 1))
    done
    if [ $count -gt 0 ]; then
        local mean=$(echo "scale=6; $sum / $count" | bc -l)
        local variance=$(echo "scale=6; ($sum_sq / $count) - ($mean * $mean)" | bc -l)
        local stddev=$(echo "scale=6; sqrt($variance)" | bc -l 2>/dev/null || echo "0")
        echo "$mean $stddev"
    fi
}

# Fonction pour profiler une configuration
profile_run() {
    local size=$1
    local threads=$2
    local layout=$3
    local run_number=$4
    local block_size="$5"
    
    echo "=== Profilage run $run_number: $layout avec taille ${size}x${size}, $threads threads, blocs $block_size ==="
    
    # Run avec profilage
    /usr/bin/time -l ./build/src/top.matrix_product $size $size $size $layout 2>&1 | tee profile_${layout}_${size}_${threads}_${block_size// /_}_${run_number}.txt
    
    sleep 2
}

# Fonction pour compiler avec une taille de bloc spécifique
compile_with_block_size() {
    local bm=$1
    local bn=$2
    local bk=$3
    
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DBLOCK_SIZE_M=$bm -DBLOCK_SIZE_N=$bn -DBLOCK_SIZE_K=$bk" ..
    make -j$(sysctl -n hw.ncpu)
    cd ..
}

echo "=== Début du profilage comparatif ==="

# Nettoyer et créer le dossier build
rm -rf build
mkdir -p build

# Créer le fichier CSV pour les résultats
echo "Layout,Block_Size,Time_ms,Time_stddev,GFLOPS,GFLOPS_stddev,Page_Faults,Instructions,Cycles" > results.csv

# Tester chaque configuration
for layout in "${LAYOUTS[@]}"; do
    # D'abord tester sans blocking
    compile_with_block_size 0 0 0
    
    echo "=== Test sans blocking, layout $layout ==="
    > temps_execution.txt
    > performance_gflops.txt
    > page_faults.txt
    > instructions.txt
    > cycles.txt
    
    for ((run=1; run<=RUNS; run++)); do
        profile_run $MATRIX_SIZE $THREADS $layout $run "no_blocking"
        
        # Extraire les métriques
        grep "Temps d'exécution:" profile_${layout}_${MATRIX_SIZE}_${THREADS}_no_blocking_${run}.txt | awk '{print $3}' >> temps_execution.txt
        grep "Performance:" profile_${layout}_${MATRIX_SIZE}_${THREADS}_no_blocking_${run}.txt | awk '{print $2}' >> performance_gflops.txt
        grep "page faults" profile_${layout}_${MATRIX_SIZE}_${THREADS}_no_blocking_${run}.txt | head -n1 | awk '{print $1}' >> page_faults.txt
        grep "instructions retired" profile_${layout}_${MATRIX_SIZE}_${THREADS}_no_blocking_${run}.txt | awk '{print $1}' >> instructions.txt
        grep "cycles elapsed" profile_${layout}_${MATRIX_SIZE}_${THREADS}_no_blocking_${run}.txt | awk '{print $1}' >> cycles.txt
    done
    
    # Calculer les statistiques
    read time_mean time_stddev <<< $(calculate_stats < temps_execution.txt)
    read perf_mean perf_stddev <<< $(calculate_stats < performance_gflops.txt)
    page_faults=$(awk '{ sum += $1 } END { print sum/NR }' page_faults.txt)
    instructions=$(awk '{ sum += $1 } END { print sum/NR }' instructions.txt)
    cycles=$(awk '{ sum += $1 } END { print sum/NR }' cycles.txt)
    
    echo "$layout,no_blocking,$time_mean,$time_stddev,$perf_mean,$perf_stddev,$page_faults,$instructions,$cycles" >> results.csv
    
    # Ensuite tester chaque taille de bloc
    for block_size in "${BLOCK_SIZES[@]}"; do
        read bm bn bk <<< $block_size
        compile_with_block_size $bm $bn $bk
        
        echo "=== Test avec blocs ${bm}x${bn}x${bk}, layout $layout ==="
        > temps_execution.txt
        > performance_gflops.txt
        > page_faults.txt
        > instructions.txt
        > cycles.txt
        
        for ((run=1; run<=RUNS; run++)); do
            profile_run $MATRIX_SIZE $THREADS $layout $run "${bm}_${bn}_${bk}"
            
            # Extraire les métriques
            grep "Temps d'exécution:" profile_${layout}_${MATRIX_SIZE}_${THREADS}_${bm}_${bn}_${bk}_${run}.txt | awk '{print $3}' >> temps_execution.txt
            grep "Performance:" profile_${layout}_${MATRIX_SIZE}_${THREADS}_${bm}_${bn}_${bk}_${run}.txt | awk '{print $2}' >> performance_gflops.txt
            grep "page faults" profile_${layout}_${MATRIX_SIZE}_${THREADS}_${bm}_${bn}_${bk}_${run}.txt | head -n1 | awk '{print $1}' >> page_faults.txt
            grep "instructions retired" profile_${layout}_${MATRIX_SIZE}_${THREADS}_${bm}_${bn}_${bk}_${run}.txt | awk '{print $1}' >> instructions.txt
            grep "cycles elapsed" profile_${layout}_${MATRIX_SIZE}_${THREADS}_${bm}_${bn}_${bk}_${run}.txt | awk '{print $1}' >> cycles.txt
        done
        
        # Calculer les statistiques
        read time_mean time_stddev <<< $(calculate_stats < temps_execution.txt)
        read perf_mean perf_stddev <<< $(calculate_stats < performance_gflops.txt)
        page_faults=$(awk '{ sum += $1 } END { print sum/NR }' page_faults.txt)
        instructions=$(awk '{ sum += $1 } END { print sum/NR }' instructions.txt)
        cycles=$(awk '{ sum += $1 } END { print sum/NR }' cycles.txt)
        
        echo "$layout,${bm}x${bn}x${bk},$time_mean,$time_stddev,$perf_mean,$perf_stddev,$page_faults,$instructions,$cycles" >> results.csv
    done
done

echo "=== Résultats sauvegardés dans results.csv ==="
echo "=== Analyse des résultats ==="
echo "Format: Layout,Block_Size,Time(ms),Time_stddev,GFLOPS,GFLOPS_stddev,Page_Faults,Instructions,Cycles"
column -t -s ',' results.csv

# Nettoyage
rm -f temps_execution.txt performance_gflops.txt page_faults.txt instructions.txt cycles.txt
rm -f profile_*.txt

echo "=== Profilage terminé ===" 