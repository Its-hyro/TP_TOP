#!/bin/bash

# Fonction pour compiler avec une configuration spécifique
compile_version() {
    local version=$1
    echo "Compilation de la version $version..."
    sed -i '' "s/static constexpr bool USE_HIERARCHICAL = .*;/static constexpr bool USE_HIERARCHICAL = $version;/" src/main.cpp
    rm -rf build
    cmake -B build -S .
    cmake --build build
}

# Fonction pour exécuter les tests avec dtrace
run_perf_test() {
    local version=$1
    local size=$2
    local threads=$3
    
    echo "Test de la version $version avec taille=$size et threads=$threads"
    
    # Configuration OpenMP
    export OMP_NUM_THREADS=$threads
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores
    
    # Création du script DTrace temporaire
    cat > dtrace_script.d << 'EOF'
pid$target:::entry
{
    self->start = timestamp;
}

pid$target:::return
/self->start/
{
    @time[probefunc] = avg(timestamp - self->start);
    self->start = 0;
}

profile:::profile-997
/pid == $target/
{
    @[cpu] = count();
}
EOF

    # Exécution avec dtrace
    sudo dtrace -s dtrace_script.d -c "./build/src/top.matrix_product $size $size $size right" > "perf_${version}_${size}_${threads}.txt" 2>&1
    
    # Nettoyage
    rm dtrace_script.d
}

# Tailles à tester
sizes=(256 512 1024)
threads=(1 2 4 8)

# Test de la version hiérarchique (L1=32, L2=64)
compile_version "true"
for size in "${sizes[@]}"; do
    for t in "${threads[@]}"; do
        run_perf_test "hierarchical" $size $t
    done
done

# Test de la version simple (64x64x64)
compile_version "false"
for size in "${sizes[@]}"; do
    for t in "${threads[@]}"; do
        run_perf_test "simple" $size $t
    done
done

# Analyse des résultats
echo "Analyse des résultats..."
for size in "${sizes[@]}"; do
    for t in "${threads[@]}"; do
        echo "=== Comparaison pour taille=$size threads=$t ==="
        echo "Version hiérarchique :"
        cat "perf_hierarchical_${size}_${t}.txt"
        echo "Version simple :"
        cat "perf_simple_${size}_${t}.txt"
        echo "================================================"
    done
done 