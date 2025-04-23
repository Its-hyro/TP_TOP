#!/bin/bash

# Configuration
MATRIX_SIZE=1024
THREADS=8
LAYOUT="right"

# Compiler en mode debug pour avoir les symboles
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(sysctl -n hw.ncpu)

# Lancer Instruments avec le template CPU Counters
xcrun xctrace record --template "CPU Counters" --launch -- ./src/top.matrix_product $MATRIX_SIZE $MATRIX_SIZE $MATRIX_SIZE $LAYOUT

echo "Le profilage est terminé. Ouvrez le fichier .trace dans Instruments.app pour voir les résultats." 