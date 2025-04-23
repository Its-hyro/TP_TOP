\documentclass[a4paper,11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{siunitx}

\begin{document}

% Page de garde
\begin{titlepage}
  \centering
  {\LARGE\bfseries Optimisation parallèle du produit matriciel avec Kokkos\par}
  \vspace{2cm}
  {\Large DRIVET Dorian\par}
  \vfill
  Rapport de TP – UE Techniques d’optimisation parallèle\\
  Encadrant·e : Pr. X \\
  \vspace{1cm}
  {\large \today\par}
\end{titlepage}

\section{Introduction}

Ce rapport porte sur l’optimisation du produit matriciel dense.  Le principal défi réside dans le déséquilibre entre la puissance de calcul des processeurs et la bande passante mémoire. L'accès aux données en mémoire crée un goulot d'étranglement. Pour passer outre cette limitation, nous explorons deux stratégies complémentaires, l'optimisation du layout mémoire pour améliorer les patterns d'accès aux données, et l'implémentation de techniques de cache blocking pour maximiser la réutilisation des données dans les différents niveaux de cache.

Dans ce TP, nous partons d’une version naïve du code de multiplication de matrices, codée avec Kokkos. Ce framework offre une abstraction portable des tableaux multidimensionnels (\texttt{Kokkos::View}) et de la parallélisation (constructeurs \texttt{parallel\_for}). Le fichier de base initialise trois matrices en disposition \texttt{LayoutRight} et lance, pour chaque ligne, une boucle imbriquée classique sur les colonnes et la dimension commune.  

L’objectif sera d'étudier l’impact du choix de layout (row-major vs column-major via \texttt{LayoutRight} et \texttt{LayoutLeft}) sur la performance (mesurée en temps d’exécution et en GFLOP/s) et d'implémenter et paramétrer un algorithme de \emph{cache blocking} afin d’améliorer la réutilisation des lignes de cache et réduire le nombre de fautes de cache. Pour finir nous réaliserons une étude de scalabilité pour chaque version du code, afin de mettre en évidence les gains effectifs sur un MacBook Air M1 standard.

La suite du document décrit le matériel, la méthodologie de mesure, les optimisations appliquées, puis présente les résultats finaux et la conclusion.

\section{Matériel}

\begin{table}[h]
  \centering
  \caption{Caractéristiques matérielles}
  \begin{tabular}{@{}ll@{}}
    \toprule
    Élément      & Description            \\
    \midrule
    \multicolumn{2}{l}{\textbf{Matériel}}                \\
    Plateforme   & MacBook Air M1         \\
    CPU          & Apple M1 (8 cœurs) \\
    & 4 performance(3.2 GHz), 4 haute efficacité énergétique(2.0 GHz)     \\
    
    Mémoire      & 8 Go LPDDR4X           \\
    Cache L1 & 192 KB par cœur de performance,128 KB par cœur d'efficience \\
    Cache L2 & 12 MB partagé\\
    OS           & macOS 14 “Sonoma”      \\
    Compilateur  & Apple Clang 16         \\
    \bottomrule
  \end{tabular}
\end{table}

\section{Méthodologie}\label{sec:methodo}

\subsection{Implémentations du produit matriciel}

\subsubsection{Version naïve (\texttt{matrix\_product})}
La version de référence implémente le calcul $C \gets \beta\,C + \alpha\,A\times B$
dans la fonction \texttt{matrix\_product} par une triple boucle imbriquée, parallélisant uniquement la boucle externe sur l’indice de ligne 
Cette implémentation ne cherche pas à optimiser l’ordre des accès mémoire au sein des boucles \texttt{j} et \texttt{k}, et ne fait pas de blocage explicite.

\subsubsection{Version bloquée (\texttt{matrix\_product\_blocked})}
La version optimisée applique un cache blocking à deux niveaux, adapté à la hiérarchie du M1. On définit d’abord des blocs L2 de \(64\times64\) doubles (soit \(\approx32\) Ko) et, au sein de chacun, des sous-blocs L1 de \(32\times32\) doubles (\(\approx8\) Ko). La parallélisation Kokkos se fait au niveau des blocs L2 : l’ensemble des lignes de \(A\) est découpé en \(\lceil M/64\rceil\) morceaux, chacun traité par un thread. Pour un bloc L2 donné, on parcourt les colonnes par pas de 64 et la dimension \(K\) par pas de 64, puis on affine chaque partie en sous-blocs L1 de 32×32×32. Enfin, la boucle interne sur \(k\) est déroulée en groupes de 8 itérations pour maximiser l’utilisation du vecteur matériel. Cette organisation améliore à la fois la localité spatiale (accès contigus au sein des blocs) et la localité temporelle (réutilisation des données en cache), réduisant ainsi sensiblement le nombre de fautes de cache comparé à la version naïve.
\subsection{Protocole de mesure}

Nous avons bâti un protocole de mesure entièrement automatisé et reproductible afin de quantifier précisément l’impact de nos optimisations de cache et de parallélisation. La première étape a consisté à valider empiriquement la taille utile des caches sur une interface Linux équipée de l’outil \texttt{perf}. En surveillant les compteurs \texttt{cache-references} et \texttt{cache-misses} lors de l’exécution de la version élémentaire de notre multiplication de matrices, nous avons constaté qu’un cœur dispose d’environ 192 Ko de cache L1 et que le cache L2 atteint plus de 12 Mo. Forts de ces mesures, nous avons arrêté notre choix sur des sous–blocs de 32 × 32 éléments pour le L1 et de 64 × 64 pour le L2, de manière à maximiser la réutilisation locale des données avant toute éviction.

Ayant défini ces dimensions de blocs, la phase suivante a été de mesurer rigoureusement le coût d’exécution de la multiplication optimisée. La mesure du temps d’exécution repose sur l’enchaînement de barrières Kokkos avant et après l’appel à la routine de multiplication, couplées à des horloges haute résolution fournies par \texttt{std::chrono::high\_resolution\_clock}. La différence de temps, exprimée en millisecondes, correspond strictement à la section critique de l’algorithme, ce qui élimine toute dérive due à l’initialisation ou à la finalisation. Le nombre exact d’opérations, égal à \(2\,M\,N\,K\) (une multiplication et une addition par élément de la matrice résultat), permet de convertir ce temps en GFLOP/s via la formule  
\[
  \text{GFLOP/s}
  = \frac{2\,M\,N\,K}{T \times 10^{9}},
\]
où \(T\) est la durée en secondes.

Pour garantir la robustesse de nos résultats face aux fluctuations système (scheduler, effets thermiques, etc.), nous répétons chaque expérience cinq fois et en reportons la moyenne ainsi que l’écart-type. Cette redondance est suffisante pour lisser la variabilité tout en conservant un coût global de campagne de tests raisonnable.

L’ensemble des tests est piloté par deux scripts Python. Le premier, \texttt{benchmark.py}, itère automatiquement sur toutes les tailles de matrices (de \(2^0\) à \(2^{11}\)) et sur les configurations de threads (1, 2, 4, 8), recueille temps et performances, puis consolide les données dans un fichier CSV. Le second, \texttt{compare\_layouts.py}, exploite ce fichier pour générer les graphiques de performance, de temps et de speedup, facilitant ainsi la comparaison des différentes versions de code.

Enfin, pour étudier la scalabilité, nous comparons systématiquement le temps mono-thread \(T(1)\) au temps multi-threads \(T(p)\) et calculons le speedup :
\[
  S(p) \;=\;\frac{T(1)}{T(p)}.
\]
Idéalement linéaire (\(S(p)=p\)), ce speedup se heurtera dans la réalité à la fraction séquentielle de l’algorithme (loi d’Amdahl), au partage de la bande passante mémoire entre les threads et aux overheads liés aux synchronisations et aux conflits de cache.  


\section{Optimisation des layouts}

Dans cette section, nous nous intéressons à l’influence du parcours mémoire sur les performances de notre algorithme de multiplication de matrices. L’idée est que pour tirer le meilleur parti des caches, il faut faire coïncider les données “voisines” dans la mémoire avec l’ordre d’itération du cœur de calcul. Pour cela, nous avons mis en œuvre deux versions de nos matrices A, B et C en Kokkos , l’une instanciée avec LayoutLeft (correspondant à un stockage colonne-major, à la manière de Fortran) et l’autre avec LayoutRight (stockage ligne-major, à la manière de C/C++).
Sur le plan pratique, chaque matrice est déclarée par,
\begin{verbatim}
using view_left  = Kokkos::View<double**,Kokkos::LayoutLeft>;
using view_right = Kokkos::View<double**,Kokkos::LayoutRight>;
\end{verbatim}
Les données sont initialisées en parallèle, puis la boucle de multiplication s’écrit de façon identique dans les deux cas. Le seul changement réside dans l’ordre physique des éléments en mémoire.

Les courbes de performance (GFLOP/s) obtenues pour 1, 2, 4 et 8 threads (disponible dans la partie comparaison du github) montrent dans la plus part des cas un avantage pour LayoutRight \ref{fig:comp1},\ref{fig:comp2}. Pour les matrices de taille intermédiaire à grande (à partir de quelques centaines d’éléments par dimension), on observe un gain de l’ordre de $5\%$ à $15 \%$ en GFLOP/s, quel que soit le degré de parallélisme. Le graphique de speedup agrégé confirme en outre que cet avantage persiste à huit threads, avec un ratio de speedup LayoutRight/LayoutLeft légèrement supérieur à 1, même si la scalabilité globale reste limitée par la bande passante mémoire et les overheads de synchronisation.

\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{comparaison left right barre d'erreur/comparaison_8threads.png}
    \caption{Comparaison layout right et layout left sur 8 threads pour differentes tailles de matrice}
    \label{fig:comp1}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{comparaison_configs_threads.png}
    \caption{Comparaison Layout right/Layout Left en fonction du nombres de threads}
    \label{fig:comp2}
\end{figure}

Ces résultats s'expliquent car dans notre ordre d’itération (i–j–k), l’accès à A se fait sur la deuxième dimension k, et la mise à jour de C sur la deuxième dimension j. Tous deux sont donc parcourus de façon contiguë lorsque le stockage est ligne-major. À contrario, le LayoutLeft favorise l’accès contigu à B(k,j) en k. Cependant, dans notre code, la lecture de B est toujours entrecoupée par les accès à A et l’écriture dans C, ce qui réduit l’impact de cette localité. Le LayoutRight aligne mieux la majorité des accès en mémoire sur des segments contigus, diminuant ainsi les défauts de cache et maximisant la bande passante utile. En réglant le layout, sans modifier la logique ou la complexité algorithmique, on améliore de plusieurs pourcents la performance par thread et renforce la robustesse de la scalabilité face aux contraintes mémoire.



\section{Implémentation du cache blocking}

Dans notre démarche d'optimisation de la localité mémoire, nous avons implémenté une stratégie de cache blocking qui restructure l'accès aux données lors de la multiplication matricielle. Cette méthode permet de découper le calcul en sous-blocs adaptés à la hiérarchie de cache du processeur M1, permettant ainsi de maximiser la réutilisation des données en cache avant leur éviction.

\subsection{Principe et implémentation}

Notre implémentation initiale repose sur une structure hiérarchique à deux niveaux, définie dans la structure CacheBlockSizes. La multiplication est organisée en trois phases. Le découpage en macro-blocs (L2) de 64×64 éléments, la subdivision en sous-blocs (L1) de 32×32 éléments et pour finir le calcul optimisé au niveau des sous-blocs avec déroulage de boucle.

\subsection{Principe et illustration de l’algorithme}

Nous commençons par définir une petite structure qui fixe nos tailles de bloc a travers struct CacheBlockSizes. Le diagramme d’accès se traduit très simplement par six boucles imbriquées. Cette structure à deux niveaux garantit qu’une fenêtre de 64×64 éléments tient dans le cache L2, et que chaque sous-bloc de 32×32 est traité entièrement dans le cache L1 avant de passer au suivant. Le déroulage par paquets de huit permet d’exploiter pleinement les registres vectoriels NEON et l’Instruction Level Parallelism du M1.

\subsection{Parcours des versions et raffinements}
La version initiale du projet reposait sur une découpe directe de la matrice en blocs de 64×64, sans différencier L1 et L2. Cette première approche a permis de passer de 8 GFLOP/s (version naïve) à environ 10 GFLOP/s sur huit threads (\ref{fig:final1}), en s’appuyant sur le cache L2 de 12 Mo.
Nous avons ensuite fait du profiling de cache à travers (perf → compteurs cache-misses). Il a révélé un taux élevé d’échecs dans le cache L1. Les blocs de 64×64 étaient trop gros pour rester efficacement en L1 (192 Ko). Nous avons alors conservé la granularité L2 (64×64) mais réduit les sous-blocs L1 à 32×32. Cela a permis au cache L1 de saturé sans éviction prématurée et au cache L2 de toujours être exploité à pleine capacité.
Cette simple modification a porté la performance à 12 GFLOP/s sur huit threads. 
Enfin une version finale optimisée avec l’ajout d’un loop unrolling facteur 8 dans le micro-kernel a apporté un léger supplément de performance (5–$10 \%$) sur les plus grandes matrices et a amélioré la tenue du pipeline NEON (\ref{fig:final1}).

\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{comparaison_right_2/comparaison_8threads.png}
    \caption{Comparaison entre layout right de base (bleu), layout right cache blocking L1=L2=64 (vert), layout right cb (L1=32 ,L2=64) (orange) et layout right cb L1=32, L2=64 et op vecto (rouge)}
    \label{fig:final1}
\end{figure}

Sur le plan applicatif, le speedup passe de 1,5× à 2× selon la taille de la matrice(\ref{fig:final2}), avec un plateau à $\approx 12 GFLOP/s$ sur huit threads. L’impact est particulièrement marqué pour les tailles intermédiaires ($2^8$ - $2^{10}$), où le compromis entre coût d’initialisation des blocs et réutilisation mémoire est optimal.

\section{Résultats finaux}
\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{comparaison_right_2/comparaison_configs_threads.png}
    \caption{Comparaison entre les différentes configurations}
    \label{fig:final2}
\end{figure}

\section{Conclusion}
Synthétisez les gains et proposez des extensions (GPU, weak scaling).

\section*{Références}
\begin{itemize}
  \item C. Trott et al., \emph{Kokkos: Enabling Performance Portability}, 2020.
  \item P. Sanders et al., \emph{Optimizing Matrix Multiplication with Blocking}, 2018.
  \item Documentation Perf : \url{https://perf.wiki.kernel.org}.
\end{itemize}

\end{document}