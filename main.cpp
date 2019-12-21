#include <iostream>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <chrono>
#include "adaptive_clustering.h"
#include "error.h"

/**
 * @file main.cpp
 */

/**
 * @struct Params
 * A structure containing parameters read from command-line.
 */
struct Params {
    string       inputFilename = "../dataset/dim032.csv"; /**< The path for the input CSV file. */
    string       outputFilename; /**< The path for the output file. */
    long         k_max = 10; /**< The maximum number of cluster to try for the K-Means algorithm. */
    double       elbowThreshold = 0.25; /**< The error tolerance for the selected metric to evaluate the elbow in K-means algorithm. */
    double       percentageIncircle = 0.9; /**< The percentage of points in a cluster to be considered as inliers. */
    double       percentageSubspaces = 0.8; /**< The percentage of subspace in which a point must be evaluated as outlier to be
 *                                          selected as general outlier. */
    int          version = 1; /**< 0 for standard K-Means and 1 for K-Means++. */
};

chrono::high_resolution_clock::time_point t1, t2;

/**
 * This function saves the actual time into global variable t1.
 */
void StartTheClock();
/**
 * This function saves the actual time into global variable t2
 * and it computes the difference between t1 and t2.
 * @return the difference between t1 and t2
 */
double StopTheClock();
/**
 * Print the needed parameters in order to run the script
 * @param cmd The name of the script.
 */
void usage(char* cmd);
/**
 * This function handles the arguments passed by command line
 * @param [in] argc it is the count of command line arguments.
 * @param [in] argv it contains command line arguments.
 * @param [in,out] params structure to save arguments.
 * @return 0 if the command line arguments are ok, otherwise -5.
 */
int parseCommandLine(int argc, char **argv, Params &params);
/**
 * This function prints the parameters used for the run.
 * @param [in] params the structure with parameters.
 */
void printUsedParameters(Params params, bool outputOnFile);

int main(int argc, char **argv) {
    cout << fixed;
    cout << setprecision(5);

    int N_DIMS; //Number of dimensions
    int N_DATA; //Number of observations
    double elapsed;
    bool outputOnFile = false; //Flag for csv output generation
    fstream fout;
    Params params;
    int programStatus = -12;

    programStatus = parseCommandLine(argc, argv, params);
    if (programStatus) {
        exit(programStatus);
    }

    outputOnFile = !(params.outputFilename.empty());

    printUsedParameters(params, outputOnFile);

    double **data = NULL, *data_storage = NULL, **pearson = NULL, *pearson_storage = NULL, **corr = NULL,
    **newspace = NULL, *newspace_storage = NULL, *cs_storage = NULL, **csi = NULL, ***cs = NULL,
    **combine = NULL, *combine_storage = NULL;
    bool **incircle = NULL, *incircle_storage = NULL;
    int corr_vars, uncorr_vars, tot_outliers, *uncorr = NULL;

    /***
     * The dataset is read and loaded in "data" from the input CSV file.
     * Then each dimension in the dataset is centered around the mean value.
    ***/
    StartTheClock();

    programStatus = getDatasetDims(params.inputFilename, N_DIMS, N_DATA);
    if (programStatus) {
        return programStatus;
    }
    data_storage = (double *) calloc(N_DIMS * N_DATA, sizeof(double));
    if (!data_storage) {
        return MemoryError(__FUNCTION__);
    }
    data = (double **) calloc(N_DIMS, sizeof(double *));
    if (!data) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    for (int i = 0; i < N_DIMS; ++i) {
        data[i] = &data_storage[i * N_DATA];
    }

    programStatus = loadData(params.inputFilename, data, N_DIMS);
    if (programStatus) {
        goto ON_EXIT;
    }

    if (!outputOnFile) {
        cout << "Dataset Loaded" << endl;
    }

    programStatus = Standardize_dataset(data, N_DIMS, N_DATA);
    if (programStatus) {
        goto ON_EXIT;
    }

    if (!outputOnFile) {
        cout << "Dataset standardized" << endl;
    }

    /***
     * The Pearson Coefficient is computed on each pair of dimensions in order
     * to partition the dimensions in CORR and UNCORR sets (CORR U UNCORR = DIMS).
     * Partitions are computed with the evaluation of the Pearson coefficient
     * row-wise sum for each dimension. If the value is greater than or equal to
     * zero, then the dimension is stored in "CORR", otherwise the index of the
     * dimension is stored in "UNCORR".
    ***/
    pearson_storage = (double *) calloc(N_DIMS * N_DIMS, sizeof(double));
    if (!pearson_storage) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    pearson = (double **) calloc(N_DIMS, sizeof(double *));
    if (!pearson) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    for (int i = 0; i < N_DIMS; ++i) {
        pearson[i] = &pearson_storage[i * N_DIMS];
    }

    programStatus = computePearsonMatrix(pearson, data, N_DATA, N_DIMS);
    if (programStatus) {
        goto ON_EXIT;
    }

    if (!outputOnFile) {
        cout << "Pearson Correlation Coefficient computed" << endl;
    }

    programStatus = computeCorrUncorrCardinality(pearson, N_DIMS, corr_vars, uncorr_vars);
    if (programStatus) {
        goto ON_EXIT;
    }
    if (corr_vars < 2) {
        programStatus = LessCorrVariablesError(__FUNCTION__);
        goto ON_EXIT;
    }
    if (uncorr_vars == 0) {
        programStatus = NoUncorrVariablesError(__FUNCTION__);
        goto ON_EXIT;
    }

    corr = (double **) calloc(corr_vars, sizeof(double *));
    if (!corr) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    uncorr = (int *) (calloc(uncorr_vars, sizeof(int)));
    if (!uncorr) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }

    corr_vars = 0, uncorr_vars = 0;
    for (int i = 0; i < N_DIMS; ++i) {
        if (isCorrDimension(N_DIMS, i, pearson)) {
            corr[corr_vars] = data[i];
            corr_vars++;
        } else {
            uncorr[uncorr_vars] = i;
            uncorr_vars++;
        }
    }

    if(pearson_storage)
        free(pearson_storage), pearson_storage = NULL;

    free(pearson), pearson = NULL;

    if (!outputOnFile) {
        cout << "Correlated dimensions: " << corr_vars << ", " << "Uncorrelated dimensions: " << uncorr_vars << endl;
    }

    /***
     * The Principal Component Analysis is computed on CORR set and the result
     * is stored in "newspace".
     * Then the candidate subspaces are created with the combination of newspace
     * and each dimension in the UNCORR set; this combination is stored in "combine"
     * in order to pass this structure to the PCA function. The PCA result is stored
     * in "cs".
    ***/
    newspace_storage = (double *) calloc(N_DATA * 2, sizeof(double));
    if (!newspace_storage) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    newspace = (double **) calloc(2, sizeof(double *));
    if (!newspace) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    for (int i = 0; i < 2; ++i) {
        newspace[i] = &newspace_storage[i*N_DATA];
    }

    programStatus = PCA_transform(corr, corr_vars, N_DATA, newspace);
    if (programStatus) {
        goto ON_EXIT;
    }

    free(corr), corr = NULL;

    if (!outputOnFile) {
        cout << "PCA computed on CORR subspace" << endl;
    }

    cs_storage = (double *) calloc(N_DATA * 2 * uncorr_vars, sizeof(double));
    if (!cs_storage) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    csi = (double **) calloc(2 * uncorr_vars, sizeof(double *));
    if (!csi) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    cs = (double ***) calloc(uncorr_vars, sizeof(double **));
    if (!cs) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    for (int i = 0; i < 2 * uncorr_vars; ++i) {
        csi[i] = &cs_storage[i * N_DATA];
    }
    for (int i = 0; i < uncorr_vars; ++i) {
        cs[i] = &csi[i * 2];
    }

    combine_storage = (double *) calloc(N_DATA * 3, sizeof(double));
    if (!combine_storage) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    combine = (double **) calloc(3, sizeof(double *));
    if (!combine) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    for (int l = 0; l < 3; ++l) {
        combine[l] = &combine_storage[l * N_DATA];
    }

    memcpy(combine[0], newspace[0], N_DATA * sizeof(double));
    memcpy(combine[1], newspace[1], N_DATA * sizeof(double));


    free(newspace_storage), newspace_storage = NULL;
    free(newspace), newspace = NULL;

    for (int i = 0; i < uncorr_vars; ++i) {
        memcpy(combine[2], data[uncorr[i]], N_DATA * sizeof(double));
        programStatus = PCA_transform(combine, 3, N_DATA, cs[i]);
        if (programStatus) {
            goto ON_EXIT;
        }

        if (!outputOnFile) {
            cout << "PCA computed on PC1_CORR, PC2_CORR and " << i + 1 << "-th dimension of UNCORR" << endl;
        }
    }

    free(uncorr), uncorr = NULL;
    free(combine_storage), combine_storage = NULL;
    free(combine), combine = NULL;

    /***
     * The K-Means with the elbow criterion is executed for each candidate subspace.
     * The structure "incircle" keeps record of the inliers for each candidate subspace.
     * Then the outlier identification process starts:
     * 1) For each cluster, the cluster size is computed to define a threshold on the
     *      number of inliers;
     * 2) Iteratively, the cluster size (without the inliers excluded in the previous
     *      iteration) and the distance between the centroids and the points in the cluster
     *      are computed.
     * 3) The radius of the circle for inliers evaluation is computed and a new set of
     *      inliers is discarded based on that radius.
     * The outlier identification process ends with the general point-wise evaluation: if a data is
     * an outlier for a chosen percentage of subspaces then it is marked as general outlier.
    ***/
    incircle_storage = (bool *) calloc(uncorr_vars * N_DATA, sizeof(bool));
    if (!incircle_storage) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    incircle = (bool **) calloc(uncorr_vars, sizeof(bool *));
    if (!incircle) {
        programStatus = MemoryError(__FUNCTION__);
        goto ON_EXIT;
    }
    for (int i = 0; i < uncorr_vars; ++i) {
        incircle[i] = &incircle_storage[i * N_DATA];
    }

    for (int i = 0; i < uncorr_vars; ++i) {
        cluster_report rep;
        if (!outputOnFile) {
            cout << "Candidate Subspace " << i + 1 << ": ";
        }
        //Clustering through Elbow criterion on i-th candidate subspace
        rep = run_K_means(cs[i], N_DATA, params.k_max, params.elbowThreshold, params.version);
        for (int j = 0; j < rep.k; ++j) {
            int k = 0, previous_k = 0;
            int cls_size = cluster_size(rep, j, N_DATA);
            while (k < params.percentageIncircle * cls_size) {
                previous_k = k;
                double dist = 0.0;
                int actual_cluster_size = computeActualClusterInfo(rep, N_DATA, j, incircle[i], cs[i], dist);
                double dist_mean = dist / (double) actual_cluster_size;
                k += computeInliers(rep, N_DATA, j, incircle[i], cs[i], dist_mean);
                //Stopping criterion
                if (k == previous_k) {
                    break;
                }
            }
        }
        if (!outputOnFile) {
            cout << "CENTROIDS" << endl;
            rep.centroids.print();
        }
    }

    if (!outputOnFile) {
        cout << "Outliers Identification Process: \n";

        tot_outliers = 0;
        for (int i = 0; i < N_DATA; ++i) {
            int occurrence = countOutliers(incircle, uncorr_vars, i);
            if (occurrence >= std::round(uncorr_vars * params.percentageSubspaces)) {
                tot_outliers++;
                cout << i << ") ";
                for (int l = 0; l < N_DIMS; ++l) {
                    //cout << cs[0][l][i] << " ";
                    cout << data[l][i] << " ";
                }
                cout << "(" << occurrence << ")\n";
            }
        }

        cout << "TOTAL NUMBER OF OUTLIERS: " << tot_outliers << endl;
    }

    // Output file generation for metrics evaluation
    if (outputOnFile) {
        fout.open(params.outputFilename, ios::out | ios::trunc);
        for (int i = 0; i < N_DATA; ++i) {
            int occurrence = countOutliers(incircle, uncorr_vars, i);
            if (occurrence >= std::round(uncorr_vars * params.percentageSubspaces)) {
                fout << i << ",";
            }
        }
        fout.close();
    }

    elapsed = StopTheClock();
    if (outputOnFile) {
        cout << "Elapsed time (seconds): " << elapsed << endl;
    }
    programStatus = 0;

    ON_EXIT:


    if (data != NULL)
        free(data), data = NULL;

    if (data_storage != NULL)
        free(data_storage), data_storage = NULL;

    if (pearson_storage != NULL)
        free(pearson_storage), pearson_storage = NULL;

    if (pearson != NULL)
        free(pearson), pearson = NULL;

    if (newspace != NULL){
        for (int i = 0; i < 2; ++i)
            if(newspace[i])
                free(newspace[i]), newspace[i] = NULL;

        free(newspace), newspace = NULL;
    }

    if (newspace_storage != NULL)
        free(newspace_storage), newspace_storage = NULL;

    if (corr != NULL)
        free(corr), corr = NULL;

    if (uncorr != NULL)
        free(uncorr), uncorr = NULL;

    if (combine_storage != NULL)
           free(combine_storage), combine_storage = NULL;

    if (combine != NULL)
        free(combine), combine = NULL;

    if (incircle_storage != NULL)
        free(incircle_storage), incircle_storage = NULL;

    if (incircle != NULL)
        free(incircle), incircle = NULL;

    if (csi != NULL)
        free(csi), csi = NULL;

    if (cs_storage != NULL)
        free(cs_storage), cs_storage = NULL;

    if (cs != NULL)
       free(cs), cs = NULL;

    return programStatus;
}

void StartTheClock() {
    t1 = chrono::high_resolution_clock::now();
}

double StopTheClock() {
    t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    return time_span.count();
}

void usage(char* cmd) {
    cerr
            << "Usage: " << cmd << endl
            << "-of         output filename, if specified a file with this name containing the final result is written" << endl
            << "-k          max number of clusters to try in elbow criterion" << endl
            << "-et         threshold for the selection of optimal number of clusters in Elbow method" << endl
            << "-pi         percentage of points in a cluster to be evaluated as inlier" << endl
            << "-ps         percentage of subspaces in which a point must be outlier to be evaluated as general outlier" << endl
            << "-if         input filename" << endl
            << "-v          K-Means version: 0 for standard K-Means and 1 for K-Means++" << endl;
}

int parseCommandLine(int argc, char **argv, Params &params) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-of") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing filename for simulation output." << endl;
                return ArgumentsError(__FUNCTION__);
            }
            params.outputFilename = string(argv[i]);
        } else if (strcmp(argv[i], "-k") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing max number of clusters for Elbow method.\n";
                return ArgumentsError(__FUNCTION__);
            }
            params.k_max = stol(argv[i]);
        } else if (strcmp(argv[i], "-et") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing threshold for Elbow method.\n";
                return ArgumentsError(__FUNCTION__);
            }
            params.elbowThreshold = stof(argv[i]);
        } else if (strcmp(argv[i], "-pi") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of percentage of inlier points.\n";
                return ArgumentsError(__FUNCTION__);
            }
            params.percentageIncircle = stof(argv[i]);
        } else if (strcmp(argv[i], "-ps") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of percentage of subspace in which an outlier must be.\n";
                return ArgumentsError(__FUNCTION__);
            }
            params.percentageSubspaces = stof(argv[i]);
        } else if (strcmp(argv[i], "-if") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing input file name.\n";
                return ArgumentsError(__FUNCTION__);
            }
            params.inputFilename = string(argv[i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing K-Means version.\n";
                return ArgumentsError(__FUNCTION__);
            }
            params.inputFilename = string(argv[i]);
        } else {
            usage(argv[0]);
            return ArgumentsError(__FUNCTION__);
        }
    }
    return 0;
}

void printUsedParameters(Params params, bool outputOnFile) {
    cout << endl << "PARAMETERS:" << endl;
    cout << "input file = " << params.inputFilename << endl;
    if (outputOnFile) {
        cout << "output file = " << params.outputFilename << endl;
    }
    cout << "percentage in circle = " << params.percentageIncircle << endl;
    cout << "elbow threshold = " << params.elbowThreshold << endl;
    cout << "percentage subspaces = " << params.percentageSubspaces << endl;
    cout << "K-Means Version = " << params.version << endl;
    cout << "k_max = " << params.k_max << endl << endl;
}
