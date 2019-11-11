#include <iostream>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include "adaptive_clustering.h"

/**
 * @file main.cpp
 */

/**
 * @struct Params
 * A structure containing parameters read from command-line.
 */
struct Params {
    string       inputFilename; /**< The path for the input CSV file. */
    string       outputFilename; /**< The path for the output file. */
    long         k_max; /**< The maximum number of cluster to try for the K-Means algorithm. */
    double       elbowThreshold; /**< The error tolerance for the selected metric to evaluate the elbow in K-means algorithm. */
    double       percentageIncircle; /**< The percentage of points in a cluster to be considered as inliers. */
    double       percentageSubspaces; /**< The percentage of subspace in which a point must be evaluated as outlier to be
 *                                          selected as general outlier. */
};

/**
 * Print the needed parameters in order to run the script
 * @param cmd The name of the script.
 */
void usage(char* cmd);

int main(int argc, char **argv) {
    cout << fixed;
    cout << setprecision(5);

    int N_DIMS; //Number of dimensions
    int N_DATA; //Number of observations
    long k_max = 10; // max number of clusters to try in elbow criterion
    double elbowThreshold = 0.25; // threshold for the selection of optimal number of clusters in Elbow method
    double percentageIncircle = 0.9; // percentage of points in a cluster to be evaluated as inlier
    double percentageSubspaces = 0.8; // percentage of subspaces in which a point must be outlier to be evaluated as general outlier
    string inFile = "../dataset/Iris.csv";
    //string inFile = "../dataset/HTRU_2.csv";
    //string inFile = "../dataset/dim032.csv";
    //string inFile = "../dataset/Absenteeism_at_work.csv";
    string outFile;
    //outFile = "../plot/iris/";
    bool outputOnFile = false; //Flag for csv output generation
    Params params;

    /*** Parse Command-Line Parameters ***/
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-of") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing filename for simulation output." << endl;
                return -1;
            }
            outFile = string(argv[i]);
        } else if (strcmp(argv[i], "-k") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing max number of clusters for Elbow method.\n";
                return -1;
            }
            k_max = stol(argv[i]);
        } else if (strcmp(argv[i], "-et") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing threshold for Elbow method.\n";
                return -1;
            }
            elbowThreshold = stof(argv[i]);
        } else if (strcmp(argv[i], "-pi") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of percentage of inlier points.\n";
                return -1;
            }
            percentageIncircle = stof(argv[i]);
        } else if (strcmp(argv[i], "-pspace") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing number of percentage of subspace in which an outlier must be.\n";
                return -1;
            }
            percentageSubspaces = stof(argv[i]);
        } else if (strcmp(argv[i], "-if") == 0) {
            i++;
            if (i >= argc) {
                cerr << "Missing input file name.\n";
                return -1;
            }
            inFile = string(argv[i]);
        } else {
            usage(argv[0]);
            return -1;
        }
    }

    /*** Assign parameters read from command line ***/
    params.outputFilename = outFile;
    params.elbowThreshold = elbowThreshold;
    params.k_max = k_max;
    params.percentageIncircle = percentageIncircle;
    params.percentageSubspaces = percentageSubspaces;
    params.inputFilename = inFile;

    outputOnFile = params.outputFilename.size() > 0;

    cout << endl << "PARAMETERS:" << endl;
    cout << "input file= " << params.inputFilename << endl;
    if (outputOnFile) {
        cout << "output file= " << params.outputFilename << endl;
    }
    cout << "percentage in circle = " << params.percentageIncircle << endl;
    cout << "elbow threshold = " << params.elbowThreshold << endl;
    cout << "percentage subspaces = " << params.percentageSubspaces << endl;
    cout << "k_max = " << params.k_max << endl << endl;

    /***
     * The dataset in the input CSV file is read and loaded in "data".
     * Then each dimension in the dataset is centered around the mean value.
    ***/
    double **data, *data_storage;
    getDatasetDims(inFile, &N_DIMS, &N_DATA);
    data_storage = (double *) malloc(N_DIMS * N_DATA * sizeof(double));
    if (!data_storage) {
        cerr << "Malloc error on data_storage" << endl;
        exit(-1);
    }
    data = (double **) malloc(N_DIMS * sizeof(double *));
    if (!data) {
        cerr << "Malloc error on data" << endl;
        exit(-1);
    }
    for (int i = 0; i < N_DIMS; ++i) {
        data[i] = &data_storage[i * N_DATA];
    }

    if (loadData(inFile, data, N_DIMS)) {
        cerr << "Error on loading dataset" << endl;
        exit(-1);
    }

    cout << "Dataset Loaded" << endl;
    auto start = chrono::steady_clock::now();

    Standardize_dataset(data, N_DIMS, N_DATA);

    cout << "Dataset standardized" << endl;

    /***
     * The Pearson Coefficient is computed on each dimensions pair in order
     * to partition the dimensions in CORR and UNCORR sets (CORR U UNCORR = DIMS).
     * The partitions are computed with the evaluation of the Pearson coefficient
     * row-wise sum for each dimension. If the value is greater than or equal to
     * zero, then the dimension is saved in "corr", otherwise the index of the
     * dimension is saved in "uncorr".
    ***/
    double **pearson, *pearson_storage;
    pearson_storage = (double *) malloc(N_DIMS * N_DIMS * sizeof(double));
    if (!pearson_storage) {
        cerr << "Malloc error on pearson_storage" << endl;
        exit(-1);
    }
    pearson = (double **) malloc(N_DIMS * sizeof(double *));
    if (!pearson) {
        cerr << "Malloc error on pearson" << endl;
        exit(-1);
    }
    for (int i = 0; i < N_DIMS; ++i) {
        pearson[i] = &pearson_storage[i * N_DIMS];
    }

    for (int i = 0; i < N_DIMS; ++i) {
        pearson[i][i] = 1;
        for (int j = i+1; j < N_DIMS; ++j) {
            double value = PearsonCoefficient(data[i], data[j], N_DATA);
            pearson[i][j] = value;
            pearson[j][i] = value;
        }
    }

    cout << "Pearson Correlation Coefficient computed" << endl;

    int corr_vars = 0, uncorr_vars = 0;
    for (int i = 0; i < N_DIMS; ++i) {
        double overall = 0.0;
        for (int j = 0; j < N_DIMS; ++j) {
            if (i != j) {
                overall += pearson[i][j];
            }
        }
        if ((overall/N_DIMS) >= 0) {
            corr_vars++;
        } else {
            uncorr_vars++;
        }
    }

    if (corr_vars < 2) {
        cout << "Error: Correlated dimensions must be more than 1 in order to apply PCA!" << endl;
        exit(-1);
    }
    if (uncorr_vars == 0) {
        cout << "Error: There are no candidate subspaces!" << endl;
        exit(-1);
    }

    double **corr;
    int *uncorr;
    corr = (double **) malloc(corr_vars * sizeof(double *));
    if (!corr) {
        cerr << "Malloc error on corr" << endl;
        exit(-1);
    }
    uncorr = (int *) (malloc(uncorr_vars * sizeof(int)));
    if (!uncorr) {
        cerr << "Malloc error on corr or uncorr" << endl;
        exit(-1);
    }

    corr_vars = 0, uncorr_vars = 0;
    for (int i = 0; i < N_DIMS; ++i) {
        double overall = 0.0;
        for (int j = 0; j < N_DIMS; ++j) {
            if (i != j) {
                overall += pearson[i][j];
            }
        }
        if ((overall/N_DIMS) >= 0) {
            corr[corr_vars] = data[i];
            corr_vars++;
        } else {
            uncorr[uncorr_vars] = i;
            uncorr_vars++;
        }
    }

    free(pearson_storage);
    free(pearson);
    cout << "Correlated dimensions: " << corr_vars << ", " << "Uncorrelated dimensions: " << uncorr_vars << endl;

    /***
     * The Principal Component Analysis is computed on CORR set and the result
     * is saved in "newspace".
     * Then the candidate subspaces are created with the combination of newspace
     * and each dimension in UNCORR set; this combination is saved in "combine"
     * in order to pass this structure to the PCA function. This result is saved
     * in "cs".
    ***/
    double **newspace, *newspace_storage;
    newspace_storage = (double *) malloc(N_DATA * 2 * sizeof(double));
    if (!newspace_storage) {
        cerr << "Malloc error on newspace_storage" << endl;
        exit(-1);
    }
    newspace = (double **) malloc(2 * sizeof(double *));
    if (!newspace) {
        cerr << "Malloc error on newspace" << endl;
        exit(-1);
    }
    for (int i = 0; i < 2; ++i) {
        newspace[i] = &newspace_storage[i*N_DATA];
    }

    PCA_transform(corr, corr_vars, N_DATA, newspace);

    free(corr);
    cout << "PCA computed on CORR subspace" << endl;

    double *cs_storage, **csi, ***cs;
    cs_storage = (double *) malloc(N_DATA * 2 * uncorr_vars * sizeof(double));
    if (!cs_storage) {
        cerr << "Malloc error on cs_storage" << endl;
        exit(-1);
    }
    csi = (double **) malloc(2 * uncorr_vars * sizeof(double *));
    if (!csi) {
        cerr << "Malloc error on csi" << endl;
        exit(-1);
    }
    cs = (double ***) malloc(uncorr_vars * sizeof(double **));
    if (!cs) {
        cerr << "Malloc error on cs" << endl;
        exit(-1);
    }
    for (int i = 0; i < 2 * uncorr_vars; ++i) {
        csi[i] = &cs_storage[i * N_DATA];
    }
    for (int i = 0; i < uncorr_vars; ++i) {
        cs[i] = &csi[i * 2];
    }

    double **combine, *combine_storage;
    combine_storage = (double *) malloc(N_DATA * 3 * sizeof(double));
    if (!combine_storage) {
        cerr << "Malloc error on combine_storage" << endl;
        exit(-1);
    }
    combine = (double **) malloc(3 * sizeof(double *));
    if (!combine) {
        cerr << "Malloc error on combine" << endl;
        exit(-1);
    }
    for (int l = 0; l < 3; ++l) {
        combine[l] = &combine_storage[l * N_DATA];
    }


    memcpy(combine[0], newspace[0], N_DATA * sizeof(double));
    memcpy(combine[1], newspace[1], N_DATA * sizeof(double));

    free(newspace_storage);
    free(newspace);

    for (int i = 0; i < uncorr_vars; ++i) {
        memcpy(combine[2], data[uncorr[i]], N_DATA * sizeof(double));
        PCA_transform(combine, 3, N_DATA, cs[i]);
        cout << "PCA computed on PC1_CORR, PC2_CORR and " << i+1 << "-th dimension of UNCORR" << endl;
    }

    free(uncorr);
    free(combine_storage);
    free(combine);

    /***
     * The K-Means with the elbow criterion is executed for each candidate subspace.
     * The structure "incircle" keep records of the inliers for each candidate subspace.
     * Then the outlier identification process starts:
     * 1) For each cluster, the cluster size is computed to define a threshold on the
     *      number of inliers;
     * 2) Iteratively, the distance between the centroids and the points in the cluster
     *      and the cluster size (without the inliers excluded in the previous iteration)
     *      are computed.
     * 3) The radius of the circle for inliers evaluation is computed and a new set of
     *      inliers is discarded based on that radius.
     * The outlier identification process ends with the general evaluation: if a data is
     * an outlier for a chosen percentage of subspaces then it is marked as general outlier.
    ***/
    bool **incircle, *incircle_storage;
    incircle_storage = (bool *) calloc(uncorr_vars * N_DATA, sizeof(bool));
    if (!incircle_storage) {
        cerr << "Malloc error on incircle_storage" << endl;
        exit(-1);
    }
    incircle = (bool **) malloc(uncorr_vars * sizeof(bool *));
    if (!incircle) {
        cerr << "Malloc error on incircle" << endl;
        exit(-1);
    }
    for (int i = 0; i < uncorr_vars; ++i) {
        incircle[i] = &incircle_storage[i * N_DATA];
    }

    for (int i = 0; i < uncorr_vars; ++i) {
        cluster_report rep;
        cout << "Candidate Subspace " << i+1 << ": ";
        rep = run_K_means(cs[i], N_DATA, params.k_max, params.elbowThreshold); //Clustering through Elbow criterion on i-th candidate subspace
        for (int j = 0; j < rep.k; ++j) {
            int k = 0, previous_k = 0;
            int cls_size = cluster_size(rep, j, N_DATA);
            while (k < params.percentageIncircle * cls_size) {
                double dist = 0.0;
                int actual_cluster_size = 0;
                for (int l = 0; l < N_DATA; ++l) {
                    if (rep.cidx[l] == j && !(incircle[i][l])) {
                        dist += L2distance(rep.centroids(0,j), rep.centroids(1,j), cs[i][0][l], cs[i][1][l]);
                        actual_cluster_size++;
                    }
                }
                double dist_mean = dist / actual_cluster_size;
                for (int l = 0; l < N_DATA; ++l) {
                    if (rep.cidx[l] == j && !(incircle[i][l])) {
                        if (L2distance(rep.centroids(0,j), rep.centroids(1,j), cs[i][0][l], cs[i][1][l]) <= dist_mean) {
                            incircle[i][l] = true;
                            k++;
                        }
                    }
                }
                //Stopping criterion when the data threshold is greater than remaining n_points in a cluster
                if (k == previous_k) {
                    break;
                } else {
                    previous_k = k;
                }
            }
        }
        if (outputOnFile) {
            string fileoutname = "dataout";
            string num = to_string(i);
            string concat = fileoutname + num + ".csv";
            csv_out_info(cs[i], N_DATA, concat, outFile, incircle[i], rep);
        }
    }

    cout << "Outliers Identification Process: \n";

    int tot_outliers = 0;
    for (int i = 0; i < N_DATA; ++i) {
        int occurrence = 0;
        for (int j = 0; j < uncorr_vars; ++j) {
            if (!incircle[j][i]) {
                occurrence++;
            }
        }

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

    free(incircle_storage);
    free(incircle);
    free(cs_storage);
    free(csi);
    free(cs);
    free(data_storage);
    free(data);

    cout << "TOTAL NUMBER OF OUTLIERS: " << tot_outliers << endl;

    auto end = chrono::steady_clock::now();

    cout << "Elapsed time in milliseconds : "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    return 0;
}

void usage(char* cmd)
{
    cerr
            << "Usage: " << cmd << "\n"
            << "-of         output filename, if specified a file with this name containing the final result is written\n"
            << "-k          max number of clusters to try in elbow criterion\n"
            << "-et         threshold for the selection of optimal number of clusters in Elbow method\n"
            << "-pi         percentage of points in a cluster to be evaluated as inlier\n"
            << "-pspace     percentage of subspaces in which a point must be outlier to be evaluated as general outlier\n"
            << "-if         input filename\n";
}