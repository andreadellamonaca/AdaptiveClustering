#include "adaptive_clustering.h"

/**
 * The algorithm calculates the number of rows and columns in a CSV file and saves the information into dim and data.
 * @param [in] fname the path referred to a CSV file.
 * @param [in,out] dim the number of columns in the CSV file.
 * @param [in,out] data the number of rows in the CSV file.
 */
void getDatasetDims(string fname, int *dim, int *data) {
    int cols = 0;
    int rows = 0;
    ifstream file(fname);
    string line;
    int first = 1;

    while (getline(file, line)) {
        if (first) {
            istringstream iss(line);
            string result;
            while (getline(iss, result, ','))
            {
                cols++;
            }
            first = 0;
        }
        rows++;
    }
    *dim = cols;
    *data = rows;
    cout << "Dataset: #DATA = " << *data << " , #DIMENSIONS = " << *dim << endl;
    file.close();
}

/**
 * The algorithm loads a CSV file into a matrix of double.
 * @param [in] fname the number of columns in the CSV file.
 * @param [in,out] array a matrix [n_data, n_dims], in which the CSV file is loaded.
 * @param [in] n_dims the number of columns in the CSV file.
 * @return 0 if the dataset is loaded correctly, otherwise -1.
 */
int loadData(string fname, double **array, int n_dims) {
    ifstream inputFile(fname);
    int row = 0;
    while (inputFile) {
        string s;
        if (!getline(inputFile, s)) break;
        if (s[0] != '#') {
            istringstream ss(s);
            while (ss) {
                for (int i = 0; i < n_dims; i++) {
                    string line;
                    if (!getline(ss, line, ','))
                        break;
                    try {
                        array[i][row] = stod(line);
                    } catch (const invalid_argument e) {
                        cout << "NaN found in file " << fname << " line " << row
                             << endl;
                        e.what();
                    }
                }
            }
        }
        row++;
    }
    if (!inputFile.eof()) {
        cerr << "Could not read file " << fname << endl;
        //__throw_invalid_argument("File not found.");
        return -1;
    }
    return 0;
}

/**
 * Get the mean value from an array of n elements.
 * @param [in] arr the array containing the data.
 * @param [in] n_data the number of elements in the array.
 * @return a double value indicating the mean.
 */
double getMean(double *arr, int n_data) {
    double sum = 0.0;

    for (int i = 0; i<n_data; i++) {
        sum += arr[i];
    }

    return sum/n_data;
}

/**
 * This function centers each dimension in the dataset around the mean value.
 * @param [in, out] data a matrix [n_dims, n_data] containing the dataset.
 * @param [in] n_dims the number of dimensions.
 * @param [in] n_data the number of data.
 */
void Standardize_dataset(double **data, int n_dims, int n_data) {

    for (int i = 0; i < n_dims; ++i) {
        double mean = getMean(data[i], n_data);
        for(int j = 0; j < n_data; j++) {
            data[i][j] = (data[i][j] - mean);
        }
    }
}

/**
 * This function computes the Pearson coefficient of dimensions centered around the mean value.
 * @param [in] X an array containing the data of the first dimension.
 * @param [in] Y an array containing the data of the first dimension.
 * @param [in] n_data the number of data in each array.
 * @return a double value indicating the Pearson coefficient between X and Y.
 */
double PearsonCoefficient(double *X, double *Y, int n_data) {
    //double sum_X = 0, sum_Y = 0;
    double sum_XY = 0, squareSum_X = 0, squareSum_Y = 0;

    for (int i = 0; i < n_data; i++) {
        //sum_X += X[i];
        //sum_Y += Y[i];
        sum_XY += X[i] * Y[i]; // sum of X[i] * Y[i]
        squareSum_X += X[i] * X[i]; // sum of square of array elements
        squareSum_Y += Y[i] * Y[i];
    }

    double corr = (double)sum_XY / sqrt(squareSum_X * squareSum_Y);
    /*double corr = (double)(n_data * sum_XY - sum_X * sum_Y)
                  / sqrt((n_data * squareSum_X - sum_X * sum_X)
                         * (n_data * squareSum_Y - sum_Y * sum_Y));*/
    return corr;
}

/**
 * This function computes the Principal Component Analysis with the 2 Principal Components.
 * @param [in] data_to_transform a matrix [data_dim, n_data] containing the data to transform.
 * @param [in] data_dim the number of dimensions in data_to_transform.
 * @param [in] n_data the number of data in data_to_transform.
 * @param [in, out] new_space the subspace resulting from the input data transformed through the 2 Principal Components.
 */
void PCA_transform(double **data_to_transform, int data_dim, int n_data, double **new_space) {
    double **covar, *covar_storage;
    covar_storage = (double *) malloc(data_dim * data_dim * sizeof(double));
    if (!covar_storage) {
        cout << "Malloc error on covar_storage" << endl;
        exit(-1);
    }
    covar = (double **) malloc(data_dim * sizeof(double *));
    if (!covar) {
        cout << "Malloc error on covar" << endl;
        exit(-1);
    }
    for (int i = 0; i < data_dim; ++i) {
        covar[i] = &covar_storage[i * data_dim];
    }

    for (int i = 0; i < data_dim; ++i) {
        for (int j = i; j < data_dim; ++j) {
            covar[i][j] = 0;
            for (int k = 0; k < n_data; ++k) {
                covar[i][j] += data_to_transform[i][k] * data_to_transform[j][k];
            }
            covar[i][j] = covar[i][j] / (n_data - 1);
            covar[j][i] = covar[i][j];
        }
    }

    mat cov_mat(covar[0], data_dim, data_dim);
    vec eigval;
    mat eigvec;
    eig_sym(eigval, eigvec, cov_mat);

    free(covar_storage);
    free(covar);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < n_data; ++j) {
            double value = 0.0;
            for (int k = 0; k < data_dim; ++k) {
                int col = data_dim - i - 1;
                value += data_to_transform[k][j] * eigvec(k, col);
            }
            new_space[i][j] = value;
        }
    }
}

/**
 * The algorithm calculates the number of elements of an indicated cluster in the given cluster report.
 * @param [in] rep a cluster_report structure describing the K-Means instance carried out.
 * @param [in] cluster_id a number (between 0 and rep.k) indicating the cluster whereby we want to know the number of data in it.
 * @param [in] n_data number of rows in the dataset matrix.
 * @return an integer indicating the number of data in the cluster with the given id.
 */
int cluster_size(cluster_report rep, int cluster_id, int n_data) {
    int occurrence = 0;
    for (int i = 0; i < n_data; ++i) {
        if (rep.cidx[i] == cluster_id) {
            occurrence++;
        }
    }
    return occurrence;
}

/**
 * This function runs the K-Means with the elbow criterion; it tries different number of clusters and
 * evaluates the instance of K-Means through the BetaCV metric in order to find the optimal number of clusters.
 * @param [in] dataset a matrix [2, n_data] containing the data for K-Means run.
 * @param [in] n_data the number of data in dataset.
 * @param [in] k_max the max number of clusters to try for elbow criterion.
 * @param [in] elbow_thr the error tolerance for BetaCV metric.
 * @return a cluster_report containing the result of the K-Means run.
 */
cluster_report run_K_means(double **dataset, int n_data, long k_max, double elbow_thr) {
    mat data(2, n_data);
    mat final;
    cluster_report final_rep, previous_rep;
    previous_rep.cidx = (int *) malloc(n_data * sizeof(int));
    final_rep.cidx = (int *) malloc(n_data * sizeof(int));
    if (previous_rep.cidx == nullptr) {
        cout << "Malloc error on cidx" << endl;
        exit(-1);
    }

    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < n_data; ++i) {
            data(j,i) = dataset[j][i];
        }
    }

    for (int j = 1; j <= k_max; ++j) {
        bool status = kmeans(final, data, j, random_subset, 30, false);
        if (!status) {
            cout << "Error in KMeans run." << endl;
            exit(-1);
        }

        if (j == 1) {
            previous_rep.centroids = final;
            previous_rep.k = j;
            previous_rep.BetaCV = 0.0;
            fill_n(previous_rep.cidx, n_data, 0);
        } else {
            final_rep.centroids = final;
            final_rep.k = j;
            create_cidx_matrix(dataset, n_data, final_rep);
            final_rep.BetaCV = BetaCV(dataset, final_rep, n_data);
            if (fabs(previous_rep.BetaCV - final_rep.BetaCV) <= elbow_thr) {
                cout << "The optimal K is " << final_rep.k << endl;
                return final_rep;
            } else {
                previous_rep = final_rep;
            }
        }
    }
    cout << "The optimal K is " << final_rep.k << endl;
    return final_rep;
}

/**
 * The algorithm creates the array [1, N_DATA] which indicates the membership of each data to a cluster through an integer.
 * @param [in] data a matrix [n_data, n_dims] on which the K-Means run was made.
 * @param [in] n_data number of rows in the dataset matrix.
 * @param [in,out] instance a cluster_report structure describing the K-Means instance carried out.
 */
void create_cidx_matrix(double **data, int n_data, cluster_report instance) {
    for (int i = 0; i < n_data; ++i) {
        double min_dist = L2distance(instance.centroids.at(0,0), instance.centroids.at(1,0), data[0][i], data[1][i]);
        instance.cidx[i] = 0;
        for (int j = 1; j < instance.k; ++j) {
            double new_dist = L2distance(instance.centroids.at(0,j), instance.centroids.at(1,j), data[0][i], data[1][i]);
            if (new_dist < min_dist) {
                min_dist = new_dist;
                instance.cidx[i] = j;
            }
        }
    }
}

/**
 * This function computes the intra-cluster weight, the sum of the Euclidean distance
 * between the centroids of a cluster and each data in it.
 * @param [in] data the data on which the K-Means was executed.
 * @param [in] instance the cluster_report containing the information about the K-Means run.
 * @param [in] n_data the number of data in data.
 * @return a double value representing the intra-cluster weight.
 */
double WithinClusterWeight(double **data, cluster_report instance, int n_data) {
    double wss = 0.0;
    for (int i = 0; i < n_data; ++i) {
        int cluster_idx = instance.cidx[i];
        wss += L2distance(instance.centroids.at(0,cluster_idx), instance.centroids.at(1,cluster_idx), data[0][i], data[1][i]);
    }
    return wss;
}

/**
 * This function computes the inter-cluster weight, the weighted sum of the Euclidean distance
 * between the centroids of each cluster and the mean value of the dataset. The weight is the number
 * of points in each cluster.
 * @param [in] data the data on which the K-Means was executed.
 * @param [in] instance the cluster_report containing the information about the K-Means run.
 * @param [in] n_data the number of data in data.
 * @return a double value representing the inter-cluster weight.
 */
double BetweenClusterWeight(double **data, cluster_report instance, int n_data) {
    double pc1_mean = getMean(data[0], n_data);
    double pc2_mean = getMean(data[1], n_data);
    double bss = 0.0;
    for (int i = 0; i < instance.k; ++i) {
        double n_points = cluster_size(instance, i, n_data);
        bss += n_points * L2distance(instance.centroids.at(0, i), instance.centroids.at(1, i), pc1_mean, pc2_mean);
    }

    return bss;
}

/**
 * This function computes the number of intra-cluster pairs.
 * @param [in] instance the cluster_report containing the information about the K-Means run.
 * @param [in] n_data the number of data on which the K-Means was executed.
 * @return an integer indicating the number of intra-cluster pairs.
 */
int WithinClusterPairs(cluster_report instance, int n_data) {
    int counter = 0;
    for (int i = 0; i < instance.k; ++i) {
        int n_points = cluster_size(instance, i, n_data);
        counter += (n_points - 1) * n_points;
    }
    return counter/2;
}

/**
 * This function computes the number of inter-cluster pairs.
 * @param [in] instance the cluster_report containing the information about the K-Means run.
 * @param [in] n_data the number of data on which the K-Means was executed.
 * @return an integer indicating the number of inter-cluster pairs.
 */
int BetweenClusterPairs(cluster_report instance, int n_data) {
    int counter = 0;
    for (int i = 0; i < instance.k; ++i) {
        int n_points = cluster_size(instance, i, n_data);
        for (int j = 0; j < instance.k; ++j) {
            if (i != j) {
                counter += n_points * cluster_size(instance, j, n_data);
            }
        }
    }
    return counter/2;
}

/**
 * This function computes the BetaCV metric for elbow criterion evaluation.
 * @param [in] data a matrix containing the data on which the K-Means run was executed.
 * @param [in] instance the cluster_report containing the information about the K-Means run.
 * @param [in] n_data the number of data.
 * @return a double value indicating the BetaCV value.
 */
double BetaCV(double **data, cluster_report instance, int n_data) {
    double bss = BetweenClusterWeight(data, instance, n_data);
    double wss = WithinClusterWeight(data, instance, n_data);
    int N_in = WithinClusterPairs(instance, n_data);
    int N_out = BetweenClusterPairs(instance, n_data);

    return ((double) N_out / N_in) * (wss / bss);
}

/**
 * The algorithm calculates the Euclidean distance between two points p_c and p_1.
 * @param [in] xc the first component of p_c.
 * @param [in] yc the second component of p_c.
 * @param [in] x1 the first component of p_1.
 * @param [in] y1 the second component of p_1.
 * @return a double value indicating the Euclidean distance between the two given points.
 */
double L2distance(double xc, double yc, double x1, double y1) {
    double x = xc - x1;
    double y = yc - y1;
    double dist;

    dist = pow(x, 2) + pow(y, 2);
    dist = sqrt(dist);

    return dist;
}

/**
 * Extra function to save centroids and candidate subspace in CSV files.
 */
void csv_out_info(double **data, int n_data, string name, string outdir, bool *incircle, cluster_report report) {
    fstream fout;
    fout.open(outdir + name, ios::out | ios::trunc);

    for (int i = 0; i < n_data; ++i) {
        for (int j = 0; j < 2; ++j) {
            fout << data[j][i] << ",";
        }
        if (incircle[i]) {
            fout << "1,";
        } else {
            fout << "0,";
        }
        fout << report.cidx[i];
        fout << "\n";
    }

    fout.close();
    fstream fout2;
    fout2.open(outdir + "centroids_" + name, ios::out | ios::trunc);
    for (int i = 0; i < report.k; ++i) {
        fout2 << report.centroids.at(0, i) << ",";
        fout2 << report.centroids.at(1, i) << "\n";
    }
    fout2.close();
}