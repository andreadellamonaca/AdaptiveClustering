#include "adaptive_clustering.h"
#include "error.h"

int getDatasetDims(string fname, int &dim, int &data) {
    dim = 0;
    data = 0;
    ifstream file(fname);

    string s;
    if(file.is_open()){

        while (getline(file, s)) {

            if (data == 0) {
                istringstream ss(s);
                string line;
                while (getline(ss, line, ',')) {
                    dim++;
                }
            }

            data++;
        }

    }
    else {
        cerr << "the file " << fname << " is not open!" << endl;
        return InputFileError(__FUNCTION__);
    }

    cout << "Dataset: #DATA = " << data << " , #DIMENSIONS = " << dim << endl;
    file.close();
    return 0;
}

int loadData(string fname, double **array, int n_dims) {
    if (!array) {
        return NullPointerError(__FUNCTION__);
    }

    ifstream inputFile(fname);
    int row = 0;

    string s;
    if(inputFile.is_open()){

        while (getline(inputFile, s)) {

             if (s[0] != '#') {

                 istringstream ss(s);

                 for (int i = 0; i < n_dims; i++) {
                     string line;
                     getline(ss, line, ',');

                    try {
                        array[i][row] = stod(line);
                    }
                    catch (const invalid_argument e) {
                        cout << "NaN found in file " << fname << " line " << row << endl;
                        inputFile.close();
                        return ConversionError(__FUNCTION__);
                    }
                }

             }

             row++;
         }

         inputFile.close();

    }
    else {
        return InputFileError(__FUNCTION__);
    }

    return 0;

}

double getMean(double *arr, int n_data) {
    if (!arr) {
        exit(NullPointerError(__FUNCTION__));
    }

    double sum = 0.0;
    for (int i = 0; i < n_data; ++i) {
        sum += arr[i];
    }

    return sum/n_data;
}

int Standardize_dataset(double **data, int n_dims, int n_data) {
    if (!data) {
        return NullPointerError(__FUNCTION__);
    }

    for (int i = 0; i < n_dims; ++i) {
        double mean = getMean(data[i], n_data);
        for(int j = 0; j < n_data; j++) {
            data[i][j] = (data[i][j] - mean);
        }
    }

    return 0;
}

double PearsonCoefficient(double *X, double *Y, int n_data) {
    if (!X || !Y) {
        exit(NullPointerError(__FUNCTION__));
    }

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

int computePearsonMatrix(double **pearson, double **data, int n_data, int n_dims) {
    if (!pearson || !data) {
        return NullPointerError(__FUNCTION__);
    }

    for (int i = 0; i < n_dims; ++i) {
        pearson[i][i] = 1;
        for (int j = i+1; j < n_dims; ++j) {
            double value = PearsonCoefficient(data[i], data[j], n_data);
            pearson[i][j] = value;
            pearson[j][i] = value;
        }
    }

    return 0;
}

bool isCorrDimension(int ndims, int dimensionID, double **pcc) {
    if (!pcc) {
        exit(NullPointerError(__FUNCTION__));
    }

    double overall = 0.0;
    for (int secondDimension = 0; secondDimension < ndims; ++secondDimension) {
        if (secondDimension != dimensionID) {
            overall += pcc[dimensionID][secondDimension];
        }
    }

    return ( (overall / ndims) >= 0 );
}

int computeCorrUncorrCardinality(double **pcc, int ndims, int &corr_vars, int &uncorr_vars) {
    if (!pcc) {
        return NullPointerError(__FUNCTION__);
    }

    corr_vars = 0, uncorr_vars = 0;
    for (int i = 0; i < ndims; ++i) {
        if (isCorrDimension(ndims, i, pcc)) {
            corr_vars++;
        }
    }
    uncorr_vars = ndims - corr_vars;

    return 0;
}

int PCA_transform(double **data_to_transform, int data_dim, int n_data, double **new_space) {
    if (!data_to_transform || !new_space) {
        return NullPointerError(__FUNCTION__);
    }

    int status = -12;
    double **covar = NULL, *covar_storage = NULL;
    covar_storage = (double *) malloc(data_dim * data_dim * sizeof(double));
    if (!covar_storage) {
        return MemoryError(__FUNCTION__);
    }
    covar = (double **) malloc(data_dim * sizeof(double *));
    if (!covar) {
        free(covar_storage), covar_storage = NULL;
        status = MemoryError(__FUNCTION__);
        return status;
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
            covar[i][j] = covar[i][j] / n_data;
            covar[j][i] = covar[i][j];
        }
    }

    mat cov_mat(covar[0], data_dim, data_dim);
    vec eigval;
    mat eigvec;
    eig_sym(eigval, eigvec, cov_mat);

    free(covar_storage), covar_storage = NULL;
    free(covar), covar = NULL;

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

    return 0;
}

int cluster_size(cluster_report rep, int cluster_id, int n_data) {
    if (!rep.cidx) {
        exit(NullPointerError(__FUNCTION__));
    }

    int occurrence = 0;
    for (int i = 0; i < n_data; ++i) {
        if (rep.cidx[i] == cluster_id) {
            occurrence++;
        }
    }

    return occurrence;
}

cluster_report run_K_means(double **dataset, int n_data, long k_max, double elbow_thr) {
    if (!dataset) {
        exit(NullPointerError(__FUNCTION__));
    }

    mat data(2, n_data);
    mat final;
    double previous_BetaCV = 0.0;
    cluster_report final_rep;
    final_rep.cidx = (int *) malloc(n_data * sizeof(int));
    if (!final_rep.cidx) {
        exit(MemoryError(__FUNCTION__));
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
        if (j > 1) {
            final_rep.centroids = final;
            final_rep.k = j;
            create_cidx_matrix(dataset, n_data, final_rep);
            final_rep.BetaCV = BetaCV(dataset, final_rep, n_data);
            if (fabs(previous_BetaCV - final_rep.BetaCV) <= elbow_thr) {
                cout << "The optimal K is " << final_rep.k << endl;
                return final_rep;
            } else {
                previous_BetaCV = final_rep.BetaCV;
            }
        }
    }
    cout << "The optimal K is " << final_rep.k << endl;

    return final_rep;
}

int create_cidx_matrix(double **data, int n_data, cluster_report instance) {
    if (!data || instance.centroids.is_empty() || !(instance.cidx)) {
        return NullPointerError(__FUNCTION__);
    }

    for (int i = 0; i < n_data; ++i) {
        double min_dist = L2distance(instance.centroids(0,0), instance.centroids(1,0), data[0][i], data[1][i]);
        instance.cidx[i] = 0;
        for (int j = 1; j < instance.k; ++j) {
            double new_dist = L2distance(instance.centroids(0,j), instance.centroids(1,j), data[0][i], data[1][i]);
            if (new_dist < min_dist) {
                min_dist = new_dist;
                instance.cidx[i] = j;
            }
        }
    }

    return 0;
}

double WithinClusterWeight(double **data, cluster_report instance, int n_data) {
    if (!data || instance.centroids.is_empty() || !(instance.cidx)) {
        exit(NullPointerError(__FUNCTION__));
    }

    double wss = 0.0;
    for (int i = 0; i < n_data; ++i) {
        int cluster_idx = instance.cidx[i];
        wss += L2distance(instance.centroids(0,cluster_idx), instance.centroids(1,cluster_idx), data[0][i], data[1][i]);
    }

    return wss;
}

double BetweenClusterWeight(double **data, cluster_report instance, int n_data) {
    if (!data || instance.centroids.is_empty()) {
        exit(NullPointerError(__FUNCTION__));
    }

    double pc1_mean = getMean(data[0], n_data);
    double pc2_mean = getMean(data[1], n_data);
    double bss = 0.0;
    for (int i = 0; i < instance.k; ++i) {
        double n_points = cluster_size(instance, i, n_data);
        bss += n_points * L2distance(instance.centroids(0, i), instance.centroids(1, i), pc1_mean, pc2_mean);
    }

    return bss;
}

int WithinClusterPairs(cluster_report instance, int n_data) {
    int counter = 0;
    for (int i = 0; i < instance.k; ++i) {
        int n_points = cluster_size(instance, i, n_data);
        counter += (n_points - 1) * n_points;
    }

    return counter/2;
}

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

double BetaCV(double **data, cluster_report instance, int n_data) {
    if (!data) {
        exit(NullPointerError(__FUNCTION__));
    }

    double bss = BetweenClusterWeight(data, instance, n_data);
    double wss = WithinClusterWeight(data, instance, n_data);
    int N_in = WithinClusterPairs(instance, n_data);
    int N_out = BetweenClusterPairs(instance, n_data);

    return ((double) N_out / N_in) * (wss / bss);
}

double L2distance(double xc, double yc, double x1, double y1) {
    double x = xc - x1;
    double y = yc - y1;
    double dist;
    dist = pow(x, 2) + pow(y, 2);
    dist = sqrt(dist);

    return dist;
}

int computeActualClusterInfo(cluster_report rep, int n_data, int clusterID, bool *incircle, double **subspace, double &dist) {
    if (!incircle || !subspace || !(rep.cidx) || rep.centroids.is_empty()) {
        exit(NullPointerError(__FUNCTION__));
    }

    dist = 0.0;
    int actual_cluster_size = 0;
    for (int l = 0; l < n_data; ++l) {
        if (rep.cidx[l] == clusterID && !(incircle[l])) {
            dist += L2distance(rep.centroids(0, clusterID), rep.centroids(1, clusterID),
                               subspace[0][l], subspace[1][l]);
            actual_cluster_size++;
        }
    }

    return actual_cluster_size;
}

int computeInliers(cluster_report rep, int n_data, int clusterID, bool *incircle, double **subspace, double radius) {
    if (!incircle || !subspace || !(rep.cidx) || rep.centroids.is_empty()) {
        exit(NullPointerError(__FUNCTION__));
    }

    int count = 0;
    for (int l = 0; l < n_data; ++l) {
        if (rep.cidx[l] == clusterID && !(incircle[l])) {
            if (L2distance(rep.centroids(0,clusterID), rep.centroids(1,clusterID), subspace[0][l], subspace[1][l]) <= radius) {
                incircle[l] = true;
                count++;
            }
        }
    }

    return count;
}

int countOutliers(bool **incircle, int uncorr_vars, int data_idx) {
    if (!incircle) {
        exit(NullPointerError(__FUNCTION__));
    }

    int count = 0;
    for (int j = 0; j < uncorr_vars; ++j) {
        if (!incircle[j][data_idx]) {
            count++;
        }
    }

    return count;
}
