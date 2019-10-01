#include <iostream>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "alglib/dataanalysis.h"
#include "alglib/stdafx.h"

using namespace std;
using namespace alglib;

#define K_MAX 10 //Max number of clusters for Elbow criterion
#define WCSS_THRES 0.1 //Threshold for Within-Cluster Sum of Squares for Elbow criterion
#define PERCENTAGE_INCIRCLE 0.80 //Percentage of points within the circle

int N_DIMS; //Number of dimensions
int N_DATA; //Number of observations
string filename = "../Iris.csv";
//string filename = "../HTRU_2.csv";
//string filename = "../dataset_benchmark/dim256.csv";

struct MaxMin {
    double min;
    double max;
};

void getDatasetDims(string fname);
void loadData(string fname, double **array);
double getMean(double *arr);
struct MaxMin getMinMax(double *arr);
double PearsonCoefficient(double *X, double *Y);
void PCA_transform(double **data_to_transform, int data_dim, double **new_space);
kmeansreport Elbow_K_means(double **data_to_transform);
double L2distance(double xc, double yc, double x1, double y1);
void csv_out_info(double **data, string name, bool *incircle, kmeansreport report);

int main() {
    cout << fixed;
    cout << setprecision(5);

    //Define the structure to load the dataset
    double **data, *data_storage;
    getDatasetDims(filename);

    data_storage = (double *) malloc(N_DIMS * N_DATA * sizeof(double));
    if (data_storage == nullptr) {
        cout << "Malloc error on data_storage" << endl;
        exit(-1);
    }
    data = (double **) malloc(N_DIMS * sizeof(double *));
    if (data == nullptr) {
        cout << "Malloc error on data" << endl;
        exit(-1);
    }

    for (int i = 0; i < N_DIMS; ++i) {
        data[i] = &data_storage[i * N_DATA];
    }

    // Fill the structure from csv
    loadData(filename, data);

    cout << "Dataset Loaded" << endl;
//    for (int i = 0; i < N_DIMS; ++i) {
//        for(int j = 0; j < N_DATA; j++) {
//            cout << data[i][j] << " ";
//        }
//       cout << "\n";
//    }

    auto start = chrono::steady_clock::now();

    //Standardization
    for (int i = 0; i < N_DIMS; ++i) {
        double mean = getMean(data[i]);
        for(int j = 0; j < N_DATA; j++) {
            data[i][j] = (data[i][j] - mean);
        }
    }

    // Normalization - MaxMinScaler
//    struct MaxMin minmax = getMinMax(data_storage);
//
//    for (int i = 0; i < N_DATA * N_DIMS; ++i) {
//        data_storage[i] = (data_storage[i] - minmax.min)/(minmax.max - minmax.min);
//    }

    cout << "Dataset standardized" << endl;
//    for (int i = 0; i < N_DIMS; ++i) {
//        for(int j = 0; j < N_DATA; j++) {
//            cout << data[i][j] << " ";
//        }
//        cout << "\n";
//    }

    //Define the structure to load the Correlation Matrix
    double **pearson, *pearson_storage;
    pearson_storage = (double *) malloc(N_DIMS * N_DIMS * sizeof(double));
    if (pearson_storage == nullptr) {
        cout << "Malloc error on pearson_storage" << endl;
        exit(-1);
    }
    pearson = (double **) malloc(N_DIMS * sizeof(double *));
    if (pearson == nullptr) {
        cout << "Malloc error on pearson" << endl;
        exit(-1);
    }

    for (int i = 0; i < N_DIMS; ++i) {
        pearson[i] = &pearson_storage[i * N_DIMS];
    }

    for (int i = 0; i < N_DIMS; ++i) {
        pearson[i][i] = 1;
        for (int j = i+1; j < N_DIMS; ++j) {
            double value = PearsonCoefficient(data[i], data[j]);
            pearson[i][j] = value;
            pearson[j][i] = value;
        }
    }

    cout << "Pearson Correlation Coefficient computed" << endl;
//    for (int i = 0; i < N_DIMS; ++i) {
//        for(int j = 0; j < N_DIMS; j++) {
//            cout << pearson[i][j] << " ";
//        }
//        cout << "\n";
//    }

    //Dimensions partitioned in CORR and UNCORR, then CORR U UNCORR = DIMS
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
    if (corr == nullptr) {
        cout << "Malloc error on corr" << endl;
        exit(-1);
    }
    uncorr = (int *) (malloc(uncorr_vars * sizeof(int)));
    if (uncorr == nullptr) {
        cout << "Malloc error on corr or uncorr" << endl;
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

//    cout << "CORR subspace" << endl;
//    for (int i = 0; i < corr_vars; ++i) {
//        for(int j = 0; j < N_DATA; j++) {
//            cout << corr[i][j] << " ";
//        }
//        cout << "\n";
//    }
//
//    cout << "UNCORR subspace" << endl;
//    for (int i = 0; i < uncorr_vars; ++i) {
//        cout << uncorr[i] << " ";
//    }
//    cout << "\n";

    //Define the structure to save PC1_corr and PC2_corr ------ matrix [2 * N_DATA]
    double **newspace, *newspace_storage;
    newspace_storage = (double *) malloc(N_DATA * 2 * sizeof(double));
    if (newspace_storage == nullptr) {
        cout << "Malloc error on newspace_storage" << endl;
        exit(-1);
    }
    newspace = (double **) malloc(2 * sizeof(double *));
    if (newspace == nullptr) {
        cout << "Malloc error on newspace" << endl;
        exit(-1);
    }
    for (int i = 0; i < 2; ++i) {
        newspace[i] = &newspace_storage[i*N_DATA];
    }

    PCA_transform(corr, corr_vars, newspace);

    free(corr);
    cout << "PCA computed on CORR subspace" << endl;

    //Define the structure to save the candidate subspaces ------ #UNCORR * matrix [2 * N_DATA]
    double *cs_storage, **csi, ***cs;
    cs_storage = (double *) malloc(N_DATA * 2 * uncorr_vars * sizeof(double));
    if (cs_storage == nullptr) {
        cout << "Malloc error on cs_storage" << endl;
        exit(-1);
    }
    csi = (double **) malloc(2 * uncorr_vars * sizeof(double *));
    if (csi == nullptr) {
        cout << "Malloc error on csi" << endl;
        exit(-1);
    }
    cs = (double ***) malloc(uncorr_vars * sizeof(double **));
    if (cs == nullptr) {
        cout << "Malloc error on cs" << endl;
        exit(-1);
    }

    for (int i = 0; i < 2 * uncorr_vars; ++i) {
        csi[i] = &cs_storage[i * N_DATA];
    }
    for (int i = 0; i < uncorr_vars; ++i) {
        cs[i] = &csi[i * 2];
    }

    //Define the structure to save the candidate subspace CS_i ------ matrix [3 * N_DATA]
    double **combine, *combine_storage;
    combine_storage = (double *) malloc(N_DATA * 3 * sizeof(double));
    if (combine_storage == nullptr) {
        cout << "Malloc error on combine_storage" << endl;
        exit(-1);
    }
    combine = (double **) malloc(3 * sizeof(double *));
    if (combine == nullptr) {
        cout << "Malloc error on combine" << endl;
        exit(-1);
    }
    for (int l = 0; l < 3; ++l) {
        combine[l] = &combine_storage[l * N_DATA];
    }

    //Concatenate PC1_corr, PC2_corr
    memcpy(combine[0], newspace[0], N_DATA);
    memcpy(combine[1], newspace[1], N_DATA);

    free(newspace_storage);
    free(newspace);

    for (int i = 0; i < uncorr_vars; ++i) {
        //Concatenate PC1_corr, PC2_corr and i-th dimension of uncorr
        memcpy(combine[2], data[uncorr[i]], N_DATA);

        PCA_transform(combine, 3, cs[i]);
        cout << "PCA computed on PC1_CORR, PC2_CORR and " << i+1 << "-th dimension of UNCORR" << endl;
    }

    free(combine_storage);
    free(combine);

    //boolean matrix [uncorr_var*N_DATA]: True says the data is in the circles, otherwise False
    bool **incircle, *incircle_storage;
    incircle_storage = (bool *) malloc(uncorr_vars * N_DATA * sizeof(bool));
    if (incircle_storage == nullptr) {
        cout << "Malloc error on incircle_storage" << endl;
        exit(-1);
    }
    incircle = (bool **) malloc(uncorr_vars * sizeof(bool *));
    if (incircle == nullptr) {
        cout << "Malloc error on incircle" << endl;
        exit(-1);
    }

    for (int i = 0; i < uncorr_vars; ++i) {
        incircle[i] = &incircle_storage[i * N_DATA];
    }

    for (int i = 0; i < uncorr_vars; ++i) {
        for(int j = 0; j < N_DATA; j++) {
            incircle[i][j] = false;
        }
    }

    for (int i = 0; i < uncorr_vars; ++i) {
        kmeansreport rep;
        cout << "Candidate Subspace " << i+1 << ": ";
        rep = Elbow_K_means(cs[i]); //Clustering through Elbow criterion on i-th candidate subspace
        for (int j = 0; j < rep.k; ++j) {
            int k = 0, previous_k = 0;
            while (k < PERCENTAGE_INCIRCLE * N_DATA/rep.k) {
                double dist = 0.0;
                int n_points = 0;
                for (int l = 0; l < N_DATA; ++l) {
                    if (rep.cidx[l] == j && !(incircle[i][l])) {
                        dist += L2distance(rep.c[j][0], rep.c[j][1], cs[i][0][l], cs[i][1][l]);
                        n_points++;
                    }
                }
                double dist_mean = dist/n_points;
                for (int l = 0; l < N_DATA; ++l) {
                    if (rep.cidx[l] == j && !(incircle[i][l])) {
                        if (L2distance(rep.c[j][0], rep.c[j][1], cs[i][0][l], cs[i][1][l]) <= dist_mean) {
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
        string fileoutname = "dataout";
        string num = to_string(i);
        string concat = fileoutname + num + ".csv";
        csv_out_info(cs[i], concat, incircle[i], rep);
    }

    auto end = chrono::steady_clock::now();

    cout << "Outliers Identification Process: \n";

    int tot_outliers = 0;
    for (int i = 0; i < N_DATA; ++i) {
        int occurrence = 0;
        for (int j = 0; j < uncorr_vars; ++j) {
            if (!incircle[j][i]) {
                occurrence++;
            }
        }

        if (occurrence > 0) {
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

    cout << "Elapsed time in milliseconds : "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    return 0;
}

void getDatasetDims(string fname) {
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
    N_DIMS = cols;
    N_DATA = rows;
    cout << "Dataset: #DATA = " << N_DATA << " , #DIMENSIONS = " << N_DIMS << endl;
    file.close();
}

void loadData(string fname, double **array) {
    ifstream inputFile(fname);
    int row = 0;
    while (inputFile) {
        string s;
        if (!getline(inputFile, s)) break;
        if (s[0] != '#') {
            istringstream ss(s);
            while (ss) {
                for (int i = 0; i < N_DIMS; i++) {
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
        __throw_invalid_argument("File not found.");
    }
}

double getMean(double *arr) {
    double sum = 0.0;

    for (int i = 0; i<N_DATA; i++) {
        sum += arr[i];
    }

    return sum/N_DATA;
}

struct MaxMin getMinMax(double *arr) {
    struct MaxMin minmax;

    /*If there is only one element then return it as min and max both*/
    if (N_DATA * N_DIMS == 1) {
        minmax.max = arr[0];
        minmax.min = arr[0];
        return minmax;
    }

    /* If there are more than one elements, then initialize min and max*/
    if (arr[0] > arr[1]) {
        minmax.max = arr[0];
        minmax.min = arr[1];
    } else {
        minmax.max = arr[1];
        minmax.min = arr[0];
    }

    for (int i = 2; i<N_DATA * N_DIMS; i++) {
        if (arr[i] >  minmax.max)
            minmax.max = arr[i];
        else if (arr[i] <  minmax.min)
            minmax.min = arr[i];
    }
    return minmax;
}

double PearsonCoefficient(double *X, double *Y) {
    double sum_X = 0, sum_Y = 0, sum_XY = 0;
    double squareSum_X = 0, squareSum_Y = 0;

    for (int i = 0; i < N_DATA; i++) {
        sum_X += X[i];
        sum_Y += Y[i];
        sum_XY += X[i] * Y[i]; // sum of X[i] * Y[i]
        squareSum_X += X[i] * X[i]; // sum of square of array elements
        squareSum_Y += Y[i] * Y[i];
    }

    double corr = (double)(N_DATA * sum_XY - sum_X * sum_Y)
                  / sqrt((N_DATA * squareSum_X - sum_X * sum_X)
                         * (N_DATA * squareSum_Y - sum_Y * sum_Y));
    return corr;
}

void PCA_transform(double **data_to_transform, int data_dim, double **new_space) {
    real_2d_array dset, basis;
    real_1d_array variances;
    variances.setlength(2);
    dset.setlength(N_DATA, data_dim);
    basis.setlength(data_dim, 2);
    for (int i = 0; i < N_DATA; ++i) {
        for(int j = 0; j < data_dim; j++) {
            dset[i][j] = data_to_transform[j][i];
        }
    }

    pcatruncatedsubspace(dset, N_DATA, data_dim, 2, 0.0, 0, variances, basis);

//    cout << "PCA result: " << endl;
//    for (int i = 0; i < data_dim; ++i) {
//        for(int j = 0; j < 2; j++) {
//            cout << basis[i][j] << " ";
//        }
//        cout << "\n";
//    }

    for (int i=0; i < N_DATA; i++) {
        for (int j=0;j<2;j++) {
            new_space[j][i]=0;
            for (int k=0;k<data_dim;k++) {
                new_space[j][i] += dset[i][k] * basis[k][j];
            }
        }
    }

//    cout << "The resulting subspace: " << endl;
//    for(int j = 0; j < 2; j++) {
//        for (int i = 0; i < N_DATA; ++i) {
//            cout << new_space[j][i] << " ";
//        }
//        cout << "\n";
//    }
}

kmeansreport Elbow_K_means(double **data_to_transform) {
    clusterizerstate status;
    real_2d_array data;
    kmeansreport final;

    data.setlength(N_DATA, 2);
    for (int i = 0; i < N_DATA; ++i) {
        for (int j = 0; j < 2; j++) {
            data[i][j] = data_to_transform[j][i];
        }
    }

    kmeansreport previous;
    for (int j = 1; j <= K_MAX; ++j) {
        clusterizercreate(status);
        clusterizersetpoints(status, data, 2);
        clusterizerrunkmeans(status, j, final);
        if (final.terminationtype < 0) {
            cout << "Error in KMeans run. Termination Type: " << final.terminationtype << endl;
            exit(-1);
        }

        if (j == 1) {
            previous = final;
        } else {
            //Sum of Squares within-clusters
            if ((previous.energy - final.energy) <= WCSS_THRES) {
                cout << "The optimal K is " << final.k << endl;
                return final;
            } else {
                previous = final;
            }
        }
    }
    cout << "The optimal K is " << final.k << endl;
    return final;
}

double L2distance(double xc, double yc, double x1, double y1)
{
    double x = xc - x1; //calculating number to square in next step
    double y = yc - y1;
    double dist;

    dist = pow(x, 2) + pow(y, 2);       //calculating Euclidean distance
    dist = sqrt(dist);

    return dist;
}

void csv_out_info(double **data, string name, bool *incircle, kmeansreport report) {
    fstream fout;
    fout.open("../plot/HTRU/" + name, ios::out | ios::app);

    for (int i = 0; i < N_DATA; ++i) {
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
    fout2.open("../plot/HTRU/centroids_" + name, ios::out | ios::app);
    for (int i = 0; i < report.k; ++i) {
        for (int j = 0; j < 2; ++j) {
            fout2 << report.c[i][j] << ",";
        }
        fout2 << "\n";
    }
    fout2.close();
}
