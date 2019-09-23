#include <iostream>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <fstream>
#include "alglib/dataanalysis.h"
#include "alglib/stdafx.h"

using namespace std;
using namespace alglib;

#define K_MAX 10 //Max number of clusters for Elbow criterion
#define DATA_THRES 25 //Max number of points within the circle

int N_ROWS; //Number of dimensions
int N_COLS; //Number of observations
string filename = "../Iris.csv";

struct stats {
    double mean;
    double stdev;
};

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
    N_ROWS = cols;
    N_COLS = rows;
    cout << "Dataset: #DATA = " << rows << " , #DIMENSIONS = " << cols << "\n";
    file.close();
}

void loadData(string fname, double **array) {
    char buf[1024];
    char *token;
    int column = 0;
    FILE *fp = fopen(fname.c_str(), "r");
    if (!fp) {
        printf("Can't open file\n");
        exit(1);
    }
    while (fgets(buf, 1024, fp)) {
        token = strtok(buf, ",");
        for (int i=0; i < N_ROWS; i++) {
            if (token == NULL) {
                break;
            }
            // convert text numeral to double
            double val = atof(token);
            array[i][column] = val;
            token = strtok(NULL, ",");
        }
        column++;
    }
    fclose(fp);
}

struct stats getStats(double *arr, int n) {
    struct stats info;
    double sum = 0.0, st_dev = 0.0;
    int i;

    for (i = 0; i<n; i++) {
        sum += arr[i];
    }
    info.mean = sum/n;
    for (i = 0; i<n; i++) {
        st_dev += pow(arr[i] - info.mean, 2);
    }
    info.stdev = st_dev;
    return info;
}

double PearsonCoefficient(double *X, double *Y, int n) {
    double sum_X = 0, sum_Y = 0, sum_XY = 0;
    double squareSum_X = 0, squareSum_Y = 0;

    for (int i = 0; i < n; i++) {
        sum_X += X[i];
        sum_Y += Y[i];
        sum_XY += X[i] * Y[i]; // sum of X[i] * Y[i]
        squareSum_X += X[i] * X[i]; // sum of square of array elements
        squareSum_Y += Y[i] * Y[i];
    }

    double corr = (double)(n * sum_XY - sum_X * sum_Y)
                  / sqrt((n * squareSum_X - sum_X * sum_X)
                         * (n * squareSum_Y - sum_Y * sum_Y));
    return corr;
}

void PCA_transform(double **data_to_transform, int data_dim, double **new_space) {
    real_2d_array dset, basis;
    real_1d_array variances;
    variances.setlength(2);
    dset.setlength(N_COLS, data_dim);
    basis.setlength(data_dim, 2);
    for (int i = 0; i < N_COLS; ++i) {
        for(int j = 0; j < data_dim; j++) {
            dset[i][j] = data_to_transform[j][i];
        }
    }

    pcatruncatedsubspace(dset, N_COLS, data_dim, 2, 0.0, 0, variances, basis);

    cout << "-------------PCA--------------------\n";
    for (int i = 0; i < data_dim; ++i) {
        for(int j = 0; j < 2; j++) {
            cout << basis[i][j] << " ";
        }
        cout << "\n";
    }

    for (int i=0;i<N_COLS;i++) {
        for (int j=0;j<2;j++) {
            new_space[j][i]=0;
            for (int k=0;k<data_dim;k++) {
                new_space[j][i] += dset[i][k] * basis[k][j];
            }
        }
    }

    cout << "-------------NEW SUBSPACE--------------------\n";
    for(int j = 0; j < 2; j++) {
        for (int i = 0; i < N_COLS; ++i) {
            cout << new_space[j][i] << " ";
        }
        cout << "\n";
    }
}

kmeansreport Elbow_K_means(double **data_to_transform) {
    clusterizerstate status;
    real_2d_array data;
    kmeansreport final;

    data.setlength(N_COLS, 2);
    for (int i = 0; i < N_COLS; ++i) {
        for (int j = 0; j < 2; j++) {
            data[i][j] = data_to_transform[j][i];
        }
    }

    kmeansreport previous;
    for (int j = 1; j <= K_MAX; ++j) {
        clusterizercreate(status);
        clusterizersetpoints(status, data, 2);
        clusterizerrunkmeans(status, j, final);
        if (j == 1) {
            previous = final;
        } else {
            if (previous.energy - final.energy <= 0.01) {
                final = previous;
                cout << "The optimal K is: " << final.k << "\n";
                return final;
            } else {
                previous = final;
            }
        }
    }
    cout << "The optimal K is: " << final.k << "\n";
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

int main() {
    cout << fixed;
    cout << setprecision(5);

    //Define the structure to load the dataset
    getDatasetDims(filename);
    double **data, *data_storage;
    data_storage = (double *) malloc(N_ROWS * N_COLS * sizeof(double));
    data = (double **) malloc(N_ROWS * sizeof(double *));
    for (int i = 0; i < N_ROWS; ++i) {
        data[i] = &data_storage[i*N_COLS];
    }

    // Fill the structure from csv
    loadData(filename, data);

    cout << "-------------DATASET LOADED--------------------\n";
    for (int i = 0; i < N_ROWS; ++i) {
        for(int j = 0; j < N_COLS; j++) {
            cout << data[i][j] << " ";
        }
       cout << "\n";
    }

    //Standardization
    for (int i = 0; i < N_ROWS; ++i) {
        struct stats info = getStats(data[i], N_COLS);
        for(int j = 0; j < N_COLS; j++) {
            data[i][j] = (data[i][j] - info.mean)/info.stdev;
        }
    }

    cout << "-------------STD DATASET--------------------\n";
    for (int i = 0; i < N_ROWS; ++i) {
        for(int j = 0; j < N_COLS; j++) {
            cout << data[i][j] << " ";
        }
        cout << "\n";
    }


//    // Normalization - MaxMinScaler
//    struct pair minmax = getMinMax(data_storage, N_ROWS * N_COLS);
//
//    for (int i = 0; i < N_ROWS * N_COLS; ++i) {
//        data_storage[i] = (data_storage[i] - minmax.min)/(minmax.max - minmax.min);
//    }


    //Define the structure to load the Correlation Matrix
    double **pearson, *pearson_storage;
    pearson_storage = (double *) malloc(N_ROWS * N_ROWS * sizeof(double));
    pearson = (double **) malloc(N_ROWS * sizeof(double *));
    for (int i = 0; i < N_ROWS; ++i) {
        pearson[i] = &pearson_storage[i*N_ROWS];
    }

    for (int i = 0; i < N_ROWS; ++i) {
        pearson[i][i] = 1;
        for (int j = i+1; j < N_ROWS; ++j) {
            double value = PearsonCoefficient(data[i], data[j], N_COLS);
            pearson[i][j] = value;
            pearson[j][i] = value;
        }
    }

    printf("----------------PEARSON MATRIX-----------------\n");
    for (int i = 0; i < N_ROWS; ++i) {
        for(int j = 0; j < N_ROWS; j++) {
            cout << pearson[i][j] << " ";
        }
        cout << "\n";
    }

    //-------------------------------------------------------------------TEST

    //Divide dimensions in CORR and UNCORR
    int corr_vars = 0, uncorr_vars = 0;
    for (int i = 0; i < N_ROWS; ++i) {
        double overall = 0.0;
        for (int j = 0; j < N_ROWS; ++j) {
            if (i != j) {
                overall += pearson[i][j];
            }
        }
        if (overall >= 0) {
            corr_vars++;
        } else {
            uncorr_vars++;
        }
    }

    double **corr, **uncorr;
    corr = (double **) malloc(corr_vars * sizeof(double *));
    uncorr = (double **) malloc(uncorr_vars * sizeof(double *));

    corr_vars = 0, uncorr_vars = 0;
    for (int i = 0; i < N_ROWS; ++i) {
        double overall = 0.0;
        for (int j = 0; j < N_ROWS; ++j) {
            if (i != j) {
                overall += pearson[i][j];
            }
        }
        if (overall >= 0) {
            corr[corr_vars] = data[i];
            corr_vars++;
        } else {
            uncorr[uncorr_vars] = data[i];
            uncorr_vars++;
        }
    }

    printf("----------------CORR DIMS-----------------\n");
    for (int i = 0; i < corr_vars; ++i) {
        for(int j = 0; j < N_COLS; j++) {
            cout << corr[i][j] << " ";
        }
        cout << "\n";
    }

    printf("----------------UNCORR DIMS-----------------\n");
    for (int i = 0; i < uncorr_vars; ++i) {
        for(int j = 0; j < N_COLS; j++) {
            cout << uncorr[i][j] << " ";
        }
        cout << "\n";
    }
    //-------------------------------------------------------------------END TEST

    //Define the structure to save PC1_corr and PC2_corr ------ matrix [2 * N_COLS]
    double **newspace, *newspace_storage;
    newspace_storage = (double *) malloc(N_COLS * 2 * sizeof(double));
    newspace = (double **) malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; ++i) {
        newspace[i] = &newspace_storage[i*2];
    }

    PCA_transform(corr, corr_vars, newspace);

    //Define the structure to save the candidate subspaces ------ #UNCORR * matrix [2 * N_COLS]
    double *storage, **csi, ***cs;
    storage = (double *) malloc(N_COLS * 2 * uncorr_vars * sizeof(double));
    csi = (double **) malloc(2 * uncorr_vars * sizeof(double *));
    cs = (double ***) malloc(uncorr_vars * sizeof(double **));

    for (int i = 0; i < 2 * uncorr_vars; ++i) {
        csi[i] = &storage[i*N_COLS];
    }
    for (int i = 0; i < uncorr_vars; ++i) {
        cs[i] = &csi[i * 2];
    }

    for (int i = 0; i < uncorr_vars; ++i) {
        //Define the structure to save the candidate subspace CS_i ------ matrix [3 * N_COLS]
        double **combine, *combine_storage;
        combine_storage = (double *) malloc(N_COLS * 3 * sizeof(double));
        combine = (double **) malloc(3 * sizeof(double *));
        for (int l = 0; l < 3; ++l) {
            combine[l] = &combine_storage[l * N_COLS];
        }

        //Concatenate PC1_corr, PC2_corr and i-th dimension of uncorr
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < N_COLS; k++) {
                if (j <= 1) {
                    combine[j][k] = newspace[j][k];
                } else {
                    combine[j][k] = uncorr[i][k];
                }
            }
        }

        PCA_transform(combine, 3, cs[i]);
    }

    //binary matrix [uncorr_var*NCOLS]: 1 says data within the circles, otherwise 0
    int **incircle, *incircle_storage;
    incircle_storage = (int *) malloc(uncorr_vars * N_COLS * sizeof(int));
    incircle = (int **) malloc(uncorr_vars * sizeof(int *));
    for (int i = 0; i < uncorr_vars; ++i) {
        incircle[i] = &incircle_storage[i * N_COLS];
    }

    for (int i = 0; i < uncorr_vars; ++i) {
        for(int j = 0; j < N_COLS; j++) {
            incircle[i][j] = 0;
        }
    }

    for (int i = 0; i < uncorr_vars; ++i) {
        kmeansreport rep;
        rep = Elbow_K_means(cs[i]); //Clustering through Elbow criterion on i-th candidate subspace
        for (int j = 0; j < rep.k; ++j) {
            int k = 0, previous_k = 0;
            while ( k < DATA_THRES) {
                double dist = 0.0;
                int n_points = 0;
                for (int l = 0; l < N_COLS; ++l) {
                    if (rep.cidx[l] == j && !(incircle[i][l])) {
                        dist += L2distance(rep.c[j][0], rep.c[j][1], cs[i][0][l], cs[i][1][l]);
                        n_points++;
                    }
                }
                double dist_mean = dist/n_points;
                for (int l = 0; l < N_COLS; ++l) {
                    if (rep.cidx[l] == j && !(incircle[i][l])) {
                        if (L2distance(rep.c[j][0], rep.c[j][1], cs[i][0][l], cs[i][1][l]) <= dist_mean) {
                            incircle[i][l] = 1;
                            k++;
                        }
                    }
                }
                //Stopping criterion when the data threshold is greater than n_points in a cluster
                if (k == previous_k) {
                    break;
                } else {
                    previous_k = k;
                }
            }
        }
    }

    cout << "Outliers Identification Process: \n";

    int tot_outliers = 0;
    for (int i = 0; i < N_COLS; ++i) {
        int occurrence = 0;
        for (int j = 0; j < uncorr_vars; ++j) {
            occurrence += incircle[j][i];
        }

        if (occurrence > 0) {
            tot_outliers++;
            cout << i << ") ";
            for (int l = 0; l < N_ROWS; ++l) {
                cout << data[l][i] << " ";
            }
            cout << "(" << occurrence << ")\n";
        }
    }

    cout << "TOTAL NUMBER OF OUTLIERS: " << tot_outliers;

    return 0;
}