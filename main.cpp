#include <iostream>
#include <cstring>
#include <cmath>
#include "alglib/dataanalysis.h"

using namespace std;
using namespace alglib;

#define N_ROWS 4 //NUMBER OF DIMENSIONS
#define N_COLS 150 //NUMBER OF OBSERVATIONS

struct minmaxinfo {
    double min;
    double max;
};

struct stats {
    double mean;
    double stdev;
};

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

struct minmaxinfo getMinMax(double *arr, int n) {
    struct minmaxinfo minmax;
    int i;

    /*If there is only one element then return it as min and max both*/
    if (n == 1) {
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

    for (i = 2; i<n; i++) {
        if (arr[i] >  minmax.max)
            minmax.max = arr[i];
        else if (arr[i] <  minmax.min)
            minmax.min = arr[i];
    }
    return minmax;
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

int main() {
    double **data, *data_storage;

    data_storage = (double *) malloc(N_ROWS * N_COLS * sizeof(double));
    data = (double **) malloc(N_ROWS * sizeof(double *));
    for (int i = 0; i < N_ROWS; ++i) {
        data[i] = &data_storage[i*N_COLS];
    }

    // Fill the matrix from csv
    string filename = "../Iris.csv";
    loadData(filename, data);

//  PRINT MATRIX
    for (int i = 0; i < N_ROWS; ++i) {
        for(int j = 0; j < N_COLS; j++) {
            cout << data[i][j] << " ";
        }
       cout << "\n";
    }

    //Standardization
    struct stats info = getStats(data_storage, N_ROWS * N_COLS);

    for (int i = 0; i < N_ROWS * N_COLS; ++i) {
        data_storage[i] = (data_storage[i] - info.mean)/info.stdev;
    }

/*
    // Normalization - MaxMinScaler
    struct pair minmax = getMinMax(data_storage, N_ROWS * N_COLS);

    for (int i = 0; i < N_ROWS * N_COLS; ++i) {
        data_storage[i] = (data_storage[i] - minmax.min)/(minmax.max - minmax.min);
    }
*/

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
    // PRINT PEARSON TABLE
    printf("----------------PEARSON-----------------\n");
    for (int i = 0; i < N_ROWS; ++i) {
        for(int j = 0; j < N_ROWS; j++) {
            cout << pearson[i][j] << " ";
        }
        cout << "\n";
    }

    //-------------------------------------------------------------------TEST
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
    double **corr, *corr_storage, **uncorr, *uncorr_storage;
    corr = (double **) malloc(corr_vars * sizeof(double *));
    uncorr = (double **) malloc(uncorr_vars * sizeof(double *));

    corr_vars = 0, uncorr_vars = 0;

    loadData(filename, data);

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

    printf("----------------CORR-----------------\n");
    for (int i = 0; i < corr_vars; ++i) {
        for(int j = 0; j < N_COLS; j++) {
            cout << corr[i][j] << " ";
        }
        cout << "\n";
    }

    printf("----------------UNCORR-----------------\n");
    for (int i = 0; i < uncorr_vars; ++i) {
        for(int j = 0; j < N_COLS; j++) {
            cout << uncorr[i][j] << " ";
        }
        cout << "\n";
    }

    real_2d_array x, eigvecs;
    x.setlength(corr_vars, N_COLS);
    for (int i = 0; i < corr_vars; ++i) {
        for(int j = 0; j < N_COLS; j++) {
            x[i][j] = corr[i][j];
        }
    }

    real_1d_array eigvals;


    return 0;
}