package com.example.ertos;

public class KalmanFilter {
    private double x;
    private double P;
    private double Q;
    private double R;

    public KalmanFilter(double processNoise, double measurementNoise, double initialEstimate, double initialUncertainty) {
        this.Q = processNoise;
        this.R = measurementNoise;
        this.x = initialEstimate;
        this.P = initialUncertainty;
    }


    public void predict() {
        P += Q;
    }


    public void update(double measurement) {
        double K = P / (P + R);
        x += K * (measurement - x);
        P *= (1 - K);
    }

    public double getEstimate() {
        return x;
    }
}

