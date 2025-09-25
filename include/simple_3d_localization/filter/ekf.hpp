#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <simple_3d_localization/filter/filter.hpp>
#include <simple_3d_localization/model/ekf_system_model.hpp>

namespace s3l::filter {

class ExtendedKalmanFilterX : public KalmanFilterX {
public:
    explicit ExtendedKalmanFilterX(
        model::EKFSystemModel& model,
        const int state_dim,
        const VectorXt& initial_state,
        const MatrixXt& initial_cov,
        const MatrixXt& process_noise)
        : state_dim_(state_dim), model_(model), X_(initial_state), P_(initial_cov), Q_(process_noise) {}

    /**
     * @brief Predict the next state with control input
     * @param dt Time step for prediction
     * @param control Control input vector
     */
    void predict(const double dt, const VectorXt& control) {
        double dt_c = std::max(std::min(dt, 1.0), 1e-6); // Clamp dt to avoid instability
        model_.setDt(dt_c);
        VectorXt X_pred = model_.f(X_, control);
        MatrixXt F = model_.stateTransitionJacobian(X_, control);
        MatrixXt P_pred = F * P_ * F.transpose() + Q_ * dt_c;
        X_ = X_pred;
        P_ = P_pred;
    }

    /**
     * @brief Correct the state with a measurement
     * @param measurement The measurement vector
     * @param measurement_noise The measurement noise covariance matrix
     */
    void correct(const VectorXt& measurement, const MatrixXt& measurement_noise) {
        VectorXt measurement_pred = model_.h(X_);
        MatrixXt H = model_.measurementJacobian(X_);
        VectorXt y = measurement - measurement_pred;
        MatrixXt S = H * P_ * H.transpose() + measurement_noise;
        MatrixXt K = P_ * H.transpose() * S.inverse(); // Kalman gain
        X_ += K * y; // Correct state estimate
        // Josephson correct for covariance
        MatrixXt I = MatrixXt::Identity(state_dim_, state_dim_);
        P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * measurement_noise * K.transpose();
    }

    // KalmanFilterX<T> overrides
    void setDt(double dt) override {
        last_dt_ = std::max(std::min(dt, 1.0), 1e-6);
        model_.setDt(last_dt_);
    }
    void setMean(const VectorXt& mean) override { X_ = mean; }
    void setProcessNoise(const MatrixXt& process_noise) override {
        Q_ = process_noise;
    }
    void setMeasurementNoise(const MatrixXt& measurement_noise) override {
        R_ = measurement_noise;
    }
    void predict(const VectorXt& control) override { predict(last_dt_, control); }
    void correct(const VectorXt& measurement) override { correct(measurement, R_); }

    [[nodiscard]] const VectorXt& getState() const override { return X_; }
    [[nodiscard]] const MatrixXt& getCovariance() const override { return P_; }

private:
    const int state_dim_;
    model::EKFSystemModel& model_; // System model
    VectorXt X_; // State vector
    MatrixXt P_; // State covariance matrix
    MatrixXt Q_; // Process noise covariance

    MatrixXt R_;       // measurement noise (for interface-based correct)
    double   last_dt_{0.01};
};

} // namespace s3l::filter
