#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>
#include <simple_3d_localization/filter/filter.hpp>
#include <simple_3d_localization/model/eskf_system_model.hpp>

namespace s3l::filter {

class ErrorStateKalmanFilterX : public KalmanFilterX {
public:
    explicit ErrorStateKalmanFilterX(
        model::ESKFSystemModel& model,
        const int state_dim,                    // 誤差状態の次元
        const VectorXt& initial_nominal_state,  // 名目状態の初期値
        const MatrixXt& initial_cov,
        const double acc_noise,
        const double gyro_noise,
        const double acc_bias_noise,
        const double gyro_bias_noise)
        : state_dim_(state_dim), model_(model), nominal_state_(initial_nominal_state), P_(initial_cov) 
    {
        model_.setIMUNoiseParameters(acc_noise, gyro_noise, acc_bias_noise, gyro_bias_noise);
        error_state_ = VectorXt::Zero(state_dim_);
        state_ = nominal_state_;
    }

    /**
     * @brief Predict the next state with control input
     * @param dt Time step for prediction
     * @param control Control input vector
     */
    void predict(const double dt, const VectorXt& control) {
        double dt_c = std::max(std::min(dt, 1.0), 1e-6); // Clamp dt to avoid instability
        model_.setDt(dt_c);
        VectorXt nominal_state = model_.f(nominal_state_, error_state_, control);
        MatrixXt Fx = model_.getJacobianFx(nominal_state_, control);
        MatrixXt Fi = model_.getJacobianFi();
        MatrixXt Qi = model_.getJacobianQi();
        MatrixXt P_pred = Fx * P_ * Fx.transpose() + Fi * Qi * Fi.transpose();
        nominal_state_ = nominal_state;
        P_ = P_pred;
    }

    /**
     * @brief Correct the state with measurement
     * @param measurement Measurement vector
     */
    void correct(const VectorXt& measurement, const MatrixXt& V) {
        VectorXt predicted_measurement = model_.h(nominal_state_);
        MatrixXt H = model_.getJacobianH(nominal_state_);   
        MatrixXt K = P_ * H.transpose() * (H * P_ * H.transpose() + V).inverse();
        VectorXt error = K * (measurement - predicted_measurement);
        P_ = (MatrixXt::Identity(state_dim_, state_dim_) - K * H) * P_;
        updateState();
        resetFilter();
    }

    // KalmanFilterX<T> overrides
    void setDt(double dt) override {
        double dt_c = std::max(std::min(dt, 1.0), 1e-6);
        model_.setDt(dt_c);
    }
    void setMean(const VectorXt& mean) override { 
        nominal_state_ = mean;
        error_state_.setZero();
        updateState();
    }
    void setProcessNoise(const MatrixXt& process_noise) override {
        // Not used directly; process noise is set via IMU noise parameters
    }
    void setMeasurementNoise(const MatrixXt& measurement_noise) override {
        V_ = measurement_noise;
    }
    void predict(const VectorXt& control) override { predict(0.01, control); }
    void correct(const VectorXt& measurement) override { 
        correct(measurement, V_);
    }

    [[nodiscard]] const VectorXt& getState() const override { return state_; }
    [[nodiscard]] const MatrixXt& getCovariance() const override { return P_; }

private:
    void resetFilter() {
        const MatrixXt G = model_.getJacobianG(error_state_);
        P_ = G * P_ * G.transpose();
        error_state_.setZero();
    }

    void updateState() {
        state_.head(3) = nominal_state_.head(3) + error_state_.head(3); // position
        state_.segment(3, 3) = nominal_state_.segment(3, 3) + error_state_.segment(3, 3); // velocity
        Quaterniont q_nominal(nominal_state_(6), nominal_state_(7), nominal_state_(8), nominal_state_(9));
        Vector3t delta_theta = error_state_.segment(6, 3);
        Quaterniont q_error = Sophus::SO3<SystemType>::exp(delta_theta).unit_quaternion();
        Quaterniont q_updated = (q_nominal * q_error).normalized();
        state_(6) = q_updated.w();
        state_(7) = q_updated.x();
        state_(8) = q_updated.y();
        state_(9) = q_updated.z();
        state_.segment(10, 3) = nominal_state_.segment(10, 3) + error_state_.segment(9, 3); // bias acc
        state_.segment(13, 3) = nominal_state_.segment(13, 3) + error_state_.segment(12, 3); // bias gyro
        state_.segment(16, 3) = nominal_state_.segment(16, 3) + error_state_.segment(15, 3);  // gravity
    }


    int state_dim_;
    model::ESKFSystemModel& model_;
    VectorXt state_;
    VectorXt nominal_state_;
    VectorXt error_state_;
    MatrixXt P_;
    MatrixXt V_; // measurement noise covariance
};

} // namespace s3l::filter
