/**
 * @file eskf_pose.hpp
 * @brief Error State Kalman Filter (ESKF) system model for 3D pose estimation
 * @author LogWat
 * @date 2025-11-11
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/so3.hpp>

#include <simple_3d_localization/model/eskf_system_model.hpp>

namespace s3l::model {

/**
 * @brief System model for Error State Kalman Filter (ESKF) in 3D pose estimation
 * @note nominal state = [x, y, z, vx, vy, vz, qw, qx, qy, qz,
 *          bias_acc_x, bias_acc_y, bias_acc_z,
 *         bias_gyro_x, bias_gyro_y, bias_gyro_z,
 *          gravity_x, gravity_y, gravity_z] (19)
 * @note error state = [delta_x, delta_y, delta_z,
 *          delta_vx, delta_vy, delta_vz,
 *          delta_theta_x, delta_theta_y, delta_theta_z,
 *          delta_bias_acc_x, delta_bias_acc_y, delta_bias_acc_z,
 *          delta_bias_gyro_x, delta_bias_gyro_y, delta_bias_gyro_z,
 *          delta_gravity_x, delta_gravity_y, delta_gravity_z] (18)
 * @note measurement = [x, y, z, qw, qx, qy, qz]
 */
class ESKFPoseSystemModel : public ESKFSystemModel {

public:
    explicit ESKFPoseSystemModel() {}

    /* TODO: FIX */
    VectorXt f(const VectorXt& state) const override {
        VectorXt next_state(state.size());
        return next_state;
    }
    VectorXt f(const VectorXt& state, const VectorXt& control) const override {
        VectorXt next_state(state.size());
        return next_state;
    }
    VectorXt h(const VectorXt& state) const override {
        VectorXt measurement(7);
        return measurement;
    }


    /**
     * @brief State transition function with error state and control input
     * @param state Current nominal state vector
     * @param error_state Current error state vector
     * @param control Control input vector
     * @param dt Time step for prediction
     * @return Next nominal state vector
     */
    VectorXt f(const VectorXt &state, const VectorXt &error_state, const VectorXt &control) const override {
        VectorXt next_nominal_state(state.size());

        const Vector3t np = state.middleRows(0, 3);
        const Vector3t nv = state.middleRows(3, 3);
        const Quaterniont nq(state(6), state(7), state(8), state(9));
        const MatrixXt R = nq.toRotationMatrix();
        const Vector3t nba = state.middleRows(10, 3);
        const Vector3t nbg = state.middleRows(13, 3);
        const Vector3t ng = state.middleRows(16, 3);

        const Vector3t raw_acc(control.head(3));
        const Vector3t raw_gyro(control.tail(3));
        const Vector3t acc_c = raw_acc - nba;
        const Vector3t gyro_c = raw_gyro - nbg;

        next_nominal_state.head(3) = np + nv * dt_ + 0.5 * (R * acc_c + ng) * dt_ * dt_; // position
        next_nominal_state.segment(3, 3) = nv + (R * acc_c + ng) * dt_; // velocity
        Quaterniont delta_q = Sophus::SO3<SystemType>::exp(gyro_c * dt_).unit_quaternion();
        Quaterniont next_nq = (nq * delta_q).normalized();
        next_nominal_state(6) = next_nq.w();
        next_nominal_state(7) = next_nq.x();
        next_nominal_state(8) = next_nq.y();
        next_nominal_state(9) = next_nq.z();
        next_nominal_state.segment(10, 3) = nba; // bias acc
        next_nominal_state.segment(13, 3) = nbg; // bias gyro
        next_nominal_state.segment(16, 3) = ng;  // gravity

        // VectorXt next_error_state(error_state.size());
        
        // const Vector3t dp = error_state.middleRows(0, 3);
        // const Vector3t dv = error_state.middleRows(3, 3);
        // const Vector3t dtheta = error_state.middleRows(6, 3);
        // const Vector3t dba = error_state.middleRows(9, 3);
        // const Vector3t dbg = error_state.middleRows(12, 3);
        // const Vector3t dg = error_state.middleRows(15, 3);

        // next_error_state.head(3) = dp + dv * dt; // p error
        // next_error_state.segment(3, 3) = dv + ( -R * Sophus::SO3<SystemType>::hat(acc_c) * dtheta - R * dba + dg ) * dt + getVi(); // v error
        // next_error_state.segment(6, 3) = R.transpose() * Sophus::SO3<SystemType>::hat(gyro_c) * dtheta * dt - dbg * dt + getWi(); // theta error
        // next_error_state.segment(9, 3) = dba + getAi(); // bias acc error
        // next_error_state.segment(12, 3) = dbg + getOi(); // bias gyro error
        // next_error_state.segment(15, 3) = dg; // gravity error

        return next_nominal_state;
    }

    /**
     * @brief Measurement function
     * @param state Current nominal state vector
     * @return Measurement vector
     */
    VectorXt h(const VectorXt &state) const override {
        VectorXt measurement(7);
        measurement.head(3) = state.head(3); // position
        measurement(3) = state(6); // qw
        measurement(4) = state(7); // qx
        measurement(5) = state(8); // qy
        measurement(6) = state(9); // qz
        return measurement;
    }

    /**
     * @brief get the Jacobian of the measurement function
     * @param state Current nominal state vector
     * @param error_state Current error state vector
     * @return Jacobian matrix of the measurement function
     */
    MatrixXt getJacobianH(const VectorXt &state) const override {
        MatrixXt H = MatrixXt::Zero(7, 18);
        const MatrixXt Hx = getHx(state);
        const MatrixXt Hdx = getHdx(state);
        H = Hx * Hdx;
        return H;
    }

    /**
     * @brief Jacobian of the state transition function
     * @param state Current state vector
     * @return Jacobian matrix of the state transition function
     */
    MatrixXt getJacobianFx(const VectorXt &state, const VectorXt &control) const override {
        MatrixXt Fx = MatrixXt::Identity(state.size() - 1, state.size() - 1);

        const Quaterniont qt(state(6), state(7), state(8), state(9));
        const MatrixXt R = qt.toRotationMatrix();
        const Vector3t bias_acc = state.middleRows(10, 3);
        const Vector3t bias_gyro = state.middleRows(13, 3);

        const Vector3t raw_acc(control.head(3));
        const Vector3t raw_gyro(control.tail(3));
        const Vector3t acc_c = raw_acc - bias_acc;
        const Vector3t gyro_c = raw_gyro - bias_gyro;

        // Position Jacobian
        Fx.block<3, 3>(0, 3) = MatrixXt::Identity(3, 3) * dt_;

        // Velocity Jacobian
        Fx.block<3, 3>(3, 6) = Sophus::SO3<SystemType>::hat(-R * acc_c) * dt_;
        Fx.block<3, 3>(3, 9) = -R * dt_;
        Fx.block<3, 3>(3, 15) = MatrixXt::Identity(3, 3) * dt_;

        // Quaternion Jacobian
        Fx.block<3, 3>(6, 6) = Sophus::SO3<SystemType>::hat(R.transpose() * gyro_c * dt_);
        Fx.block<3, 3>(6, 12) = -MatrixXt::Identity(3, 3) * dt_;
        return Fx;
    }

    /**
     * @brief Jacobian of the perturbation
     * @param state Current state vector
     * @return Jacobian matrix of the perturbation
     */
    MatrixXt getJacobianFi() const override {
        MatrixXt Fi = MatrixXt::Zero(18, 12);
        Fi.block<12, 12>(3, 0) = MatrixXt::Identity(12, 12);
        return Fi;
    }

    /**
     * @brief Process noise covariance in the error state
     * @return Process noise covariance matrix
     */
    MatrixXt getJacobianQi() const override {
        MatrixXt Qi = MatrixXt::Zero(12, 12);
        Qi.block<3, 3>(0, 0) = getVi();
        Qi.block<3, 3>(3, 3) = getWi();
        Qi.block<3, 3>(6, 6) = getAi();
        Qi.block<3, 3>(9, 9) = getOi();
        return Qi;
    }

    /**
     * @brief Jacobian for filter initialization
     * @return Jacobian matrix for filter initialization
     */
    MatrixXt getJacobianG(const VectorXt &error_state) const override {
        MatrixXt G = MatrixXt::Identity(18, 18);
        const Vector3t dtheta = error_state.middleRows(6, 3);
        G.block<3, 3>(6, 6) = MatrixXt::Identity(3, 3) - 0.5 * Sophus::SO3<SystemType>::hat(dtheta);
        return G;
    }

    /**
     * @brief Set IMU noise parameters
     * @param acc_noise Accelerometer noise density (m/s^2/√Hz)
     * @param gyro_noise Gyroscope noise density (rad/s/√Hz)
     * @param acc_bias_noise Accelerometer bias random walk (m/s^2/√Hz)
     * @param gyro_bias_noise Gyroscope bias random walk (rad/s/√Hz)
     */
    void setIMUNoiseParameters(double acc_noise, double gyro_noise, double acc_bias_noise, double gyro_bias_noise) override {
        acc_noise_ = acc_noise;
        gyro_noise_ = gyro_noise;
        acc_bias_noise_ = acc_bias_noise;
        gyro_bias_noise_ = gyro_bias_noise;
    }

private:
    MatrixXt getVi() const {
        MatrixXt Vi = MatrixXt::Zero(3, 3);
        Vi = dt_ * dt_ * MatrixXt::Identity(3, 3) * acc_noise_;
        return Vi;
    }
    MatrixXt getWi() const {
        MatrixXt Wi = MatrixXt::Zero(3, 3);
        Wi = dt_ * dt_ * MatrixXt::Identity(3, 3) * gyro_noise_;
        return Wi;
    }
    MatrixXt getAi() const {
        MatrixXt Ai = MatrixXt::Zero(3, 3);
        Ai = dt_ * MatrixXt::Identity(3, 3) * acc_bias_noise_;
        return Ai;
    }
    MatrixXt getOi() const {
        MatrixXt Oi = MatrixXt::Zero(3, 3);
        Oi = dt_ * MatrixXt::Identity(3, 3) * gyro_bias_noise_;
        return Oi;
    }

    // ∂h/∂x
    MatrixXt getHx(const VectorXt &measurement) const {
        MatrixXt Hx = MatrixXt::Zero(7, 19);
        Hx.block<3, 3>(0, 0) = MatrixXt::Identity(3, 3); // position
        Hx.block<4, 4>(3, 6) = MatrixXt::Identity(4, 4); // quaternion
        return Hx;
    }

    // ∂X/∂dx
    MatrixXt getHdx(const VectorXt &state) const {
        MatrixXt Hdx = MatrixXt::Identity(19, 18);
        MatrixXt Qth = MatrixXt::Zero(4, 3);
        Qth << -state(7), -state(8), -state(9),
                 state(6), -state(9),  state(8),
                 state(9),  state(6), -state(7),
                -state(8),  state(7),  state(6);
        Qth *= 0.5;
        Hdx.block<4, 3>(6, 6) = Qth;
        return Hdx;
    }


    double acc_noise_ = 5e-4;  // m/s^2/√Hz
    double gyro_noise_ = 5e-5; // rad/s/√Hz
    double acc_bias_noise_ = 1e-6;  // m/s^2/√Hz
    double gyro_bias_noise_ = 1e-6; // rad/s/√Hz
};

} // s3l::model
