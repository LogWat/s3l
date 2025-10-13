/**
 * @file 3d_pose_ekf.hpp
 * @brief Extended Kalman Filter (EKF) system model for 3D pose estimation
 * @author LogWat
 */


#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/so3.hpp>

#include <simple_3d_localization/model/ekf_system_model.hpp>

namespace s3l::model {

/**
 * @brief System model for Extended Kalman Filter (EKF) in 3D pose estimation
 * @note state = [x, y, z, vx, vy, vz, qw, qx, qy, qz,
 *          bias_acc_x, bias_acc_y, bias_acc_z,
 *         bias_gyro_x, bias_gyro_y, bias_gyro_z,
 *          gravity_x, gravity_y, gravity_z] (19)
 * @note measurement = [x, y, z, qw, qx, qy, qz]
 */
class EKFPoseSystemModel : public EKFSystemModel {

public:
    explicit EKFPoseSystemModel() {}

    /**
     * @brief Predict the next state based on the current state and time step
     * @param state Current state vector
     * @param dt Time step for prediction
     * @return Next state vector
     */
    VectorXt f(const VectorXt& state) const override {
        VectorXt next_state(state.size());

        Vector3t pt = state.middleRows(0, 3);
        Vector3t vt = state.middleRows(3, 3);
        Quaterniont qt(state(6), state(7), state(8), state(9));
        qt.normalize(); // Ensure quaternion is normalized

        next_state.head(3) = pt + vt * dt_;
        next_state.segment(3, 3) = vt;
        next_state(6) = qt.w(); next_state(7) = qt.x(); next_state(8) = qt.y(); next_state(9) = qt.z();
        next_state.segment(10, 3) = state.segment(10, 3);
        next_state.segment(13, 3) = state.segment(13, 3);
        next_state.segment(16, 3) = state.segment(16, 3);

        return next_state;
    }

    /**
     * @brief Predict the next state with control input
     * @param state Current state vector
     * @param control Control input vector
     * @param dt Time step for prediction
     * @return Next state vector
     * */
    VectorXt f(const VectorXt& state, const VectorXt& control) const override {
        VectorXt next_state(state.size());

        const Vector3t pt = state.middleRows(0, 3);
        const Vector3t vt = state.middleRows(3, 3);
        const Quaterniont qt(state(6), state(7), state(8), state(9)); 
        const MatrixXt R = qt.toRotationMatrix();
        const Vector3t bias_acc = state.middleRows(10, 3);
        const Vector3t bias_gyro = state.middleRows(13, 3);
        const Vector3t gravity = state.middleRows(16, 3);
        
        // Control input is assumed to be [ax, ay, az, wx, wy, wz]
        Vector3t raw_acc(control.head(3));
        Vector3t raw_gyro(control.tail(3));
        Vector3t acc_global = R * (raw_acc - bias_acc) + gravity; // Transform acceleration to global frame
        Vector3t gyro = raw_gyro - bias_gyro;

        // Update position and velocity based on control input
        next_state.head(3) = pt + vt * dt_ + 0.5 * acc_global * dt_ * dt_;
        // next_state.head(3) = pt + vt * dt_;
        next_state.segment(3, 3) = vt + acc_global * dt_;
        // next_state.segment(3, 3) = vt;
        Quaterniont next_q(qt * Sophus::SO3<SystemType>::exp(gyro * dt_).matrix());
        next_q.normalize();

        next_state(6) = next_q.w();
        next_state(7) = next_q.x();
        next_state(8) = next_q.y();
        next_state(9) = next_q.z();
        next_state.segment(10, 3) = bias_acc;
        next_state.segment(13, 3) = bias_gyro;
        next_state.segment(16, 3) = gravity;

        return next_state;
    }

    /**
     * @brief Measurement function to extract pose from state
     * @param state Current state vector
     * @return Measurement vector containing position and orientation
     */
    VectorXt h(const VectorXt& state) const override {
        VectorXt measurement(7);
        measurement.head(3) = state.head(3); // Position
        measurement.segment(3, 4) = state.segment(6, 4); // Quaternion coefficients 
        return measurement;
    }

private:

    /**
     * @brief Jacobian of the measurement function
     * @param state Current state vector
     * @return Jacobian matrix of the measurement function
     */
    MatrixXt measurementJacobian(const VectorXt& state) const override {
        MatrixXt H = MatrixXt::Zero(7, state.size());
        H.block<3, 3>(0, 0) = MatrixXt::Identity(3, 3); // Position Jacobian
        H.block<4, 4>(3, 6) = MatrixXt::Identity(4, 4); // Quaternion Jacobian 
        return H;
    }

    /**
     * @brief Jacobian of the state transition function
     * @param state Current state vector
     * @param dt Time step for prediction
     * @return Jacobian matrix of the state transition function
     */
    MatrixXt stateTransitionJacobian(const VectorXt& state) const override {
        MatrixXt F = MatrixXt::Zero(state.size(), state.size());

        // Position and velocity Jacobian
        F.block<3, 3>(0, 0) = MatrixXt::Identity(3, 3);
        F.block<3, 3>(0, 7) = MatrixXt::Identity(3, 3) * dt_;

        // Quaternion Jacobian (assuming small angle approximation)
        F.block<4, 4>(3, 3) = Matrix4t::Identity();

        // Velocity Jacobian
        F.block<3, 3>(7, 7) = MatrixXt::Identity(3, 3);
        F.block<3, 3>(7, 16) = MatrixXt::Identity(3, 3) * dt_;
        // Bias Jacobian (assumed constant for simplicity)
        F.block<3, 3>(10, 10) = MatrixXt::Identity(3, 3);
        F.block<3, 3>(13, 13) = MatrixXt::Identity(3, 3);
        F.block<3, 3>(16, 16) = MatrixXt::Identity(3, 3);

        return F;
    }

    /**
     * @brief Jacobian of the state transition function with control input
     * @param state Current state vector
     * @param control Control input vector
     * @param dt Time step for prediction
     * @return Jacobian matrix of the state transition function with control input
     */
    MatrixXt stateTransitionJacobian(const VectorXt& state, const VectorXt& control) const override {
        MatrixXt F = MatrixXt::Zero(state.size(), state.size());

        const Quaterniont qt(state(6), state(7), state(8), state(9));
        const MatrixXt R = qt.toRotationMatrix();
        const Vector3t bias_acc = state.middleRows(10, 3);
        // const Vector3t bias_gyro = state.middleRows(13, 3);
        // const Vector3t gravity = state.middleRows(16, 3); 

        const Vector3t raw_acc(control.head(3));
        const Vector3t raw_gyro(control.tail(3));
        const Vector3t acc_corrected = raw_acc - bias_acc;

        const Eigen::Matrix<SystemType, 3, 4> dR_dq_result = dR_dq(qt, acc_corrected);

        // Position Jacobian
        F.block<3, 3>(0, 0) = MatrixXt::Identity(3, 3);
        F.block<3, 3>(0, 3) = MatrixXt::Identity(3, 3) * dt_;
        F.block<3, 4>(0, 6) = 0.5 * dt_ * dt_ * dR_dq_result;
        F.block<3, 3>(0, 10) = -0.5 * dt_ * dt_ * R;
        F.block<3, 3>(0, 16) = 0.5 * dt_ * dt_ * MatrixXt::Identity(3, 3);

        // Velocity Jacobian
        F.block<3, 3>(3, 3) = MatrixXt::Identity(3, 3);
        F.block<3, 4>(3, 6) = dt_ * dR_dq_result;
        F.block<3, 3>(3, 10) = -dt_ * R;
        F.block<3, 3>(3, 16) = dt_ * MatrixXt::Identity(3, 3);

        // Quaternion Jacobian
        F.block<4, 4>(6, 6) = dq_dqp(raw_gyro, dt_);
        F.block<4, 3>(6, 13) = dq_dbg(qt, dt_);

        // Bias & Gravity Jacobian
        F.block<3, 3>(10, 10) = MatrixXt::Identity(3, 3);
        F.block<3, 3>(13, 13) = MatrixXt::Identity(3, 3);
        F.block<3, 3>(16, 16) = MatrixXt::Identity(3, 3);

        return F;
    }


    /**
     * @brief Partial derivative of Ra' with respect to quaternion coefficients
     * @param qt Quaternion coefficients
     * @param acc_c Corrected acceleration
     * @return Jacobian matrix of the rotation matrix with respect to quaternion coefficients
     */
    Eigen::Matrix<SystemType, 3, 4> dR_dq(const Quaterniont& qt, const Vector3t& acc_c) const {
        Eigen::Matrix<SystemType, 3, 4> result;
        SystemType qw = qt.w(), qx = qt.x(), qy = qt.y(), qz = qt.z();
        SystemType ax = acc_c.x(), ay = acc_c.y(), az = acc_c.z();

        // ∂(R*a')/∂qw
        result.col(0) << 2 * (qw * ax - qz * ay + qy * az),
                         2 * (qw * ay + qz * ax - qx * az),
                         2 * (qw * az - qy * ax + qx * ay);
                         
        // ∂(R*a')/∂qx
        result.col(1) << 2 * (qx * ax + qy * ay + qz * az),
                         2 * (qy * ax - qx * ay - qw * az),
                         2 * (qz * ax + qw * ay - qx * az);

        // ∂(R*a')/∂qy
        result.col(2) << 2 * (-qy * ax + qx * ay + qw * az),
                         2 * (qx * ax + qy * ay + qz * az),
                         2 * (-qw * ax + qz * ay - qy * az);
                         
        // ∂(R*a')/∂qz
        result.col(3) << 2 * (-qz * ax - qw * ay + qx * az),
                         2 * (qw * ax - qz * ay + qy * az),
                         2 * (qx * ax + qy * ay + qz * az);

        return result;
    }

    /**
     * @brief Partial derivative of q(k) with respect to q(k-1)
     * @param gyro Angular velocity vector
     * @param dt Time step
     * @return Jacobian matrix of the quaternion with respect to previous quaternion
     */
    Eigen::Matrix<SystemType, 4, 4> dq_dqp(const Vector3t& gyro, const double dt) const {
        Eigen::Matrix<SystemType, 4, 4> result;
        SystemType half_dt = 0.5 * dt;
        if (gyro.norm() < 1e-8) {
            // Small angle approximation
            result = Eigen::Matrix<SystemType, 4, 4>::Identity();
        } else {
            SystemType norm = gyro.norm();
            SystemType dqw = cos(half_dt * norm);
            SystemType factor = sin(half_dt * norm) / norm;
            SystemType dqx = factor * gyro.x();
            SystemType dqy = factor * gyro.y();
            SystemType dqz = factor * gyro.z();

            result << dqw, -dqx, -dqy, -dqz,
                      dqx,  dqw,  dqz, -dqy,
                      dqy, -dqz,  dqw,  dqx,
                      dqz,  dqy, -dqx,  dqw;
        }
        return result;
    }

    /**
     * @brief Partial derivative of q with respect to bias_gyro
     * @param qt Quaternion coefficients
     * @param dt Time step
     * @return Jacobian matrix of the quaternion with respect to bias gyro
     */
    Eigen::Matrix<SystemType, 4, 3> dq_dbg(const Quaterniont& qt, const double dt) const {
        Eigen::Matrix<SystemType, 4, 3> result;
        SystemType qw = qt.w(), qx = qt.x(), qy = qt.y(), qz = qt.z();
        SystemType half_dt = 0.5 * dt;

        // ∂(q(k))/∂(bias_gyro) = [q(k-1)]L * ∂(q(dq))/∂(bias_gyro) (連鎖率)

        // 左クォータニオン積行列
        // Eigen::Matrix<SystemType, 4, 4> left_quat_mult;
        // left_quat_mult << qw, -qx, -qy, -qz,
        //                   qx,  qw, -qz,  qy,
        //                   qy,  qz,  qw, -qx,
        //                   qz, -qy,  qx,  qw; // left quaternion multiplication

        // // ∂(q(dq))/∂(bias_gyro) (0付近近似利用)
        // Eigen::Matrix<EigenT, 4, 3> dq_dq_bias;
        // dq_dq_bias << 0, 0, 0,
        //               -half_dt, 0, 0,
        //               0, -half_dt, 0,
        //               0, 0, -half_dt; // assuming small angle approximation
                      
        result << -qx, -qy, -qz,
                  qw, -qz,  qy,
                  qz,  qw, -qx,
                 -qy,  qx,  qw;

        return -half_dt * result;
    }
};


} // s3l::model
