/**
 * @file ekf_odom_pose.hpp
 * @brief Extended Kalman Filter (EKF) system model using odometry for 3D pose estimation
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

#include <simple_3d_localization/type.hpp>
#include <simple_3d_localization/model/ekf_system_model.hpp>

namespace s3l::model {

/**
 * state: [px, py, pz, vx, vy, vz, qw, qx, qy, qz] (10)
 * measurement: [px, py, pz, qw, qx, qy, qz] (7)
 * control: [vel_x, vel_y, vel_z, gyro_x, gyro_y, gyro_z]  (6)
 *  - vel_* はbody座標系速度（LIO推定など）
 *  - gyro_* はbody角速度
 */
class EKFOdomPoseSystemModel : public EKFSystemModel {
public:
    explicit EKFOdomPoseSystemModel() {}

    // f(x)
    VectorXt f(const VectorXt& state) const override {
        VectorXt next_state(state.size());
        const Vector3t p = state.middleRows(0, 3);
        const Vector3t v = state.middleRows(3, 3);
        Quaterniont q(state(6), state(7), state(8), state(9));
        q.normalize();

        next_state.middleRows(0, 3) = p + v * dt_;
        next_state.middleRows(3, 3) = v;
        next_state.middleRows(6, 4) = Vector4t(q.w(), q.x(), q.y(), q.z());
        return next_state;
    }

    // f(x, u)
    VectorXt f(const VectorXt& state, const VectorXt& control) const override {
        VectorXt next_state(state.size());

        const Vector3t p = state.middleRows(0, 3);
        const Vector3t v = state.middleRows(3, 3);
        Quaterniont q(state(6), state(7), state(8), state(9));
        q.normalize();

        Vector3t vel_body = control.head<3>();
        Vector3t gyro     = control.tail<3>();

        Vector3t p_next = p + v * dt_;
        
        MatrixXt R = q.toRotationMatrix(); // body -> world
        Vector3t vel_world = R * vel_body;
        Vector3t v_next = vel_world;

        Vector3t theta = gyro * dt_;
        Sophus::SO3<SystemType> dR = Sophus::SO3<SystemType>::exp(theta);
        Sophus::SO3<SystemType> Rcurr(q.toRotationMatrix());
        Sophus::SO3<SystemType> Rnext = Rcurr * dR;
        Quaterniont q_next(Rnext.unit_quaternion());

        q_next.normalize();

        next_state.middleRows(0, 3) = p_next;
        next_state.middleRows(3, 3) = v_next;
        next_state.middleRows(6, 4) = Vector4t(q_next.w(), q_next.x(), q_next.y(), q_next.z());
        return next_state;
    }

    // 観測モデル h(x)
    VectorXt h(const VectorXt& state) const override {
        VectorXt meas(7);
        meas.middleRows(0, 3) = state.middleRows(0, 3);
        Quaterniont q(state(6), state(7), state(8), state(9));
        q.normalize();
        meas.middleRows(3, 4) = Vector4t(q.w(), q.x(), q.y(), q.z());
        return meas;
    }

    // F = ∂f/∂x （u 未使用版）: 簡易な定数速度モデル
    MatrixXt stateTransitionJacobian(const VectorXt& state) const override {
        return stateTransitionJacobian(state, VectorXt::Zero(6));
    }

    // F = ∂f/∂x （制御入力利用）
    MatrixXt stateTransitionJacobian(const VectorXt& state, const VectorXt& control) const override {
        MatrixXt F = MatrixXt::Identity(state.size(), state.size());
        const Quaterniont q(state(6), state(7), state(8), state(9));
        const MatrixXt R = q.toRotationMatrix();
        const Vector3t vel_body = control.head<3>();
        const Vector3t gyro     = control.tail<3>();

        // Position Jacobian
        F.block<3, 3>(0, 0) = MatrixXt::Identity(3, 3);
        F.block<3, 3>(0, 3) = MatrixXt::Identity(3, 3) * dt_;
    
        // Velocity Jacobian
        F.block<3, 4>(3, 6) = dR_dq(q, vel_body);

        // Orientation Jacobian
        F.block<4, 4>(6, 6) = dq_dqp(gyro, dt_);

        return F;
    }

    // H = ∂h/∂x
    MatrixXt measurementJacobian(const VectorXt& state) const override {
        (void)state;
        MatrixXt H = MatrixXt::Zero(7, 10);
        H.block<3,3>(0,0) = MatrixXt::Identity(3,3);   // position
        H.block<4,4>(3,6) = MatrixXt::Identity(4,4);   // quaternion
        return H;
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
};

} // namespace s3l::model