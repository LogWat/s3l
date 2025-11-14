#pragma once

#include <Eigen/Core>

using SystemType = float;
using VectorXt = Eigen::Matrix<SystemType, Eigen::Dynamic, 1>;
using MatrixXt = Eigen::Matrix<SystemType, Eigen::Dynamic, Eigen::Dynamic>;
using Vector3t = Eigen::Matrix<SystemType, 3, 1>;
using Vector4t = Eigen::Matrix<SystemType, 4, 1>;
using Matrix4t = Eigen::Matrix<SystemType, 4, 4>;
using Quaterniont = Eigen::Quaternion<SystemType>;

enum FilterType {
    UKF,
    EKF,
    ESKF
};

struct ImuData {
    double timestamp;
    Eigen::Vector3d angular_velocity;
    Eigen::Vector3d linear_acceleration;
};

struct ImuInitializationResult {
    bool success = false;
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d gravity_vec = Eigen::Vector3d(0, 0, -9.80665);
};
