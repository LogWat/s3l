#pragma once

#include <simple_3d_localization/model/system_model.hpp>

namespace s3l::model
{

/**
 * @brief Definition of system to be estimated by ukf
 * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, 
 * acc_bias_x, acc_bias_y, acc_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z
 * gravity_x, gravity_y, gravity_z] (19)
 */
class UKFPoseSystemModel : public SystemModel {
public:
    explicit UKFPoseSystemModel() {
        dt_ = 0.01;
    }

    VectorXt f(const VectorXt& state) const override {
        VectorXt next_state(19);

        Vector3t pt = state.middleRows(0, 3);
        Vector3t vt = state.middleRows(3, 3);
        Quaterniont qt(state(6), state(7), state(8), state(9));
        qt.normalize();

        // position
        next_state.middleRows(0, 3) = pt + dt_ * vt;
        // velocity
        next_state.middleRows(3, 3) = vt;
        // orientation
        Quaterniont qt_ = qt;

        next_state.middleRows(6, 4) << qt_.w(), qt_.x(), qt_.y(), qt_.z();
        next_state.middleRows(10, 3) = state.middleRows(10, 3);
        next_state.middleRows(13, 3) = state.middleRows(13, 3);
        next_state.middleRows(16, 3) = state.middleRows(16, 3);
        
        return next_state;
    }


    VectorXt f(const VectorXt& state, const VectorXt& control) const override {
        VectorXt next_state(19);

        Vector3t pt = state.middleRows(0, 3);
        Vector3t vt = state.middleRows(3, 3);
        Quaterniont qt(state[6], state[7], state[8], state[9]);
        qt.normalize();

        Vector3t acc_bias = state.middleRows(10, 3);
        Vector3t gyro_bias = state.middleRows(13, 3);

        Vector3t raw_acc = control.middleRows(0, 3);
        Vector3t raw_gyro = control.middleRows(3, 3);

        // position
        next_state.middleRows(0, 3) = pt + vt * dt_;

        // velocity
        Vector3t g = state.middleRows(16, 3);
        Vector3t acc_ = raw_acc - acc_bias;
        Vector3t acc = qt * acc_;
        // next_state.middleRows(3, 3) = vt + (acc + g) * dt_;
        next_state.middleRows(3, 3) = vt; // + (acc - g) * dt_;		// acceleration didn't contribute to accuracy due to large noise

        // orientation
        Vector3t gyro = raw_gyro - gyro_bias;
        Quaterniont dq(1, gyro[0] * dt_ / 2, gyro[1] * dt_ / 2, gyro[2] * dt_ / 2);
        dq.normalize();
        Quaterniont qt_ = (qt * dq).normalized();
        next_state.middleRows(6, 4) << qt_.w(), qt_.x(), qt_.y(), qt_.z();

        next_state.middleRows(10, 3) = state.middleRows(10, 3);  // constant bias on acceleration
        next_state.middleRows(13, 3) = state.middleRows(13, 3);  // constant bias on angular velocity
        next_state.middleRows(16, 3) = state.middleRows(16, 3);  // constant gravity vector

        return next_state;
    }

    // 観測モデル
    VectorXt h(const VectorXt& state) const override {
        VectorXt observation(7);
        observation.middleRows(0, 3) = state.middleRows(0, 3);
        observation.middleRows(3, 4) = state.middleRows(6, 4).normalized();

        return observation;
    }

    // double dt_; // time step
};

} // namespace s3l::model
