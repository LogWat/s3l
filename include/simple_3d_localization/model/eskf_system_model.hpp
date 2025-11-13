#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <simple_3d_localization/model/system_model.hpp>

namespace s3l::model
{

class ESKFSystemModel: public SystemModel {
public:
    virtual VectorXt f(const VectorXt &state, const VectorXt &error_state, const VectorXt &control) const = 0;
    virtual VectorXt h(const VectorXt &state) const = 0;
    virtual MatrixXt getJacobianH(const VectorXt &state) const = 0;
    virtual MatrixXt getJacobianFx(const VectorXt &state, const VectorXt &control) const = 0;
    virtual MatrixXt getJacobianFi() const = 0;
    virtual MatrixXt getJacobianQi() const = 0;
    virtual MatrixXt getJacobianG(const VectorXt &error_state) const = 0;
    virtual void setIMUNoiseParameters(double acc_noise, double gyro_noise, double acc_bias_noise, double gyro_bias_noise) = 0;
};

} // namespace s3l::model
