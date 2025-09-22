#pragma once

#include <Eigen/Dense>

#include <simple_3d_localization/model/system_model.hpp>

namespace s3l::model
{

class EKFSystemModel: public SystemModel {
public:
    virtual MatrixXt measurementJacobian(const VectorXt& state) const = 0;
    virtual MatrixXt stateTransitionJacobian(const VectorXt& state) const = 0;
    virtual MatrixXt stateTransitionJacobian(const VectorXt& state, const VectorXt& control) const = 0;
};

} // namespace s3l::model
