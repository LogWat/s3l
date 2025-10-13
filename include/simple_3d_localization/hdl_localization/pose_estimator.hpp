#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include <simple_3d_localization/type.hpp>

#include <simple_3d_localization/filter/filter.hpp>
#include <simple_3d_localization/filter/ukf.hpp>
#include <simple_3d_localization/filter/ekf.hpp>

#include <simple_3d_localization/model/ukf_pose.hpp>
#include <simple_3d_localization/model/ekf_odom_pose.hpp>
#include <simple_3d_localization/model/ekf_pose.hpp>

#include <simple_3d_localization/mahalanobis.hpp>

namespace s3l
{

namespace hdl_localization 
{

/**
 * @brief scan matching-based pose estimator
 */
class PoseEstimator {
public:
    using PointT = pcl::PointXYZI;

    /**
     * @brief Constructor
     * @param registration          registration method
     * @param pos                   initial pose
     * @param quat                  initial quaternion
     * @param use_odom              use odometry or not
     * @param filter_type           filter type (UKF or EKF)
     * @param cool_time_duration    during "cool time", prediction is not performed
     * @param use_mahalanobis_gating   use mahalanobis gating or not
     * @param mahalanobis_threshold    threshold for mahalanobis gating
     */
    explicit PoseEstimator(
        std::shared_ptr<pcl::Registration<PointT, PointT>>& registration,
        const Vector3t& pos,
        const Quaterniont& quat,
        bool use_odom,
        FilterType filter_type,
        double cool_time_duration = 1.0,
        bool use_mahalanobis_gating = false,
        double mahalanobis_threshold = 33.11
    ): 
        cool_time_duration_(cool_time_duration),
        registration_(registration),
        filter_type_(filter_type),
        use_mahalanobis_(use_mahalanobis_gating),
        mahalanobis_threshold_(mahalanobis_threshold),
        use_odom_(use_odom)
    {
        last_observation_ = Matrix4t::Identity();
        last_observation_.block<3, 3>(0, 0) = quat.toRotationMatrix();
        last_observation_.block<3, 1>(0, 3) = pos;

        VectorXt mean;
        MatrixXt cov;
        int state_dim = use_odom_ ? 16 : 19;
        int input_dim = 6; // IMU: acc(3) + gyro(3), odom: vel(3) + gyro(3)
        int measurement_dim = 7;

        if (!use_odom_) {
            // 位置(3) + 速度(3) + 姿勢(4) + bias(3) + bias_gyro(3) + gravity(3) = 19
            process_noise_ = MatrixXt::Identity(19, 19);
            // TODO: パラメータ化
            process_noise_.middleRows(0, 3) *= std::pow(1.0, 2); // (1.0 m)^2 /s
            process_noise_.middleRows(3, 3) *= std::pow(1.0, 2); // (1.0 m/s)^2 /s
            process_noise_.middleRows(6, 4) *= std::pow(10.0 * M_PI / 180.0, 2); // (10 deg)^2 /s
            process_noise_.middleRows(10, 3) *= 1e-6;
            process_noise_.middleRows(13, 3) *= 1e-6;
            process_noise_.middleRows(16, 3) *= 1e-9;

            // 初期状態
            mean = VectorXt::Zero(19);
            mean.middleRows(0, 3) = pos;
            mean.middleRows(3, 3).setZero();
            mean.middleRows(6, 4) = Vector4t(quat.w(), quat.x(), quat.y(), quat.z()).normalized();
            mean.middleRows(10, 3).setZero();
            mean.middleRows(13, 3).setZero();
            mean.middleRows(16, 3) = Vector3t(0, 0, -9.80665f);

            cov = MatrixXt::Identity(19, 19) * 0.01;
        } else {
            // odom_system の stateベクトルの次元
            // 位置(3) + 速度(3) + 姿勢(4) + bias_速度(3) + bias_gyro(3) = 16
            process_noise_ = MatrixXt::Identity(16, 16);
            process_noise_.middleRows(0, 3) *= std::pow(0.1, 2); // (0.1 m)^2 /s
            process_noise_.middleRows(3, 3) *= std::pow(0.1, 2); // (0.1 m/s)^2 /s
            process_noise_.middleRows(6, 4) *= std::pow(10.0 * M_PI / 180.0, 2); // (10 deg)^2 /s
            process_noise_.middleRows(10, 3) *= 1e-6;
            process_noise_.middleRows(13, 3) *= 1e-6;

            // 初期状態
            mean = VectorXt::Zero(16);
            mean.middleRows(0, 3) = pos;
            mean.middleRows(3, 3).setZero();
            mean.middleRows(6, 4) = Vector4t(quat.w(), quat.x(), quat.y(), quat.z()).normalized();
            mean.middleRows(10, 3).setZero();
            mean.middleRows(13, 3).setZero();

            cov = MatrixXt::Identity(16, 16) * 0.01;
        }

        // 位置(3) + 姿勢(4) = 7
        measurement_noise_ = MatrixXt::Identity(7, 7);
        measurement_noise_.middleRows(0, 3) *= 0.01;
        measurement_noise_.middleRows(3, 4) *= 0.01;
        translation_noise_floor_ = 0.01;
        rotation_noise_floor_ = 0.01;



        if (filter_type_ == FilterType::UKF) {
            ukf_system_model_ = std::make_unique<model::UKFPoseSystemModel>();
            filter_ = std::make_unique<filter::UnscentedKalmanFilterX>(
                *ukf_system_model_, 19, input_dim, measurement_dim, process_noise_, measurement_noise_, mean, cov
            );
        } else if (filter_type_ == FilterType::EKF) {
            if (use_odom_)  ekf_system_model_ = std::make_unique<model::EKFOdomPoseSystemModel>();
            else ekf_system_model_ = std::make_unique<model::EKFPoseSystemModel>();
            filter_ = std::make_unique<filter::ExtendedKalmanFilterX>(
                *ekf_system_model_, state_dim, mean, cov, process_noise_
            );
        }
        filter_->setMean(mean);
        filter_->setMeasurementNoise(measurement_noise_);
    }

    ~PoseEstimator() {}

    /**
     * @brief predict
     * @param stamp    timestamp
     * @param u       control input (acc(3) + gyro(3) or vel(3) + gyro(3))
     */
    void predict(const rclcpp::Time& stamp, const VectorXt& u) {
        if (init_stamp_ == rclcpp::Time()) {
            init_stamp_ = stamp;
        }
        if ((stamp - init_stamp_).seconds() < cool_time_duration_ || prev_stamp_ == rclcpp::Time() || (stamp - prev_stamp_).seconds() < 0.01) {
            prev_stamp_ = stamp;
            return;
        }

        double dt = (stamp - prev_stamp_).seconds();
        prev_stamp_ = stamp;

        filter_->setDt(dt);
        filter_->setProcessNoise(process_noise_ * dt);

        filter_->predict(u);

        auto& state_after = const_cast<VectorXt&>(filter_->getState());
        Eigen::Map<Quaterniont> q_pred(const_cast<SystemType*>(&state_after[6]));
        if (std::isfinite(q_pred.w()) && q_pred.norm() > 1e-8f) q_pred.normalize();
        else q_pred = Quaterniont::Identity();
    }


    /**
     * @brief predict
     * @param stamp    timestamp
     * @param acc      acceleration
     * @param gyro     angular velocity
     */
    void predict(const rclcpp::Time& stamp, const Vector3t& acc, const Vector3t& gyro) {
        if (init_stamp_ == rclcpp::Time()) {
            init_stamp_ = stamp;
        }
        if ((stamp - init_stamp_).seconds() < cool_time_duration_ || prev_stamp_ == rclcpp::Time() || (stamp - prev_stamp_).seconds() < 0.01) {
            prev_stamp_ = stamp;
            return;
        }

        double dt = (stamp - prev_stamp_).seconds();
        prev_stamp_ = stamp;

        filter_->setDt(dt);
        filter_->setProcessNoise(process_noise_ * dt);

        VectorXt u(6); u.head<3>() = acc; u.tail<3>() = gyro;
        filter_->predict(u);

        auto& state_after = const_cast<VectorXt&>(filter_->getState());
        Eigen::Map<Quaterniont> q_pred(const_cast<SystemType*>(&state_after[6]));
        if (std::isfinite(q_pred.w()) && q_pred.norm() > 1e-8f) q_pred.normalize();
        else q_pred = Quaterniont::Identity();
    }

    /**
     * @brief correct
     * @param cloud   input cloud
     * @return cloud aligned to the globalmap
     */
    pcl::PointCloud<PointT>::Ptr correct(const rclcpp::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
        if (init_stamp_ == rclcpp::Time()) {
            init_stamp_ = stamp;
        }

        last_correct_stamp_ = stamp;

        Matrix4t prev_observation = last_observation_;
        Matrix4t no_guess = prev_observation;
        Matrix4t imu_guess;
        Matrix4t init_guess = Matrix4t::Identity();

        init_guess = imu_guess = matrix();

        pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
        registration_->setInputSource(cloud);
        registration_->align(*aligned, init_guess); // (NDT_OMP, GICP, etc.)

        Matrix4t trans = registration_->getFinalTransformation();
        bool converged = registration_->hasConverged();

        if (!converged || !trans.allFinite()) {
            RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
                        "Alignment failed (converged=%d, finite=%d). Using init guess.",
                        converged ? 1 : 0, trans.allFinite() ? 1 : 0);
            trans = init_guess;
            return aligned;
        }

        Vector3t p = trans.block<3, 1>(0, 3);
        Quaterniont q(trans.block<3, 3>(0, 0));

        if (quat().coeffs().dot(q.coeffs()) < 0.0f) q.coeffs() *= -1.0f; // quaternionの符号を合わせる

        VectorXt observation(7);
        observation.middleRows(0, 3) = p;
        observation.middleRows(3, 4) = Vector4t(q.w(), q.x(), q.y(), q.z()).normalized();
        last_observation_ = trans;

       // --------- Mahalanobis Gate (6DOF) ----------
       if (use_mahalanobis_) {
            Eigen::Matrix<double,6,1> r6;
            const auto& s = filter_->getState();
            Eigen::Quaterniond q_pred(s[6], s[7], s[8], s[9]);
            Eigen::Quaterniond q_obs(observation[3], observation[4], observation[5], observation[6]);
            if (q_pred.coeffs().dot(q_obs.coeffs()) < 0.0) q_obs.coeffs() *= -1.0;
            q_pred.normalize(); q_obs.normalize();

            // 並進誤差(3) + 回転誤差(3)
            r6.head<3>() = (observation.head<3>() - s.head<3>()).cast<double>(); // 並進誤差
            Eigen::Quaterniond q_err = q_obs * q_pred.conjugate();
            Eigen::AngleAxisd aa(q_err);
            Eigen::Vector3d rv = aa.axis() * aa.angle();                         // 回転誤差 (4->3)
            if (!rv.allFinite()) rv.setZero();
            r6.tail<3>() = rv;

            Eigen::Matrix3d Jr_inv = rightJacobianInverseSO3(rv);                    // quatの共分散をSO(3)の接空間に変換するためのヤコビアン
            Eigen::Matrix<double,6,Eigen::Dynamic> H(6, filter_->getState().size()); // 観測モデルのヤコビアン
            H.setZero();
            H.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
            H.block<3,3>(3,7) = Jr_inv * 2.0 * Eigen::Matrix3d::Identity();

            Eigen::Matrix<double,6,6> R6 = Eigen::Matrix<double,6,6>::Zero();   // 観測ノイズの共分散行列 (6x6)
            const auto& R7 = measurement_noise_;
            if (R7.rows() >= 7 && R7.cols() >= 7) {
                R6.topLeftCorner<3,3>() = R7.topLeftCorner(3,3).cast<double>();
                Eigen::Matrix4d quat_cov = R7.block(3,3,4,4).cast<double>();
                Eigen::Matrix3d rot_block = quat_cov.block<3,3>(1,1) * 4.0;
                R6.bottomRightCorner<3,3>() = 0.5 * (rot_block + rot_block.transpose());
            } else {
                R6.setIdentity();
                R6.topLeftCorner<3,3>() *= translation_noise_floor_;
                R6.bottomRightCorner<3,3>() *= rotation_noise_floor_;
            }

            Eigen::Matrix<double,6,6> S = H * filter_->getCovariance().cast<double>() * H.transpose() + R6;
            for (int i = 0; i < 6; ++i) if (S(i, i) < 1e-12) S(i, i) += 1e-9; // 対角成分の数値不安定化防止

            Eigen::VectorXd r6v = r6;
            Eigen::VectorXd zero = Eigen::VectorXd::Zero(6);
            last_mahalanobis_d2_ = squaredMahalanobis(r6v, zero, S); // 観測誤差のマハラノビス距離の2乗

            if (last_mahalanobis_d2_ > mahalanobis_threshold_) {
                RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
                            "Mahalanobis reject d2=%.3f > thresh=%.3f (df=6)",
                            last_mahalanobis_d2_, mahalanobis_threshold_);
                consecutive_reject_count_++;
                wo_pred_error_ = no_guess.inverse() * trans;
                imu_pred_error_ = imu_guess.inverse() * trans;
                return aligned;
            }
        }
        // --------------------------------------------

        filter_->setMeasurementNoise(measurement_noise_);
        wo_pred_error_ = no_guess.inverse() * trans;
        filter_->correct(observation);

        auto& state_after = const_cast<VectorXt&>(filter_->getState());
        Eigen::Map<Quaterniont> q_corr(const_cast<SystemType*>(&state_after[6]));
        if (std::isfinite(q_corr.w()) && q_corr.norm() > 1e-8f) q_corr.normalize();
        else q_corr = Quaterniont::Identity();

        imu_pred_error_ = imu_guess.inverse() * trans;
        last_observation_ = trans;
        consecutive_reject_count_ = 0;
        last_correct_stamp_ = stamp;
        return aligned;
    }

    /* getters */
    rclcpp::Time last_correct_time() const {
        return last_correct_stamp_;
    }
    Vector3t pos() const {
        const auto& s = filter_->getState();
        return { s[0], s[1], s[2] };
    }
    Vector3t vel() const {
        const auto& s = filter_->getState();
        return { s[3], s[4], s[5] };
    }
    Quaterniont quat() const {
        const auto& s = filter_->getState();
        return Quaterniont(s[6], s[7], s[8], s[9]).normalized();
    }
    Matrix4t matrix() const {
        Matrix4t mat = Matrix4t::Identity();
        mat.block<3, 3>(0, 0) = quat().toRotationMatrix();
        mat.block<3, 1>(0, 3) = pos();
        return mat;
    }
    const std::optional<Matrix4t>& wo_prediction_error() const {
        return wo_pred_error_; // スキャンマッチだけの相対変位
    }
    const std::optional<Matrix4t>& imu_prediction_error() const {
        return imu_pred_error_; // 予測と観測の残差
    }

    void useIMUAcc(bool use) {
        use_imu_acc_ = use; // TODO: implement
    }


    /* utils */
    void initializeWithBiasAndGravity(const Vector3t& gravity, const Vector3t& accel_bias, const Vector3t& gyro_bias) {
        VectorXt mean = filter_->getState();
        // IMU mode only
        if (mean.size() != 19) {
            RCLCPP_ERROR(rclcpp::get_logger("PoseEstimator"), "State size is invalid (%ld)", mean.size());
            return;
        }
        mean.middleRows(10, 3) = accel_bias;
        mean.middleRows(16, 3) = gravity;
        mean.middleRows(13, 3) = gyro_bias;
        filter_->setMean(mean);
    }

private:
    // SO(3)の右ヤコビアンの逆元
    static Eigen::Matrix3d rightJacobianInverseSO3(const Eigen::Vector3d& phi) {
        const double theta = phi.norm();
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        const Eigen::Matrix3d A = Sophus::SO3d::hat(phi);
        if (theta < 1e-5) return I + 0.5 * A + (1.0 / 12.0) * A * A;

        const double half = 0.5 * theta;
        const double cot_half = std::cos(half) / std::sin(half);
        const double coeff = (1.0 - 0.5 * theta * cot_half) / (theta * theta);
        return I + 0.5 * A + coeff * A * A; // (式. 184 in "Quanternion kinematics for the error-state Kalman filter")
    }


    rclcpp::Time init_stamp_;             // when the estimator was initialized
    rclcpp::Time prev_stamp_;             // when the estimator was updated last time
    rclcpp::Time last_correct_stamp_;      // when the estimator performed the correct step
    double cool_time_duration_;

    std::shared_ptr<pcl::Registration<PointT, PointT>> registration_;

    MatrixXt process_noise_{MatrixXt::Identity(19, 19)};
    MatrixXt measurement_noise_;

    Matrix4t last_observation_;
    std::optional<Matrix4t> wo_pred_error_;
    std::optional<Matrix4t> imu_pred_error_;

    FilterType filter_type_;
    std::unique_ptr<model::UKFPoseSystemModel> ukf_system_model_;
    std::unique_ptr<model::EKFSystemModel> ekf_system_model_;
    std::unique_ptr<filter::KalmanFilterX> filter_;

    bool use_mahalanobis_;
    double last_mahalanobis_d2_{0.0};
    double mahalanobis_threshold_; // 99% confidence interval for chi-squared distribution with 6 DOF
    // (calculated by scipy.stats.chi2.ppf(0.99, df=6))

    int consecutive_reject_count_{0};
    const int init_consecutive_reject_{5};

    bool use_odom_;
    bool use_imu_acc_{true};

    // matching evaluator
    double translation_noise_floor_{0.01};       // 並進誤差の下限 (m^2)
    double rotation_noise_floor_{0.01};          // 回転誤差の下限 (rad^2)
};

} // namespace hdl_localization
} // namespace s3l

