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
     */
    explicit PoseEstimator(
        std::shared_ptr<pcl::Registration<PointT, PointT>>& registration,
        const Vector3t& pos,
        const Quaterniont& quat,
        bool use_odom,
        FilterType filter_type,
        double cool_time_duration = 1.0
    ): 
        cool_time_duration_(cool_time_duration),
        registration_(registration),
        filter_type_(filter_type),
        use_odom_(use_odom)
    {
        last_observation_ = Matrix4t::Identity();
        last_observation_.block<3, 3>(0, 0) = quat.toRotationMatrix();
        last_observation_.block<3, 1>(0, 3) = pos;

        VectorXt mean;
        MatrixXt cov;
        int state_dim = use_odom_ ? 10 : 19;
        int input_dim = 6;
        int measurement_dim = 7;

        if (!use_odom_) {
            // pose_system の stateベクトルの次元
            // 位置(3) + 速度(3) + 姿勢(4) + bias(3) + bias_gyro(3) + gravity(3) = 19
            process_noise_ = MatrixXt::Identity(19, 19);
            process_noise_.middleRows(0, 3) *= 1.0;
            process_noise_.middleRows(3, 3) *= 1.0;
            process_noise_.middleRows(6, 4) *= 0.5;
            process_noise_.middleRows(10, 3) *= 1e-6;
            process_noise_.middleRows(13, 3) *= 1e-5;
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
            // 位置(3) + 速度(3) + 姿勢(4) = 10
            process_noise_ = MatrixXt::Identity(10, 10);
            process_noise_.middleRows(0, 3) *= 1.0;
            process_noise_.middleRows(3, 3) *= 1.0;
            process_noise_.middleRows(6, 4) *= 0.5;

            // 初期状態
            mean = VectorXt::Zero(10);
            mean.middleRows(0, 3) = pos;
            mean.middleRows(3, 3).setZero();
            mean.middleRows(6, 4) = Vector4t(quat.w(), quat.x(), quat.y(), quat.z()).normalized();

            cov = MatrixXt::Identity(10, 10) * 0.01;
        }

        // 位置(3) + 姿勢(4) = 7
        measurement_noise_ = MatrixXt::Identity(7, 7);
        measurement_noise_.middleRows(0, 3) *= 0.01;
        measurement_noise_.middleRows(3, 4) *= 0.001;


        if (filter_type_ == FilterType::UKF) {
            ukf_system_model_ = std::make_unique<model::UKFPoseSystemModel>();
            filter_ = std::make_unique<filter::UnscentedKalmanFilterX>(
                *ukf_system_model_, 19, 6, 7, process_noise_, measurement_noise_, mean, cov
            );
        } else if (filter_type_ == FilterType::EKF) {
            if (use_odom_) {
                odom_system_model_ = std::make_unique<model::EKFOdomPoseSystemModel>();
                filter_ = std::make_unique<filter::ExtendedKalmanFilterX>(
                    *odom_system_model_, state_dim, mean, cov, process_noise_
                );
            } else {
                ekf_system_model_ = std::make_unique<model::EKFPoseSystemModel>();
                filter_ = std::make_unique<filter::ExtendedKalmanFilterX>(
                    *ekf_system_model_, state_dim, mean, cov, process_noise_
                );
            }
            filter_->setMeasurementNoise(measurement_noise_);
        }
        filter_->setMean(mean);
    }

    ~PoseEstimator() {}


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

        // test mid360 coord ned?
        float coeff = (1.0 / 6.0);
        Vector3t gyro_m(gyro.x() * coeff, gyro.y() * coeff, gyro.z() * coeff);

        VectorXt u(6); u.head<3>() = acc; u.tail<3>() = gyro_m;

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

        Matrix4t no_guess = last_observation_;
        Matrix4t imu_guess;
        Matrix4t init_guess = Matrix4t::Identity();

        init_guess = imu_guess = matrix();
        // init_guess = last_observation_; // 前回の観測値を初期値とする
        // RCLCPP_INFO(rclcpp::get_logger("PoseEstimator"),
        //             "Initial guess for alignment: t = [%.3f, %.3f, %.3f], q = [%.3f, %.3f, %.3f, %.3f]",
        //             imu_guess(0, 3), imu_guess(1, 3), imu_guess(2, 3),
        //             Quaterniont(imu_guess.block<3, 3>(0, 0)).w(),
        //             Quaterniont(imu_guess.block<3, 3>(0, 0)).x(),
        //             Quaterniont(imu_guess.block<3, 3>(0, 0)).y(),
        //             Quaterniont(imu_guess.block<3, 3>(0, 0)).z());

        pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
        registration_->setInputSource(cloud);
        registration_->align(*aligned, init_guess); // 事前に設定されているregistration方法でalign (NDT_OMP, GICP, etc.)

        // return aligned;

        Matrix4t trans = registration_->getFinalTransformation();
        bool converged = registration_->hasConverged();

        if (!converged || !trans.allFinite()) {
            RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
                        "Alignment failed (converged=%d, finite=%d). Using init guess.",
                        converged ? 1 : 0, trans.allFinite() ? 1 : 0);
            trans = init_guess;
            return aligned;
        }

        // score?
        // double score = registration_->getFitnessScore();
        // RCLCPP_INFO(rclcpp::get_logger("PoseEstimator"),
        //             "Alignment succeeded (score=%.6f).", score);
        // const double max_fitness_score = 1.0; // TODO: parameterize
        // if (score > max_fitness_score) {
        //     RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
        //                 "Fitness score is too high (%.3f > %.3f). Rejecting observation.",
        //                 score, max_fitness_score);
        //     return aligned;
        // }

        Vector3t p = trans.block<3, 1>(0, 3);
        Quaterniont q(trans.block<3, 3>(0, 0));

        if (quat().coeffs().dot(q.coeffs()) < 0.0f) q.coeffs() *= -1.0f; // quaternionの符号を合わせる

        VectorXt observation(7);
        observation.middleRows(0, 3) = p;
        observation.middleRows(3, 4) = Vector4t(q.w(), q.x(), q.y(), q.z()).normalized();
        last_observation_ = trans;


       // --------- Mahalanobis Gate (6DOF) ----------
       if (use_mahalanobis_) {
            // 残差 r6 = [ Δp ; rotvec(q_obs * q_pred^{-1}) ]
            Eigen::Matrix<double,6,1> r6;
            {
                const auto& s = filter_->getState();
                // 予測姿勢
                Eigen::Quaterniond q_pred(s[6], s[7], s[8], s[9]);
                Eigen::Quaterniond q_obs(observation[3], observation[4], observation[5], observation[6]);
                if (q_pred.coeffs().dot(q_obs.coeffs()) < 0.0) q_obs.coeffs() *= -1.0;
                q_pred.normalize(); q_obs.normalize();
                // 位置差
                r6.head<3>() = (observation.head<3>() - s.head<3>()).cast<double>();
                // 回転差
                Eigen::Quaterniond q_err = q_obs * q_pred.conjugate();
                Eigen::AngleAxisd aa(q_err);
                Eigen::Vector3d rv = aa.axis() * aa.angle();
                if (!rv.allFinite()) rv.setZero();
                r6.tail<3>() = rv;
            }
            // 測定ノイズ 7x7 -> 6x6 射影（quat 部は等方化）
            Eigen::Matrix<double,6,6> R6 = Eigen::Matrix<double,6,6>::Zero();
            {
                const auto& R7 = measurement_noise_;
                if (R7.rows() >= 7 && R7.cols() >= 7) {
                    R6.topLeftCorner<3,3>() = R7.topLeftCorner(3,3).cast<double>();
                    double s = R7.block(3,3,4,4).trace() / 4.0;
                    if (!(s > 0.0)) s = 1e-6;
                    R6.bottomRightCorner<3,3>() = Eigen::Matrix3d::Identity() * s;
                } else {
                    R6.setIdentity();
                }
            }
            // H: 位置3 + quat xyz を小角近似で使用
            Eigen::Matrix<double,6,19> H = Eigen::Matrix<double,6,19>::Zero();
            H.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
            H.block<3,3>(3,7) = Eigen::Matrix3d::Identity(); // quat x,y,z
            Eigen::Matrix<double,6,6> S = H * filter_->getCovariance().cast<double>() * H.transpose() + R6;
            for (int i=0;i<6;i++) if (S(i,i) < 1e-12) S(i,i) += 1e-9; // 数値安定化

            Eigen::VectorXd r6v = r6;
            last_mahalanobis_d2_ = squaredMahalanobis(r6v, Eigen::VectorXd::Zero(6), S);

            if (last_mahalanobis_d2_ > mahalanobis_threshold_) {
                RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
                            "Mahalanobis reject d2=%.3f > thresh=%.3f (df=6)",
                            last_mahalanobis_d2_, mahalanobis_threshold_);
                // 初期状態では観測が不安定なので、rejectしても状態を更新する
                consecutive_reject_count_++;
                if (consecutive_reject_count_ > init_consecutive_reject_) {
                    wo_pred_error_ = no_guess.inverse() * registration_->getFinalTransformation();  
                    return aligned;
                }
            } 
            // else {
            //     RCLCPP_INFO(rclcpp::get_logger("PoseEstimator"),
            //                 "Mahalanobis accept d2=%.3f <= thresh=%.3f (df=6)",
            //                 last_mahalanobis_d2_, mahalanobis_threshold_);
            // }
            // --------- End Gate -------------------------
        }

        wo_pred_error_ = no_guess.inverse() * registration_->getFinalTransformation();

        filter_->correct(observation);

        auto& state_after = const_cast<VectorXt&>(filter_->getState());
        Eigen::Map<Quaterniont> q_corr(const_cast<SystemType*>(&state_after[6]));
        if (std::isfinite(q_corr.w()) && q_corr.norm() > 1e-8f) q_corr.normalize();
        else q_corr = Quaterniont::Identity();

        // DEBUG LOG
        // if (!use_odom_) {
        //     RCLCPP_INFO(rclcpp::get_logger("PoseEstimator"),
        //                 "Corrected: pos=(%.3f, %.3f, %.3f), quat=(%.3f, %.3f, %.3f, %.3f), vel=(%.3f, %.3f, %.3f), bias_acc=(%.6f, %.6f, %.6f), bias_gyro=(%.6f, %.6f, %.6f), gravity=(%.3f, %.3f, %.3f)",
        //                 state_after[0], state_after[1], state_after[2],
        //                 state_after[6], state_after[7], state_after[8], state_after[9],
        //                 state_after[3], state_after[4], state_after[5],
        //                 state_after[10], state_after[11], state_after[12],
        //                 state_after[13], state_after[14], state_after[15],
        //                 state_after[16], state_after[17], state_after[18]
        //     );
        // }

        imu_pred_error_ = imu_guess.inverse() * registration_->getFinalTransformation();
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

    void setMahalanobisThreshold(double threshold) {
        mahalanobis_threshold_ = threshold;
    }
    void useMahalanobisGating(bool use) {
        use_mahalanobis_ = use;
    }


    /* utils */
    void initializeWithBiasAndGravity(const Vector3t& gravity, const Vector3t& accel_bias, const Vector3t& gyro_bias) {
        VectorXt mean = filter_->getState();
        mean.middleRows(10, 3) = accel_bias;
        mean.middleRows(13, 3) = gyro_bias;
        mean.middleRows(16, 3) = gravity;
        filter_->setMean(mean);
    }

private:
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
    std::unique_ptr<model::EKFPoseSystemModel> ekf_system_model_;
    std::unique_ptr<model::EKFOdomPoseSystemModel> odom_system_model_;
    std::unique_ptr<filter::KalmanFilterX> filter_;

    bool use_mahalanobis_{false};
    double last_mahalanobis_d2_{0.0};
    double mahalanobis_threshold_{16.81}; // 99% confidence interval for chi-squared distribution with 6 DOF
    // (calculated by scipy.stats.chi2.ppf(0.99, df=6))

    int consecutive_reject_count_{0};
    const int init_consecutive_reject_{5};

    bool use_odom_{false}; // odomを使う場合は別モデル
};

} // namespace hdl_localization
} // namespace s3l

