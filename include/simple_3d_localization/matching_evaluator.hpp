#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <sophus/so3.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <pcl/common/point_tests.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#if __has_include(<pcl/registration/ndt.h>)
#include <pcl/registration/ndt.h>
#endif
#if __has_include(<pclomp/ndt_omp.h>)
#include <pclomp/ndt_omp.h>
#endif

namespace s3l::scan_matching {

template <typename PointT>
struct ScanMatchingMetrics {
    std::size_t total_points{0};
    std::size_t inlier_count{0};
    double inlier_ratio{0.0};
    double fitness_score{std::numeric_limits<double>::quiet_NaN()};
    double rmse{std::numeric_limits<double>::quiet_NaN()};
    double mean_error{std::numeric_limits<double>::quiet_NaN()};
    double max_error{0.0};
    double transformation_probability{std::numeric_limits<double>::quiet_NaN()};
    double nvtl{std::numeric_limits<double>::quiet_NaN()};
    Eigen::Matrix3d translation_covariance{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d rotation_covariance{Eigen::Matrix3d::Identity()};
    bool reliable{false};
};

namespace detail {
template <typename PointT>
double try_get_transformation_probability(pcl::Registration<PointT, PointT>& registration) {
#if __has_include(<pcl/registration/ndt.h>)
    if (auto* ndt = dynamic_cast<pcl::NormalDistributionsTransform<PointT, PointT>*>(&registration)) {
        return ndt->getTransformationProbability();
    }
#endif
#if __has_include(<pclomp/ndt_omp.h>)
    if (auto* ndt = dynamic_cast<pclomp::NormalDistributionsTransform<PointT, PointT>*>(&registration)) {
        return ndt->getTransformationProbability();
    }
#endif
    return std::numeric_limits<double>::quiet_NaN();
}
}  // namespace detail

template <typename PointT>
ScanMatchingMetrics<PointT> evaluate(
    pcl::Registration<PointT, PointT>& registration,
    const typename pcl::PointCloud<PointT>::ConstPtr& aligned,
    double max_correspondence_distance = std::numeric_limits<double>::quiet_NaN()) {

    ScanMatchingMetrics<PointT> metrics;
    metrics.total_points = aligned ? aligned->size() : 0;
    metrics.fitness_score = registration.getFitnessScore();

    if (!aligned || aligned->empty()) {
        return metrics;
    }

    auto target = registration.getInputTarget();
    auto search = registration.getSearchMethodTarget();
    if (!target || !search) {
        return metrics;
    }

    double max_dist = max_correspondence_distance;
    if (!std::isfinite(max_dist)) {
        max_dist = registration.getMaxCorrespondenceDistance();
    }
    if (!(max_dist > 0.0)) {
        max_dist = std::numeric_limits<double>::infinity();
    }

    std::vector<Eigen::Vector3d> residuals;
    std::vector<Eigen::Vector3d> aligned_points;
    residuals.reserve(aligned->size());
    aligned_points.reserve(aligned->size());

    double sum_error_norm = 0.0;
    double sum_error_sq = 0.0;
    Eigen::Vector3d sum_residual = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_aligned = Eigen::Vector3d::Zero();
    double max_error = 0.0;

    std::vector<int> nn_indices(1);
    std::vector<float> nn_sq_dists(1);

    // 各点に対して最近傍探索を行い、対応点が見つかれば残差を計算
    for (std::size_t i = 0; i < aligned->size(); ++i) {
        const auto& src_pt = aligned->points[i];
        if (!pcl::isFinite(src_pt)) {
            continue;
        }
        if (search->nearestKSearch(src_pt, 1, nn_indices, nn_sq_dists) <= 0) {
            continue;
        }
        const double dist = std::sqrt(nn_sq_dists[0]);
        if (dist > max_dist) {
            continue;
        }
        const auto& tgt_pt = target->points[nn_indices[0]];
        if (!pcl::isFinite(tgt_pt)) {
            continue;
        }

        Eigen::Vector3d aligned_vec(src_pt.x, src_pt.y, src_pt.z);
        Eigen::Vector3d target_vec(tgt_pt.x, tgt_pt.y, tgt_pt.z);
        Eigen::Vector3d residual = target_vec - aligned_vec; // 対応点の残差

        residuals.push_back(residual);
        aligned_points.push_back(aligned_vec);
        sum_residual += residual;
        sum_aligned += aligned_vec;

        const double err_norm = residual.norm();
        sum_error_norm += err_norm;
        sum_error_sq += err_norm * err_norm;
        max_error = std::max(max_error, err_norm);
    }

    // 統計量を計算
    metrics.inlier_count = residuals.size();
    if (metrics.total_points > 0) {
        metrics.inlier_ratio = static_cast<double>(metrics.inlier_count) /
                               static_cast<double>(metrics.total_points);
    }

    // inlier_countが少なすぎる場合は、詳細な統計量を計算せずに終了
    if (metrics.inlier_count < 4) {
        metrics.mean_error = metrics.inlier_count > 0 ? sum_error_norm / metrics.inlier_count
                                                      : std::numeric_limits<double>::quiet_NaN();
        metrics.rmse = metrics.inlier_count > 0 ? std::sqrt(sum_error_sq / metrics.inlier_count)
                                                : std::numeric_limits<double>::quiet_NaN();
        metrics.max_error = max_error;
        metrics.transformation_probability = detail::try_get_transformation_probability(registration);
        metrics.nvtl = metrics.transformation_probability;
        metrics.reliable = false;
        return metrics;
    }

    Eigen::Vector3d mean_residual = sum_residual / static_cast<double>(metrics.inlier_count);
    Eigen::Vector3d centroid = sum_aligned / static_cast<double>(metrics.inlier_count);

    Eigen::Matrix3d trans_cov = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d AtA = Eigen::Matrix3d::Zero();

    // 分散共分散行列を計算
    for (std::size_t i = 0; i < residuals.size(); ++i) {
        Eigen::Vector3d diff = residuals[i] - mean_residual;
        trans_cov += diff * diff.transpose();

        Eigen::Vector3d centered = aligned_points[i] - centroid;
        Eigen::Matrix3d skew = Sophus::SO3d::hat(centered);
        AtA += skew.transpose() * skew;
    }
    trans_cov /= static_cast<double>(metrics.inlier_count - 1);

    const double rmse = std::sqrt(sum_error_sq / static_cast<double>(metrics.inlier_count));
    const double sigma2 = std::max(rmse * rmse, 1e-9);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(AtA);
    Eigen::Vector3d eigvals = solver.eigenvalues();
    Eigen::Matrix3d rot_cov;
    if ((eigvals.array() > 1e-8).all()) {
        Eigen::Vector3d inv = eigvals.array().inverse();
        rot_cov = solver.eigenvectors() * inv.asDiagonal() * solver.eigenvectors().transpose();
        rot_cov *= sigma2;
    } else {
        Eigen::Matrix3d regularized = AtA;
        regularized.diagonal().array() += 1e-6;
        rot_cov = regularized.ldlt().solve(Eigen::Matrix3d::Identity()) * sigma2;
    }

    metrics.mean_error = sum_error_norm / static_cast<double>(metrics.inlier_count);
    metrics.rmse = rmse;
    metrics.max_error = max_error;
    metrics.translation_covariance = 0.5 * (trans_cov + trans_cov.transpose());
    metrics.rotation_covariance = 0.5 * (rot_cov + rot_cov.transpose());
    metrics.transformation_probability = detail::try_get_transformation_probability(registration);
    metrics.nvtl = std::isfinite(metrics.transformation_probability)
                       ? metrics.transformation_probability /
                             static_cast<double>(std::max<std::size_t>(1, metrics.total_points))
                       : metrics.rmse;
    metrics.reliable = metrics.inlier_count >= 10 && metrics.inlier_ratio > 0.05 && std::isfinite(metrics.rmse);

    return metrics;
}

} // namespace s3l::scan_matching
