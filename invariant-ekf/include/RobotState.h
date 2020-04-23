/* ----------------------------------------------------------------------------
 * Copyright 2018, Ross Hartley
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   RobotState.h
 *  @author Ross Hartley
 *  @brief  Header file for RobotState
 *  @date   September 25, 2018
 **/

#ifndef ROBOTSTATE_H
#define ROBOTSTATE_H
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#if INEKF_USE_MUTEX
#include <mutex>
#endif

namespace inekf {

class CameraPose{
    public:
        CameraPose(const Eigen::MatrixXd& R, const Eigen::VectorXd& pos) :
        R_(R), pos_(pos){}

        void setRotation(Eigen::MatrixXd new_R) {
            assert (new_R.rows() == R_.rows());
            assert (new_R.cols() == R_.cols());
            R_ = new_R;
        }

        void setPosition(Eigen::VectorXd new_pos) {
            assert (new_pos.rows() == pos_.rows());
            assert (new_pos.cols() == pos_.cols());
            pos_ = new_pos;
        }

        Eigen::MatrixXd getRotation() const{
            return R_;
        }

        Eigen::VectorXd getPosition() const{
            return pos_;
        }
    private:
        Eigen::MatrixXd R_;
        Eigen::VectorXd pos_;
};



class VisualFeature
{
public:
    VisualFeature();
    VisualFeature(const VisualFeature& orig);

    ~VisualFeature(){}

    Eigen::Vector3d getPosition() const;

    void setPosition(Eigen::Vector3d new_pos);

    std::vector<CameraPose> poses_seen_from_;

    VisualFeature& operator=(const VisualFeature& orig);
    int id_;

protected:
    static int nextID;

private:
    Eigen::Vector3d pos_;
};


class RobotState {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        RobotState();
        RobotState(const Eigen::MatrixXd& X);
        RobotState(const Eigen::MatrixXd& X, const Eigen::VectorXd& Theta);
        RobotState(const Eigen::MatrixXd& X, const Eigen::VectorXd& Theta, const Eigen::MatrixXd& P);

#if INEKF_USE_MUTEX
        // RobotState(RobotState&& other); // Move initialization
        RobotState(const RobotState& other); // Copy initialization
        // RobotState& operator=(RobotState&& other); // Move assignment
        RobotState& operator=(const RobotState& other); // Copy assignment
#endif

        const Eigen::MatrixXd getX();
        const Eigen::VectorXd getTheta();
        const Eigen::MatrixXd getP();
        const Eigen::Matrix3d getRotation();
        const Eigen::Vector3d getVelocity();
        const Eigen::Vector3d getPosition();
        const Eigen::Vector3d getGyroscopeBias();
        const Eigen::Vector3d getAccelerometerBias();
        const int dimX();
        const int dimTheta();
        const int dimP();

        void setX(const Eigen::MatrixXd& X);
        void setP(const Eigen::MatrixXd& P);
        void setTheta(const Eigen::VectorXd& Theta);
        void setRotation(const Eigen::Matrix3d& R);
        void setVelocity(const Eigen::Vector3d& v);
        void setPosition(const Eigen::Vector3d& p);
        void setGyroscopeBias(const Eigen::Vector3d& bg);
        void setAccelerometerBias(const Eigen::Vector3d& ba);

        void copyDiagX(int n, Eigen::MatrixXd& BigX);

        friend std::ostream& operator<<(std::ostream& os, const RobotState& s);

        void augmentState(CameraPose cam_pose);
        void updateCameraEstimate(const int cam_index, CameraPose cam_pose);
        CameraPose getCameraEstimate(const int cam_index);
        int getNumberCameras() const {return Cameras_.size();}

        std::vector<CameraPose> getCameras() {return Cameras_;}
        std::vector<CameraPose> Cameras_;

    private:
        Eigen::MatrixXd X_;
        Eigen::VectorXd Theta_;
        Eigen::MatrixXd P_;

#if INEKF_USE_MUTEX
        mutable std::mutex mutex_;
#endif

};

} // end inekf namespace
#endif

