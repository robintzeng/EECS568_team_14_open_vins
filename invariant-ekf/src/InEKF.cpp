/* ----------------------------------------------------------------------------
 * Copyright 2018, Ross Hartley <m.ross.hartley@gmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   InEKF.cpp
 *  @author Ross Hartley
 *  @brief  Source file for Invariant EKF
 *  @date   September 25, 2018
 **/

#include "InEKF.h"

namespace inekf {

using namespace std;

void removeRowAndColumn(Eigen::MatrixXd& M, int index);

// ------------ Observation -------------
// Default constructor
Observation::Observation(Eigen::VectorXd& Y, Eigen::VectorXd& b, Eigen::MatrixXd& H, Eigen::MatrixXd& N, Eigen::MatrixXd& PI) :
    Y(Y), b(b), H(H), N(N), PI(PI) {}

// Check if empty
bool Observation::empty() { return Y.rows() == 0; }

ostream& operator<<(ostream& os, const Observation& o) {
    os << "---------- Observation ------------" << endl;
    os << "Y:\n" << o.Y << endl << endl;
    os << "b:\n" << o.b << endl << endl;
    os << "H:\n" << o.H << endl << endl;
    os << "N:\n" << o.N << endl << endl;
    os << "PI:\n" << o.PI << endl;
    os << "-----------------------------------";
    return os;
}

// ------------ InEKF -------------
// Default constructor
InEKF::InEKF() : g_((Eigen::VectorXd(3) << 0,0,-9.81).finished()){}

// Constructor with noise params
InEKF::InEKF(NoiseParams params) : g_((Eigen::VectorXd(3) << 0,0,-9.81).finished()), noise_params_(params) {}

// Constructor with initial state
InEKF::InEKF(RobotState state) : g_((Eigen::VectorXd(3) << 0,0,-9.81).finished()), state_(state) {}

// Constructor with initial state and noise params
InEKF::InEKF(RobotState state, NoiseParams params) : g_((Eigen::VectorXd(3) << 0,0,-9.81).finished()), state_(state), noise_params_(params) {}

// Return robot's current state
RobotState InEKF::getState() {
    return state_;
}

// Sets the robot's current state
void InEKF::setState(RobotState state) {
    state_ = state;
}

// Return noise params
NoiseParams InEKF::getNoiseParams() {
    return noise_params_;
}

// Sets the filter's noise parameters
void InEKF::setNoiseParams(NoiseParams params) {
    noise_params_ = params;
}

// Return filter's prior (static) landmarks
mapIntVector3d InEKF::getPriorLandmarks() {
    return prior_landmarks_;
}

// Set the filter's prior (static) landmarks
void InEKF::setPriorLandmarks(const mapIntVector3d& prior_landmarks) {
    prior_landmarks_ = prior_landmarks;
}

// Return filter's estimated landmarks
map<int,int> InEKF::getEstimatedLandmarks() {
    return estimated_landmarks_;
}


// InEKF Propagation - Inertial Data
void InEKF::Propagate(const Eigen::Matrix<double,6,1>& m, double dt) {

    PropagateIMU(m, dt);

    PropagateCameras(m, dt);

    auto X = state_.getX();
    Eigen::Matrix3d R = state_.getRotation();


    /// Now for the covariance
    int dimP = state_.dimP();
    Eigen::MatrixXd P = state_.getP();

   // ---- Linearized invariant error dynamics -----
    int dimX = state_.dimX();
    int dimTheta = state_.dimTheta();

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dimP,dimP);
    // Inertial terms (called F in RI-MSCKF paper)
    A.block<3,3>(3,0) = skew(g_); // TODO: Efficiency could be improved by not computing the constant terms every time
    A.block<3,3>(6,3) = Eigen::Matrix3d::Identity();
    // Bias terms
    A.block<3,3>(0,dimP-dimTheta) = -R;
    A.block<3,3>(3,dimP-dimTheta+3) = -R;
    for (int i=3; i<dimX; ++i) {
        A.block<3,3>(3*i-6,dimP-dimTheta) = -skew(X.block<3,1>(0,i))*R;
    }

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dimP,dimP);
    Eigen::MatrixXd Phi_n = I + A*dt; // Fast approximation of exp(A*dt). TODO: explore using the full exp() instead

    // // P_new = Phi * P * Phi.transpose() + Q
    Eigen::MatrixXd Phi = I + A*dt;
    // Phi.block<15, 15>(0, 0) = Phi_n;

    // Eigen::MatrixXd Qk_hat = Eigen::MatrixXd::Zero(dimP, dimP);
    // Q.block<15, 15>(0, 0) = Q_d;

    // Eigen::MatrixXd P_pred = Phi * P * Phi.transpose() + Qk_hat;

    // // Set new covariance
    // state_.setP(P_pred);

    // return;

    // Noise terms
    Eigen::MatrixXd Qk = Eigen::MatrixXd::Zero(dimP,dimP);

    // Fill in top 15x15 of Qk with Q_dn
    Qk.block<3,3>(0,0) = noise_params_.getGyroscopeCov();
    Qk.block<3,3>(3,3) = noise_params_.getAccelerometerCov();
    // for(map<int,int>::iterator it=estimated_contact_positions_.begin(); it!=estimated_contact_positions_.end(); ++it) {
    //     Qk.block<3,3>(3+3*(it->second-3),3+3*(it->second-3)) = noise_params_.getContactCov(); // Contact noise terms
    // }
    Qk.block<3,3>(dimP-dimTheta,dimP-dimTheta) = noise_params_.getGyroscopeBiasCov();
    Qk.block<3,3>(dimP-dimTheta+3,dimP-dimTheta+3) = noise_params_.getAccelerometerBiasCov();

    // Discretization
    Eigen::MatrixXd Adj = I;
    Adj.block(0,0,dimP-dimTheta,dimP-dimTheta) = Adjoint_SEK3(X); // Approx 200 microseconds
    Eigen::MatrixXd PhiAdj = Phi * Adj;
    Eigen::MatrixXd Qk_hat = PhiAdj * Qk * PhiAdj.transpose() * dt; // Approximated discretized noise matrix (faster by 400 microseconds)

    // Propagate Covariance
    Eigen::MatrixXd P_pred = Phi * P * Phi.transpose() + Qk_hat;

    // Set new covariance
    state_.setP(P_pred);

    return;
}

// IMU Propagation: X_imu (+) e_I
void InEKF::PropagateIMU(const Eigen::Matrix<double,6,1>& m, double dt) {
    Eigen::Vector3d w = m.head(3) - state_.getGyroscopeBias();    // Angular Velocity
    Eigen::Vector3d a = m.tail(3) - state_.getAccelerometerBias(); // Linear Acceleration

    Eigen::MatrixXd X = state_.getX();
    Eigen::MatrixXd P = state_.getP();

    // Extract State
    Eigen::Matrix3d R = state_.getRotation();
    Eigen::Vector3d v = state_.getVelocity();
    Eigen::Vector3d p = state_.getPosition();

    // Strapdown IMU motion model
    Eigen::Vector3d phi = w*dt;
    Eigen::Matrix3d R_pred = R * Exp_SO3(phi);
    Eigen::Vector3d v_pred = v + (R*a + g_)*dt;
    Eigen::Vector3d p_pred = p + v*dt + 0.5*(R*a + g_)*dt*dt;

    // Set new state (bias has constant dynamics)
    state_.setRotation(R_pred);
    state_.setVelocity(v_pred);
    state_.setPosition(p_pred);
}

// Cam Propagation: C_i (+) e_theta
void InEKF::PropagateCameras(const Eigen::Matrix<double,6,1>& m, double dt) {
    Eigen::Vector3d e_theta = m.head(3) - state_.getGyroscopeBias();
    Eigen::Vector3d e_p = m.tail(3) - state_.getAccelerometerBias();

    for (int ind=0; ind < state_.Cameras_.size(); ind++){
        auto cam_i = state_.Cameras_[ind];
        auto R_i = cam_i.getRotation();
        auto pos_i = cam_i.getPosition();

        Eigen::MatrixXd new_R_i = Exp_SO3(e_theta) * R_i;
        Eigen::VectorXd new_pos_i = Exp_SO3(e_theta) * pos_i + Exp_SEK3(-m)*e_p;

        cam_i.setRotation(new_R_i);
        cam_i.setPosition(new_pos_i);

        state_.updateCameraEstimate(ind, cam_i);
    }

}

// Correct State: Right-Invariant Observation
void InEKF::Correct(const Observation& obs) {
    std::cout << "Computing Kalman Gain" << std::endl;

    // Compute Kalman Gain
    Eigen::MatrixXd P = state_.getP();
    Eigen::MatrixXd PHT = P * obs.H.transpose();
    Eigen::MatrixXd S = obs.H * PHT + obs.N;
    Eigen::MatrixXd K = PHT * S.inverse();

    std::cout << "Copying along diagonals" << std::endl;
    // Copy X along the diagonals if more than one measurement
    Eigen::MatrixXd BigX;
    state_.copyDiagX(obs.Y.rows()/state_.dimX(), BigX);
    // Compute correction terms
    Eigen::MatrixXd Z = BigX*obs.Y - obs.b;
    Eigen::VectorXd delta = K*obs.PI*Z;
    Eigen::MatrixXd dX = Exp_SEK3(delta.segment(0,delta.rows()-state_.dimTheta()));
    Eigen::VectorXd dTheta = delta.segment(delta.rows()-state_.dimTheta(), state_.dimTheta());

    std::cout << "Updating state" << std::endl;
    // Update state
    Eigen::MatrixXd X_new = dX*state_.getX(); // Right-Invariant Update
    Eigen::VectorXd Theta_new = state_.getTheta() + dTheta;
    state_.setX(X_new);
    state_.setTheta(Theta_new);

    std::cout << "Updating covariance" << std::endl;
    // Update Covariance
    Eigen::MatrixXd IKH = Eigen::MatrixXd::Identity(state_.dimP(),state_.dimP()) - K*obs.H;
    Eigen::MatrixXd P_new = IKH * P * IKH.transpose() + K*obs.N*K.transpose(); // Joseph update form

    state_.setP(P_new);
}

// Create Observation from vector of landmark measurements
void InEKF::CorrectLandmarks(const vectorLandmarks& measured_landmarks) {

    Eigen::VectorXd Y;
    Eigen::VectorXd b;
    Eigen::MatrixXd H;
    Eigen::MatrixXd N;
    Eigen::MatrixXd PI;

    Eigen::Matrix3d R = state_.getRotation();
    vectorLandmarks new_landmarks;
    vector<int> used_landmark_ids;

    int lm_ind = 0;
    for (vectorLandmarksIterator it=measured_landmarks.begin(); it!=measured_landmarks.end(); ++it) {
        std::cout << "landmark " << lm_ind << std::endl;
        lm_ind ++;

        // Detect and skip if an ID is not unique (this would cause singularity issues in InEKF::Correct)
        if (find(used_landmark_ids.begin(), used_landmark_ids.end(), it->id) != used_landmark_ids.end()) {
            cout << "Duplicate landmark ID detected! Skipping measurement.\n";
            continue;
        } else { used_landmark_ids.push_back(it->id); }

        // See if we can find id in prior_landmarks or estimated_landmarks
        mapIntVector3dIterator it_prior = prior_landmarks_.find(it->id);
        map<int,int>::iterator it_estimated = estimated_landmarks_.find(it->id);
        if (it_prior!=prior_landmarks_.end()) {
            std::cout << "found in prior landmark set" << std::endl;
            // Found in prior landmark set
            int dimX = state_.dimX();
            int dimP = state_.dimP();
            int startIndex;

            // Fill out Y
            startIndex = Y.rows();
            Y.conservativeResize(startIndex+dimX, Eigen::NoChange);
            Y.segment(startIndex,dimX) = Eigen::VectorXd::Zero(dimX);
            Y.segment(startIndex,3) = it->position; // p_bl
            Y(startIndex+4) = 1;

            // Fill out b
            startIndex = b.rows();
            b.conservativeResize(startIndex+dimX, Eigen::NoChange);
            b.segment(startIndex,dimX) = Eigen::VectorXd::Zero(dimX);
            b.segment(startIndex,3) = it_prior->second; // p_wl
            b(startIndex+4) = 1;

            // Fill out H
            startIndex = H.rows();
            H.conservativeResize(startIndex+3, dimP);
            H.block(startIndex,0,3,dimP) = Eigen::MatrixXd::Zero(3,dimP);
            H.block(startIndex,0,3,3) = skew(it_prior->second); // skew(p_wl)
            H.block(startIndex,6,3,3) = -Eigen::Matrix3d::Identity(); // -I

            // Fill out N
            startIndex = N.rows();
            N.conservativeResize(startIndex+3, startIndex+3);
            N.block(startIndex,0,3,startIndex) = Eigen::MatrixXd::Zero(3,startIndex);
            N.block(0,startIndex,startIndex,3) = Eigen::MatrixXd::Zero(startIndex,3);
            N.block(startIndex,startIndex,3,3) = R * noise_params_.getLandmarkCov() * R.transpose();

            // Fill out PI
            startIndex = PI.rows();
            int startIndex2 = PI.cols();
            PI.conservativeResize(startIndex+3, startIndex2+dimX);
            PI.block(startIndex,0,3,startIndex2) = Eigen::MatrixXd::Zero(3,startIndex2);
            PI.block(0,startIndex2,startIndex,dimX) = Eigen::MatrixXd::Zero(startIndex,dimX);
            PI.block(startIndex,startIndex2,3,dimX) = Eigen::MatrixXd::Zero(3,dimX);
            PI.block(startIndex,startIndex2,3,3) = Eigen::Matrix3d::Identity();

        } else if (it_estimated!=estimated_landmarks_.end()) {;
            std::cout << "found in estimated landmark set" << std::endl;
            // Found in estimated landmark set
            int dimX = state_.dimX();
            int dimP = state_.dimP();
            int startIndex;

            std::cout << "Y" << std::endl;
            std::cout << "dimX: " << dimX << std::endl;
            std::cout << "dimP: " << dimP << std::endl;
            // Fill out Y
            startIndex = Y.rows();
            std::cout << "startIndex: " << startIndex << std::endl;
            std::cout << "it->second: " << it_estimated->second << std::endl;
            Y.conservativeResize(startIndex+dimX, Eigen::NoChange);
            Y.segment(startIndex,dimX) = Eigen::VectorXd::Zero(dimX);
            Y.segment(startIndex,3) = it->position; // p_bl
            Y(startIndex+4) = 1;
            Y(startIndex+it_estimated->second) = -1;

            std::cout << "b" << std::endl;
            // Fill out b
            startIndex = b.rows();
            b.conservativeResize(startIndex+dimX, Eigen::NoChange);
            b.segment(startIndex,dimX) = Eigen::VectorXd::Zero(dimX);
            b(startIndex+4) = 1;
            b(startIndex+it_estimated->second) = -1;

            std::cout << "H" << std::endl;
            // Fill out H
            startIndex = H.rows();
            H.conservativeResize(startIndex+3, dimP);
            H.block(startIndex,0,3,dimP) = Eigen::MatrixXd::Zero(3,dimP);
            H.block(startIndex,6,3,3) = -Eigen::Matrix3d::Identity(); // -I
            H.block(startIndex,3*it_estimated->second-6,3,3) = Eigen::Matrix3d::Identity(); // I

            // Fill out N
            startIndex = N.rows();
            N.conservativeResize(startIndex+3, startIndex+3);
            N.block(startIndex,0,3,startIndex) = Eigen::MatrixXd::Zero(3,startIndex);
            N.block(0,startIndex,startIndex,3) = Eigen::MatrixXd::Zero(startIndex,3);
            N.block(startIndex,startIndex,3,3) = R * noise_params_.getLandmarkCov() * R.transpose();

            // Fill out PI
            startIndex = PI.rows();
            int startIndex2 = PI.cols();
            PI.conservativeResize(startIndex+3, startIndex2+dimX);
            PI.block(startIndex,0,3,startIndex2) = Eigen::MatrixXd::Zero(3,startIndex2);
            PI.block(0,startIndex2,startIndex,dimX) = Eigen::MatrixXd::Zero(startIndex,dimX);
            PI.block(startIndex,startIndex2,3,dimX) = Eigen::MatrixXd::Zero(3,dimX);
            PI.block(startIndex,startIndex2,3,3) = Eigen::Matrix3d::Identity();


        } else {
            std::cout << "first time detected" << std::endl;
            // First time landmark as been detected (add to list for later state augmentation)
            new_landmarks.push_back(*it);
        }
    }

    std::cout << "Correcting state using stacked observations" << std::endl;
    // Correct state using stacked observation
    Observation obs(Y,b,H,N,PI);
    if (!obs.empty()) {
        this->Correct(obs);
    }

    // We don't need to augment state with landmarks because this is
    // handles by the MSCKF
    std::cout << "Updating estimated landmarks" << std::endl;
    int startIndex = state_.getX().rows();
    for (vectorLandmarksIterator it=new_landmarks.begin(); it!=new_landmarks.end(); ++it) {
        startIndex++;
        estimated_landmarks_.insert(pair<int,int> (it->id, startIndex));
    }

    // std::cout << "Augmenting state with new landmarks" << std::endl;
    // // Augment state with newly detected landmarks
    // if (new_landmarks.size() > 0) {
    //     Eigen::MatrixXd X_aug = state_.getX();
    //     Eigen::MatrixXd P_aug = state_.getP();
    //     Eigen::Vector3d p = state_.getPosition();
    //     for (vectorLandmarksIterator it=new_landmarks.begin(); it!=new_landmarks.end(); ++it) {
    //         // Initialize new landmark mean
    //         int startIndex = X_aug.rows();
    //         X_aug.conservativeResize(startIndex+1, startIndex+1);
    //         X_aug.block(startIndex,0,1,startIndex) = Eigen::MatrixXd::Zero(1,startIndex);
    //         X_aug.block(0,startIndex,startIndex,1) = Eigen::MatrixXd::Zero(startIndex,1);
    //         X_aug(startIndex, startIndex) = 1;
    //         X_aug.block(0,startIndex,3,1) = p + R*it->position;

    //         // Initialize new landmark covariance - TODO:speed up
    //         Eigen::MatrixXd F = Eigen::MatrixXd::Zero(state_.dimP()+3,state_.dimP());
    //         F.block(0,0,state_.dimP()-state_.dimTheta(),state_.dimP()-state_.dimTheta()) = Eigen::MatrixXd::Identity(state_.dimP()-state_.dimTheta(),state_.dimP()-state_.dimTheta()); // for old X
    //         F.block(state_.dimP()-state_.dimTheta(),6,3,3) = Eigen::Matrix3d::Identity(); // for new landmark
    //         F.block(state_.dimP()-state_.dimTheta()+3,state_.dimP()-state_.dimTheta(),state_.dimTheta(),state_.dimTheta()) = Eigen::MatrixXd::Identity(state_.dimTheta(),state_.dimTheta()); // for theta
    //         Eigen::MatrixXd G = Eigen::MatrixXd::Zero(F.rows(),3);
    //         G.block(G.rows()-state_.dimTheta()-3,0,3,3) = R;
    //         P_aug = (F*P_aug*F.transpose() + G*noise_params_.getLandmarkCov()*G.transpose()).eval();

    //         // Update state and covariance
    //         state_.setX(X_aug);
    //         state_.setP(P_aug);

    //         // Add to list of estimated landmarks
    //         estimated_landmarks_.insert(pair<int,int> (it->id, startIndex));
    //     }
    // }
    return;
}


void removeRowAndColumn(Eigen::MatrixXd& M, int index) {
    unsigned int dimX = M.cols();
    // cout << "Removing index: " << index<< endl;
    M.block(index,0,dimX-index-1,dimX) = M.bottomRows(dimX-index-1).eval();
    M.block(0,index,dimX,dimX-index-1) = M.rightCols(dimX-index-1).eval();
    M.conservativeResize(dimX-1,dimX-1);
}

} // end inekf namespace
