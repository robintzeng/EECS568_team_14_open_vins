#include <vector>
#include <iostream>

#include "../../invariant-ekf/include/InEKF.h"
#include "../../ov_core/src/feat/Feature.h"


// Convert OpenVINS feature objects to InEKF Landmark objects
std::vector<inekf::Landmark> convert_ov_features_to_landmarks(const std::vector<ov_core::Feature> features){
    std::vector<inekf::Landmark> landmarks;

    for (auto feature: features){

        auto id = feature.featid; // unique id
        auto p_bl = feature.p_FinG; // position of feature in global frame

        inekf::Landmark landmark(id, p_bl);
        landmarks.push_back(landmark);
    }

    return landmarks;
}
