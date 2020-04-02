#include "FeatureLandmarkConverter.h"
#include <vector>
#include <iostream>


int main(){

    std::cout << "Running..." << std::endl;

    std::vector<ov_core::Feature> features;

    auto f1 = ov_core::Feature();
    f1.featid = 1;
    f1.p_FinG = Eigen::Vector3d(1.0, 2.0, 0.5);

    features.push_back(f1);

    std::cout << "Number of features: " << features.size() << std::endl;

    auto landmarks = convert_ov_features_to_landmarks(features);

    for (inekf::Landmark landmark: landmarks){
        const auto x = landmark.position[0];
        const auto y = landmark.position[1];
        const auto z = landmark.position[2];

        assert(x == 1.0);
        assert(y == 2.0);
        assert(z == 0.5);
    }

    return 0;
}
