#include <sensor_msgs/RegionOfInterest.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cstdlib>
#include <random>


void nonMaximumSuppression(const float nmsThresh,  
std::vector<std::vector<sensor_msgs::RegionOfInterest>>& sync_camera_objects,
std::vector<std::vector<int>>& sync_camera_objects_classid, 
std::vector<std::vector<float>>& sync_camera_objects_confidence) 
{

    for (size_t frame = 0; frame < sync_camera_objects.size(); ++frame) 
    {
        std::vector<sensor_msgs::RegionOfInterest>& objects = sync_camera_objects[frame];
        std::vector<int>& classids = sync_camera_objects_classid[frame];
        std::vector<float>& confidences = sync_camera_objects_confidence[frame];

        auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
            if (x1min > x2min)
            {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
            }
            return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
        };
        auto computeIoU = [&overlap1D](const sensor_msgs::RegionOfInterest& bbox1, const sensor_msgs::RegionOfInterest& bbox2) -> float {
            float overlapX = overlap1D(bbox1.x_offset, bbox1.x_offset + bbox1.width, bbox2.x_offset, bbox2.x_offset + bbox2.width);
            float overlapY = overlap1D(bbox1.y_offset, bbox1.y_offset + bbox1.height, bbox2.y_offset, bbox2.y_offset + bbox2.height);
            float area1 = (bbox1.width * bbox1.height);
            float area2 = (bbox2.width * bbox2.height);
            float overlap2D = overlapX * overlapY;
            float u = area1 + area2 - overlap2D;
            return u == 0 ? 0 : overlap2D / u;
        };

        std::vector<int> indices(objects.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::stable_sort(indices.begin(), indices.end(), [&confidences](int i1, int i2) {
            return confidences[i1] > confidences[i2];

        });

        std::vector<sensor_msgs::RegionOfInterest> out_objects;
        std::vector<int> out_classids;
        std::vector<float> out_confidences;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            bool keep = true;
            for (size_t j = 0; j < out_objects.size(); ++j) 
            {
                // Confirm whether the two bounding box are the same class
                if (classids[indices[i]] == out_classids[j]) 
                {
                    float overlap = computeIoU(objects[indices[i]], out_objects[j]);
                    if (overlap > nmsThresh) 
                    {
                        keep = false;
                        break;
                    }
                }
            }
            if (keep)
            {
                out_objects.push_back(objects[indices[i]]);
                out_classids.push_back(classids[indices[i]]);
                out_confidences.push_back(confidences[indices[i]]);
            }
        }

        sync_camera_objects[frame] = std::move(out_objects);
        sync_camera_objects_classid[frame] = std::move(out_classids);
        sync_camera_objects_confidence[frame] = std::move(out_confidences);
    }
}

int main(){

    std::vector<std::vector<sensor_msgs::RegionOfInterest>> sync_camera_objects(5, std::vector<sensor_msgs::RegionOfInterest>(5));
    std::vector<std::vector<int>> sync_camera_objects_classid(5,std::vector<int>(5));
    std::vector<std::vector<float>> sync_camera_objects_confidence(5,std::vector<float>(5));

    std::vector<int> class_id = {1,2,3};
    // random 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, class_id.size() - 1);

    for (int frame = 0; frame < 5; ++frame) {
        for (int i = 0; i < 5; ++i) {

            int rand_id = distrib(gen);
            int e = class_id[rand_id];

            sync_camera_objects[frame][i].x_offset = i * 10;
            sync_camera_objects[frame][i].y_offset = i * 10;
            sync_camera_objects[frame][i].width = 100; 
            sync_camera_objects[frame][i].height = 100;
            sync_camera_objects_classid[frame][i] = e;
            sync_camera_objects_confidence[frame][i] = i * 0.1 + 0.5;
        }
    }

    std::cout << "****Original Frame"<< std::endl;
    for (size_t frame = 0 ; frame < sync_camera_objects.size(); ++frame)
    {
            std::cout << "Frame" << frame << std::endl;
            for (size_t i = 0; i < sync_camera_objects[frame].size(); i++)
            {
                std::cout << "Object " << i << ": (" 
                      << sync_camera_objects[frame][i].x_offset << ","
                      << sync_camera_objects[frame][i].y_offset << "), width: "
                      << sync_camera_objects[frame][i].width << ", height: "
                      << sync_camera_objects[frame][i].height << ", classid: "
                      << sync_camera_objects_classid[frame][i] << ", confidence: "
                      << sync_camera_objects_confidence[frame][i] << std::endl;
            }
    }

    float nmsThresh=0.1;
    std::cout << "**nmsThresh: "<< nmsThresh <<std::endl;

    nonMaximumSuppression(nmsThresh, sync_camera_objects, sync_camera_objects_classid, sync_camera_objects_confidence);

    std::cout << "****After NMS" <<std::endl;
    for (size_t frame = 0; frame < sync_camera_objects.size(); ++frame)
    {
        std::cout << "Frame " << frame << ":\n";
        for (size_t i = 0; i < sync_camera_objects[frame].size(); ++i)
        {
            std::cout << "Object " << i << ": (" 
                      << sync_camera_objects[frame][i].x_offset << ","
                      << sync_camera_objects[frame][i].y_offset << "), width: "
                      << sync_camera_objects[frame][i].width << ", height: "
                      << sync_camera_objects[frame][i].height << ", classid: "
                      << sync_camera_objects_classid[frame][i] << ", confidence: "
                      << sync_camera_objects_confidence[frame][i] << std::endl;
        }
    }
    return 0;
}
