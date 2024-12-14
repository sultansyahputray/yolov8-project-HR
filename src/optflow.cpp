#include <iostream>
#include <vector>
#include <getopt.h>

#include <string.h>
#include <time.h>

#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include "yaml-cpp/yaml.h"

#include "inference.h"

using namespace std;
using namespace cv;

double fps;

bool isCounting(int threshold, int a, int b){
    return a <= threshold && b >= threshold; 
}

struct TimeCounter {
  public:
    TimeCounter() : reset(true) {}
  
    float Count() {
      if (reset) {
        previous_time   = std::chrono::system_clock::now();
        reset = false;
      }

      current_time = std::chrono::system_clock::now();
      elapsed_time = current_time - previous_time;

      return elapsed_time.count();
    }

    void Reset() { 
      reset = true; 
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> current_time, previous_time;
    std::chrono::duration<float> elapsed_time;

    bool reset;
};

int main(int argc, char **argv)
{
    TimeCounter time;
    VideoCapture cap("/home/eros/projek_HR/samples/200sample.mp4");
    
    if(!cap.isOpened()){
        std::cout<<"gabisa buka"<<std::endl;
    }

    bool runOnGPU = false;

    std::string model_path = "/home/eros/models/80.onnx";

    int m_size = 480;
    Inference inf(model_path, cv::Size(m_size, m_size), "classes.txt", runOnGPU);

    std::vector<std::string> imageNames;

    int num_frames=0;
    clock_t start,end;

    double ms,fpsLive;

    int threshold = 400;

    bool toggle=false;

    int hitung=0;

    while(true)
    {   
        if(time.Count() >= 1){
            // fprintf(stderr, "FPS >> %d\n\n", num_frames);
            num_frames = 0;
            time.Reset();
	    }
        num_frames++;
        start=clock(); 
	
        fps=cap.get(CAP_PROP_FPS);
        // string str=to_string(fps);
        // cout<<fps<<endl;
        
        cv::Mat frame;
        cap>>frame;
        
        std::vector<Detection> output = inf.runInference(frame);

        // cv::putText(frame, str, Point(20, 20), FONT_HERSHEY_DUPLEX,1, Scalar(255, 0, 0),2, 0);

        int detections = output.size();
        // std::cout << "Number of detections:" << detections;
        
        cv::Mat frameClone = frame.clone();
        cv::line(frameClone, cv::Point2f(threshold, 0), cv::Point2f(threshold, frameClone.rows - 1), cv::Scalar(0, 255, 0), 1);
        bool temp;
        for (int i = 0; i < detections; ++i){
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;
            detection.isCount = isCounting(threshold, detection.box.x, detection.box.x + detection.box.width); 
            temp = detection.isCount;
            // Detection box
            if(detection.class_id == 0){
                cv::rectangle(frameClone, box, cv::Scalar(0, 0, 255), 2);
            }else{
                cv::rectangle(frameClone, box, (detection.isCount) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0), 2);
            }
            std::cout<<((detection.isCount) ? "true":"false")<<std::endl;
        }
        // Inference ends here...
        end=clock();
        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frameClone, frameClone, cv::Size(frameClone.cols*scale, frameClone.rows*scale));

        double sc=(double(end)-double(start))/double(CLOCKS_PER_SEC);
        fpsLive=double(num_frames)/double(sc);
        // cout<<"\t"<<fpsLive<<endl;

        cv::imshow("Inference", frameClone);
        // cv::imshow("mask", mask);
    
        if(waitKey(1)==27)break;
    }
}
