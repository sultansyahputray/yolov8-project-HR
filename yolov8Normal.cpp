#include <iostream>
#include <vector>
#include <getopt.h>

#include <string.h>
#include <time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "inference.h"

using namespace std;
using namespace cv;

double fps;

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

int getBinaryPeak(cv::Mat& roi){
  int peak = 0;

  for(int i = 0; i < roi.rows; i++){
    for(int j = 0; j < roi.cols; j++){
      if (roi.at<uchar>(j, i) == 255) {
        peak = i;  
        break;
      }
      if(peak != 0){
        break;
      }
    }
  }
  return peak;
}

int main(int argc, char **argv)
{
  TimeCounter time;
  VideoCapture cap("/home/eros/project_HR/sample.mp4");

  bool runOnGPU = false;
  
  int m_size = 480; // config["size"].as<int>();
  std::string model_path = "/home/eros/project_HR/HR.onnx"; // config["model"].as<std::string>();
  
  // int whiteBalanceValue = 4600;
  // namedWindow("WBT", WINDOW_GUI_NORMAL);
  // createTrackbar("wb", "WBT", &whiteBalanceValue, 6200);
  
  Inference inf(model_path, cv::Size(m_size, m_size), "classes.txt", runOnGPU);
  std::vector<std::string> imageNames;

  int num_frames = 0;
  clock_t start, end;

  Detection peak_;

  bool hasCollided = false;  // Flag to track if the collision has already happened

  double ms, fpsLive;
  
  
  while (true)
  {
    if (time.Count() >= 1) {
      num_frames = 0;
      time.Reset();
    }
    num_frames++;
    start = clock(); 
    fps = cap.get(CAP_PROP_FPS);
    string str = to_string(fps);
    // cout << fps << endl;
    
    cv::Mat frame; // = cv::imread(cap);
    cap >> frame;
    // Inference starts here...
    std::vector<Detection> output = inf.runInference(frame);
    
    int detections = output.size();
    int line = frame.cols / 2;
    cv::Mat frameClone = frame.clone();
    for (int i = 0; i < detections; ++i)
    {
      Detection detection = output[i];
      cv::Rect box = detection.box;
      cv::Scalar color = detection.color;
      
      cv::rectangle(frameClone, box, color, 2);
      // Detection box text
      std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
      cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
      cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
      
      cv::rectangle(frameClone, textBox, color, cv::FILLED);
      cv::putText(frameClone, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
      
      cv::line(frameClone, cv::Point2f(line, 0), cv::Point2f(line, frameClone.rows - 1), cv::Scalar(0, 255, 0), 1);
      if (output[i].class_id == 1) {
        peak_ = output[i];
        
        // tabrakan
        if (!hasCollided && peak_.box.x <= line && peak_.box.x + peak_.box.width >= line) {
          cout << "Koordinat Pojok Kiri Atas: (" << peak_.box.x << ", " << peak_.box.y << ") ";
          hasCollided = true;
          cv::Mat roi = frame(peak_.box);
          cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);
          
          cv::Mat roiBinary;
          cv::threshold(roi, roiBinary, 200, 255, cv::THRESH_BINARY);
          
          cv::Mat roiInverted;
          cv::bitwise_not(roiBinary, roiInverted);

          if(!roi.empty()){
            // cv::imshow("threshold", roiBinary);
            cv::imshow("invert", roiInverted);
          }

          // perhitungan peak
          int binaryPeak = getBinaryPeak(roiInverted);
          int framePeak = peak_.box.y + (binaryPeak/2);
          
          cout<<framePeak<<endl;
          cv::line(frameClone, cv::Point2f(0, framePeak), cv::Point2f(frameClone.cols, framePeak), cv::Scalar(255, 0, 255), 1);
        }
        else if(peak_.box.x + peak_.box.width <= line){
          hasCollided = false;
        }  
      }
    }
    
    
    // Inference ends here...
    end = clock();
    
    // This is only for preview purposes
    float scale = 0.8;
    cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
    double sc = (double(end) - double(start)) / double(CLOCKS_PER_SEC);
    fpsLive = double(num_frames) / double(sc);
    cv::imshow("Inference", frameClone);

    if (waitKey(1) == 27) break;
  }
}
