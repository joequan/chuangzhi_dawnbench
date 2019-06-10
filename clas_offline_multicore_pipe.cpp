/*!
*  Copyright (c) 2018 by Contributors
* \file clas_offline_multicore_pipe.cpp
* \brief
*
* \author
*/
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <sys/time.h>
#include <algorithm>
#include <condition_variable> // NOLINT
#include <iomanip>
#include <iosfwd>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread> // NOLINT
#include <utility>
#include <vector>
#include <atomic>

#include "runtime.hpp"
#include "blocking_queue.hpp"
#include "func_runner.hpp"

using std::map;
using std::max;
using std::min;
using std::queue;
using std::thread;
using std::stringstream;
using std::vector;

std::condition_variable condition;
std::mutex condition_m;
bool ready_start = false;

#define PRE_READ
#ifdef USE_OPENCV

DEFINE_string(offlinemodel, "",
    "The prototxt file used to find net configuration");
DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "123.68,116.779,103.939",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(use_mean, "on",
    "if it does not need to subtract mean, then use_mean must be 'off'.");
DEFINE_string(stdt_value, "58.395,57.12,57.375", "stdt value");
DEFINE_int32(data_parallelism, 1,
    "1, 2, 4, 8, 16, 32, there are data_parallelism * batch will be processed.");
DEFINE_int32(model_parallelism, 1,
    "1, 2, 4, 8, 16, 32, number of cores that used to process one batch.");
DEFINE_int32(threads, 1,
    "1-4 when data_parallelism is 8, 1-32 when data_parallelism is 1");
DEFINE_int32(fix8, 0,
    "0 or 1, fix8 mode");
DEFINE_int32(normalize,
             2,
             "0 or 1 or 2, 0 means the first layer is convfirst with mean and stdt,"
             " 1 means the first layer is convfirst with mean without stdt,"
             " 2 means the first layer is batchnorm or convfirst is false.");
DEFINE_double(scale, 1.0,
    "Optional; sometimes the input needs to be scaled, MobileNet for example");
DEFINE_string(img_dir, "",
    "path to images");
DEFINE_string(images, "",
    "file name of images");
DEFINE_string(label_file, "",
    "path to img_to_label file");
DEFINE_int32(batch, 1,
    "batch of the input data");
DEFINE_string(debug, "false",
    "whether to print top1 and top5 image names");
DEFINE_string(synset, "synset_words_mxnet",
    "path to synset words");


class PostProcessor;

class Inferencer {
 public:
  Inferencer(
    const string& offlinemodel,
    const int& thread_id,
    const int& data_parallelism);
  int n() {return in_n_;}
  int c() {return in_c_;}
  int h() {return in_h_;}
  int w() {return in_w_;}
  unsigned int output_chw() {return out_c_ * out_h_ * out_w_;}
  void pushValidInputData(void** data);
  void pushFreeInputData(void** data);
  void** popValidInputData();
  void** popFreeInputData();
  void pushValidOutputData(void** data);
  void pushFreeOutputData(void** data);
  void** popValidOutputData();
  void** popFreeOutputData();
  void pushValidInputNames(vector<string> imgs);
  vector<string> popValidInputNames();
  void notify();
  void run();

  cnrtDataDescArray_t inputDescS() {return inputDescS_;}
  cnrtDataDescArray_t outputDescS() {return outputDescS_;}
  mof::BlockingQueue<void**> validInputFifo_;
  mof::BlockingQueue<void**> freeInputFifo_;
  mof::BlockingQueue<void**> validOutputFifo_;
  mof::BlockingQueue<void**> freeOutputFifo_;
  mof::BlockingQueue<vector<string> > imagesFifo_;

  cnrtDataDescArray_t inputDescS_, outputDescS_;
  cnrtStream_t stream_;
  int inputNum, outputNum;
  cnrtModel_t model_;
  cnrtFunction_t function;
  unsigned int in_n_, in_c_, in_h_, in_w_;
  unsigned int out_n_, out_c_, out_h_, out_w_;
  int out_count_;
  cnrtDim3_t dim_;
  bool running_;
  int thread_id_;
  int data_parallelism_;
  cnrtFunctionType_t func_type_;
  PostProcessor* post_processor_;
  double invoke_time;
  cnrtInvokeFuncParam_t invoke_param_;
};


class PostProcessor {
 public:
  PostProcessor();
  string get_string(string names, int index);
  void notify();
  void run();
  void top5_print(float* result_buffer, unsigned int count, const string& image);
  void get_label(const string& label_file);

  bool running_;
  Inferencer* inferencer_;
  int thread_id_;
  std::map<std::string, int> accurate_labels_;
  unsigned int top1_;
  unsigned int top5_;
  unsigned int total_;
};

template<class datatype>
class DataProvider {
 public:
  DataProvider(
               const string& mean_file,
               const string& mean_value,
               const queue<string>& images,
               float scale = 1.0);
  void SetMean(const string&, const string&);
  void SetStdt(const string&);
  void preRead();
  void run();
  void WrapInputLayer(std::vector<std::vector<cv::Mat> >* input_imgs);
  void Preprocess(const std::vector<cv::Mat>& imgs);
  void Preprocess(const std::vector<std::vector<cv::Mat> >& imgs);
  cv::Mat mean_;
  cv::Mat stdt_;
  void** cpu_data_;
  datatype* cpy_data_;
  int in_n_, in_c_, in_h_, in_w_;
  float scale_;
  queue<string> images_;
  Inferencer* inferencer_;
  cv::Size input_geometry_;
  int thread_id_;
  bool need_mean_;
  vector<vector<cv::Mat> > v_images;
  vector<vector<string> > v_names;
};

template<class datatype>
DataProvider<datatype>::DataProvider(
               const string& mean_file,
               const string& mean_value,
               const queue<string>& images,
               float scale) {
  images_ = images;
  thread_id_ = 0;
  scale_ = scale;

  need_mean_ = true;
  if (FLAGS_use_mean == "off") {
    need_mean_ = false;
  }
}

template<class datatype>
void DataProvider<datatype>::preRead() {
  in_n_ = inferencer_->n();
  std::string img_dir = FLAGS_img_dir;
  while (images_.size()) {
    vector<cv::Mat> imgs;
    vector<string> img_names;
    int left_num = images_.size();
    for (int i = 0; i < in_n_; i++) {
      if (i < left_num) {
        string file = images_.front();
        images_.pop();
        cv::Mat img;
        cv::cvtColor(cv::imread(img_dir + file), img, cv::COLOR_BGR2RGB);
        imgs.push_back(img);
        img_names.push_back(file);
      } else {
        cv::Mat img;
        mof::Timer one_picture_;
        cv::cvtColor(cv::imread(img_dir + img_names[0]), img, cv::COLOR_BGR2RGB);
        imgs.push_back(img);
        img_names.push_back("null");
      }
    }
    v_images.push_back(imgs);
    v_names.push_back(img_names);
  }

  cpu_data_ = reinterpret_cast<void**>(malloc(sizeof(void*) * 1));
  in_c_ = inferencer_->c();
  in_h_ = inferencer_->h();
  in_w_ = inferencer_->w();
  // cpy_data_ = (datatype*)malloc(v_images.size() * in_n_ * in_c_ * in_h_ * in_w_ * sizeof(datatype));
  cpy_data_ = reinterpret_cast<datatype*>(malloc(v_images.size()
                                                 * in_n_ * in_c_ * in_h_ * in_w_ * sizeof(datatype)));
  input_geometry_ = cv::Size(in_w_, in_h_);
  SetMean(FLAGS_mean_file, FLAGS_mean_value);
  SetStdt(FLAGS_stdt_value);
  Preprocess(v_images);
}

template<class datatype>
void DataProvider<datatype>::run() {
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, 0));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  if (FLAGS_data_parallelism * FLAGS_model_parallelism <= 8) {
    cnrtSetCurrentChannel((cnrtChannelType_t)thread_id_);
  }

  std::unique_lock<std::mutex> lk(condition_m);
  LOG(INFO) << "Waiting ...";
  condition.wait(lk, [](){return ready_start;});
  lk.unlock();

#ifdef PRE_READ
  for (int i = 0; i < v_images.size(); i++) {
    mof::Timer prepareInput;
    vector<cv::Mat> imgs = v_images[i];
    vector<string> img_names = v_names[i];
    cpu_data_[0] = reinterpret_cast<void*>(cpy_data_ +
        i * imgs.size() * in_c_ * in_h_ * in_w_);
#else
  cpu_data_ = reinterpret_cast<void**>(malloc(sizeof(void*) * 1));
  in_n_ = inferencer_->n();
  in_c_ = inferencer_->c();
  in_h_ = inferencer_->h();
  in_w_ = inferencer_->w();
  // cpy_data_ = (datatype*)malloc(in_n_ * in_c_ * in_h_ * in_w_ * sizeof(datatype));
  cpy_data_ = reinterpret_cast<datatype*>(malloc(in_n_ * in_c_ * in_h_ * in_w_ * sizeof(datatype)));
  cpu_data_[0] = reinterpret_cast<void*>(cpy_data_);
  input_geometry_ = cv::Size(in_w_, in_h_);
  SetMean(FLAGS_mean_file, FLAGS_mean_value);
  SetStdt(FLAGS_stdt_value);
  std::string img_dir = FLAGS_img_dir;
  while (images_.size()) {
    mof::Timer prepareInput;
    vector<cv::Mat> imgs;
    vector<string> img_names;
    int left_num = images_.size();
    for (int i = 0; i < in_n_; i++) {
      if (i < left_num) {
        string file = images_.front();
        images_.pop();
        cv::Mat img;
        // BGR --> RGB
        cv::cvtColor(cv::imread(img_dir + file), img, cv::COLOR_BGR2RGB);
        imgs.push_back(img);
        img_names.push_back(file);
      } else {
        cv::Mat img;
        // BGR --> RGB
        cv::cvtColor(cv::imread(img_dir + img_names[0]), img, cv::COLOR_BGR2RGB);
        imgs.push_back(img);
        img_names.push_back("null");
      }
    }
    Preprocess(imgs);
#endif
    prepareInput.duration("prepare input data ...");

    void** mlu_data = inferencer_->popFreeInputData();
    mof::Timer copyin;
    CNRT_CHECK(cnrtMemcpyBatchByDescArray(
        mlu_data, cpu_data_, inferencer_->inputDescS(),
        inferencer_->inputNum, FLAGS_data_parallelism, CNRT_MEM_TRANS_DIR_HOST2DEV));
    copyin.duration("copyin time ...");
    inferencer_->pushValidInputData(mlu_data);
    inferencer_->pushValidInputNames(img_names);
  }

  LOG(INFO) << "DataProvider: no data ...";
  inferencer_->notify();
  free(cpy_data_);
  free(cpu_data_);
}

template<class datatype>
void DataProvider<datatype>::Preprocess(const std::vector<std::vector<cv::Mat> >& v_imgs) {
  /* Convert the input image to the input image format of the network. */
  datatype* input_data = reinterpret_cast<datatype*>(cpy_data_);
  for (int j = 0; j < v_imgs.size(); ++j) {
    std::vector<cv::Mat> imgs = v_imgs[j];
    for (int i = 0; i < imgs.size(); ++i) {
      cv::Mat sample;
      if (imgs[i].channels() == 3 && in_c_ == 1)
        cv::cvtColor(imgs[i], sample, cv::COLOR_BGR2GRAY);
      else if (imgs[i].channels() == 4 && in_c_ == 1)
        cv::cvtColor(imgs[i], sample, cv::COLOR_BGRA2GRAY);
      else if (imgs[i].channels() == 4 && in_c_ == 3)
        cv::cvtColor(imgs[i], sample, cv::COLOR_BGRA2BGR);
      else if (imgs[i].channels() == 1 && in_c_ == 3)
        cv::cvtColor(imgs[i], sample, cv::COLOR_GRAY2BGR);
      else
        sample = imgs[i];
      //squared phote
      int short_edge = std::min(sample.rows, sample.cols);
      int yy = (int)((sample.rows - short_edge) / 2);
      int xx = (int)((sample.cols - short_edge) / 2);
      cv::Rect rect(xx, yy, short_edge, short_edge);
      cv::Mat sample_squared(sample, rect);
      //resize or crop
      cv::Mat sample_resized, sample_resized_temp;
      if (short_edge < inferencer_->h()) {
        cv::resize(sample_squared, sample_resized, input_geometry_);
      } else {
        if (short_edge != 256) {
          cv::resize(sample_squared, sample_resized_temp, cv::Size(256, 256));
        } else {
          sample_resized_temp = sample_squared;
        }
        int crop_off = (int)((256 - inferencer_->h()) / 2);
        if (crop_off < 0) {
          cv::resize(sample_resized_temp, sample_resized, input_geometry_);
        } else {
          cv::Rect crop_rect(
              crop_off, crop_off, inferencer_->w(), inferencer_->h());
          cv::Mat sample_crop(sample_resized_temp, crop_rect);
          sample_resized = sample_crop;
        }
      }
      cv::Mat sample_normalized_temp;
      cv::Mat sample_normalized;
      if (FLAGS_normalize == 2) {
        cv::Mat sample_float;
        if (in_c_ == 3) {
          sample_resized.convertTo(sample_float, CV_32FC3);
        } else if (in_c_ == 1) {
          sample_resized.convertTo(sample_float, CV_32FC1);
        } else {
          LOG(ERROR) << "input channel err";
        }
        if (need_mean_ == true) {
          cv::subtract(sample_float, mean_, sample_normalized_temp);
          cv::divide(sample_normalized_temp, stdt_, sample_normalized);
        } else {
          sample_normalized = sample_float;
        }
        if (scale_ != 1.0) {
          sample_normalized *= scale_;
        }
      } else {
        cv::Mat sample_float;
        if (in_c_ == 3) {
          sample_resized.convertTo(sample_float, CV_8UC3);
        } else if (in_c_ == 1) {
          sample_resized.convertTo(sample_float, CV_8UC1);
        } else {
          LOG(ERROR) << "input channel err";
        }
        sample_normalized = sample_float;
      }

      // hwc to chw
      for (int c = 0; c < in_c_; c++) {
        for (int h = 0; h < in_h_; h++) {
          const datatype* p = reinterpret_cast<const datatype*>((sample_normalized.ptr(h)));
          for (int w = 0; w < in_w_; w++) {
            input_data[c * in_h_ * in_w_ + h * in_w_ + w] = p[in_c_ * w + c];
          }
        }
      }
      input_data += (in_c_ * in_h_ * in_w_);
    }
  }
}

template<class datatype>
void DataProvider<datatype>::Preprocess(const std::vector<cv::Mat>& imgs) {
  /* Convert the input image to the input image format of the network. */
  datatype* input_data = reinterpret_cast<datatype*>(cpy_data_);
  for (int i = 0; i < imgs.size(); ++i) {
    cv::Mat sample;
    if (imgs[i].channels() == 3 && in_c_ == 1)
      cv::cvtColor(imgs[i], sample, cv::COLOR_BGR2GRAY);
    else if (imgs[i].channels() == 4 && in_c_ == 1)
      cv::cvtColor(imgs[i], sample, cv::COLOR_BGRA2GRAY);
    else if (imgs[i].channels() == 4 && in_c_ == 3)
      cv::cvtColor(imgs[i], sample, cv::COLOR_BGRA2BGR);
    else if (imgs[i].channels() == 1 && in_c_ == 3)
      cv::cvtColor(imgs[i], sample, cv::COLOR_GRAY2BGR);
    else
      sample = imgs[i];
    //squared phote
    int short_edge = std::min(sample.rows, sample.cols);
    int yy = (int)((sample.rows - short_edge) / 2);
    int xx = (int)((sample.cols - short_edge) / 2);
    cv::Rect rect(xx, yy, short_edge, short_edge);
    cv::Mat sample_squared(sample, rect);
    //resize or crop
    cv::Mat sample_resized, sample_resized_temp;
    if (short_edge < inferencer_->h()) {
      cv::resize(sample_squared, sample_resized, input_geometry_);
    } else {
      if (short_edge != 256) {
        cv::resize(sample_squared, sample_resized_temp, cv::Size(256, 256));
      } else {
        sample_resized_temp = sample_squared;
      }
      int crop_off = (int)((256 - inferencer_->h()) / 2);
      if (crop_off < 0) {
        cv::resize(sample_resized_temp, sample_resized, input_geometry_);
      } else {
        cv::Rect crop_rect(
            crop_off, crop_off, inferencer_->w(), inferencer_->h());
        cv::Mat sample_crop(sample_resized_temp, crop_rect);
        sample_resized = sample_crop;
      }
    }
    cv::Mat sample_normalized_temp;
    cv::Mat sample_normalized;
    if (FLAGS_normalize == 2) {
      cv::Mat sample_float;
      if (in_c_ == 3) {
        sample_resized.convertTo(sample_float, CV_32FC3);
      } else if (in_c_ == 1) {
        sample_resized.convertTo(sample_float, CV_32FC1);
      } else {
          LOG(ERROR) << "input channel err";
      }
      if (need_mean_ == true) {
        cv::subtract(sample_float, mean_, sample_normalized_temp);
        cv::divide(sample_normalized_temp, stdt_, sample_normalized);
      } else {
        sample_normalized = sample_float;
      }
      if (scale_ != 1.0) {
        sample_normalized *= scale_;
      }
    } else {
      cv::Mat sample_float;
      if (in_c_ == 3) {
        sample_resized.convertTo(sample_float, CV_8UC3);
      } else if (in_c_ == 1) {
        sample_resized.convertTo(sample_float, CV_8UC1);
      } else {
          LOG(ERROR) << "input channel err";
      }
      sample_normalized = sample_float;
    }

    // hwc to chw
    for (int c = 0; c < in_c_; c++) {
      for (int h = 0; h < in_h_; h++) {
        const datatype* p = reinterpret_cast<const datatype*>((sample_normalized.ptr(h)));
        for (int w = 0; w < in_w_; w++) {
          input_data[c * in_h_ * in_w_ + h * in_w_ + w] = p[in_c_ * w + c];
        }
      }
    }
    input_data += (in_c_ * in_h_ * in_w_);
  }
}

template<class datatype>
void DataProvider<datatype>::SetMean(const string& mean_file,
                           const string& mean_value) {
  if (need_mean_ == false)
    return;

  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    cv::Scalar channel_mean;
    unsigned int mean_size = in_c_ * in_h_ * in_w_;
    float* mean_data = new float[mean_size];
    std::ifstream fin(mean_file.c_str(), std::ios::in);
    for (unsigned int i = 0; i < mean_size; i++) {
      fin >> mean_data[i];
    }
    fin.close();

    /* The format of the mean file is planar 32-bit float RGB or grayscale. */
    std::vector<cv::Mat> channels;
    for (int i = 0; i < in_c_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(in_h_, in_w_, CV_32FC1, mean_data);
      channels.push_back(channel);
      mean_data += in_h_ * in_w_;
    }
    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);
    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    delete [] mean_data;
  }

  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == in_c_) <<
      "Specify either 1 mean_value or as many as channels: " << in_c_;
    std::vector<cv::Mat> channels;
    for (int i = 0; i < in_c_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

template <class datatype>
void DataProvider<datatype>::SetStdt(const string& stdt_value) {
  if (!stdt_value.empty()) {
    stringstream ss(stdt_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == in_c_)
        << "Specify either 1 mean_value or as many as channels: " << in_c_;
    std::vector<cv::Mat> channels;
    for (int i = 0; i < in_c_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height,
                      input_geometry_.width,
                      CV_32FC1,
                      cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, stdt_);
  } else {
    /* default stdt = '1,1,1' */
    std::vector<cv::Mat> channels;
    for (int i = 0; i < in_c_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height,
                      input_geometry_.width,
                      CV_32FC1,
                      cv::Scalar(1.0));
      channels.push_back(channel);
    }
    cv::merge(channels, stdt_);
  }
}

Inferencer::Inferencer(
    const string& offlinemodel,
    const int& thread_id,
    const int& data_parallelism) {

  invoke_time = 0.;
  running_  = true;
  thread_id_ = thread_id;
  data_parallelism_ = data_parallelism;
  unsigned int dev_num;
  cnrtGetDeviceCount(&dev_num);
  if (dev_num == 0) {
    LOG(ERROR) << "no device found";
    exit(-1);
  }
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, 0));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  func_type_ = CNRT_FUNC_TYPE_BLOCK;
  switch (FLAGS_data_parallelism) {
    case 1:
      func_type_ = CNRT_FUNC_TYPE_BLOCK;
      break;
    case 2:
      func_type_ = CNRT_FUNC_TYPE_BLOCK1;
      break;
    case 4:
      func_type_ = CNRT_FUNC_TYPE_UNION1;
      break;
    case 8:
      func_type_ = CNRT_FUNC_TYPE_UNION2;
      break;
    case 16:
      func_type_ = CNRT_FUNC_TYPE_UNION4;
      break;
    case 32:
      func_type_ = CNRT_FUNC_TYPE_UNION8;
      break;
    default:
      LOG(ERROR) << "not support data_parallelism: " << FLAGS_data_parallelism;
      exit(-1);
  }

  // func_type_ = CNRT_FUNC_TYPE_BLOCK;
  if (FLAGS_data_parallelism * FLAGS_model_parallelism <= 8) {
    cnrtSetCurrentChannel((cnrtChannelType_t)thread_id_);
    // func_type_ = CNRT_FUNC_TYPE_UNION2;
  }

  // 2. load model and get function
  LOG(INFO) << "load file: " <<  offlinemodel.c_str();
  cnrtLoadModel(&model_, offlinemodel.c_str());
  string name = "fusion_0";
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model_, name.c_str());
  // 3. get function's I/O DataDesc
  cnrtGetInputDataDesc(&inputDescS_, &inputNum , function);
  cnrtGetOutputDataDesc(&outputDescS_, &outputNum, function);
  // 4. allocate I/O data space on CPU memory and prepare Input data
  int in_count;

  LOG(INFO) << "input blob num is " << inputNum;
  for (int i = 0; i < inputNum; i++) {
    unsigned int in_n, in_c, in_h, in_w;
    cnrtDataDesc_t inputDesc = inputDescS_[i];
    cnrtGetHostDataCount(inputDesc, &in_count);
    if (FLAGS_normalize == 0 || FLAGS_normalize == 1) {
      cnrtSetHostDataLayout(inputDesc, CNRT_UINT8, CNRT_NCHW);
    } else {
      cnrtSetHostDataLayout(inputDesc, CNRT_FLOAT32, CNRT_NCHW);
    }
    cnrtGetDataShape(inputDesc, &in_n, &in_c, &in_h, &in_w);
    CHECK(FLAGS_batch == in_n)
      << "the input batch != the input batch extracted from the offline model."
      << std::endl;
    in_count *= FLAGS_data_parallelism;
    in_n *= FLAGS_data_parallelism;
    LOG(INFO) << "shape " << in_n;
    LOG(INFO) << "shape " << in_c;
    LOG(INFO) << "shape " << in_h;
    LOG(INFO) << "shape " << in_w;
    if (i == 0) {
      in_n_ = in_n;
      in_c_ = in_c;
      in_w_ = in_w;
      in_h_ = in_h;
    } else {
      cnrtGetHostDataCount(inputDesc, &in_count);
    }
  }

  for (int i = 0; i < outputNum; i++) {
    cnrtDataDesc_t outputDesc = outputDescS_[i];
    cnrtSetHostDataLayout(outputDesc, CNRT_FLOAT32, CNRT_NCHW);
    cnrtGetHostDataCount(outputDesc, &out_count_);
    cnrtGetDataShape(outputDesc, &out_n_, &out_c_, &out_h_, &out_w_);
    out_count_ *= FLAGS_data_parallelism;
    out_n_ *= FLAGS_data_parallelism;
    LOG(INFO) << "output shape " << out_n_;
    LOG(INFO) << "output shape " << out_c_;
    LOG(INFO) << "output shape " << out_h_;
    LOG(INFO) << "output shape " << out_w_;
  }

  // 5. allocate I/O data space on MLU memory and copy Input data
  void** inputMluPtrS;
  void** outputMluPtrS;
  cnrtMallocBatchByDescArray(
      &inputMluPtrS ,
      inputDescS_,
      inputNum,
      FLAGS_data_parallelism);
  cnrtMallocBatchByDescArray(
      &outputMluPtrS,
      outputDescS_,
      outputNum,
      FLAGS_data_parallelism);

  freeInputFifo_.push(inputMluPtrS);
  freeOutputFifo_.push(outputMluPtrS);

  cnrtMallocBatchByDescArray(
      &inputMluPtrS ,
      inputDescS_,
      inputNum,
      FLAGS_data_parallelism);
  cnrtMallocBatchByDescArray(
      &outputMluPtrS,
      outputDescS_,
      outputNum,
      FLAGS_data_parallelism);

  freeInputFifo_.push(inputMluPtrS);
  freeOutputFifo_.push(outputMluPtrS);

  dim_ = {1, 1, 1};
  invoke_param_.data_parallelism = &FLAGS_data_parallelism;
  invoke_param_.end = CNRT_PARAM_END;
}

void** Inferencer::popFreeInputData() {
  return freeInputFifo_.pop();
}

void** Inferencer::popValidInputData() {
  return validInputFifo_.pop();
}

void Inferencer::pushFreeInputData(void** data) {
  freeInputFifo_.push(data);
}

void Inferencer::pushValidInputData(void** data) {
  validInputFifo_.push(data);
}

void** Inferencer::popFreeOutputData() {
  return freeOutputFifo_.pop();
}

void** Inferencer::popValidOutputData() {
  return validOutputFifo_.pop();
}

void Inferencer::pushFreeOutputData(void** data) {
  freeOutputFifo_.push(data);
}

void Inferencer::pushValidOutputData(void** data) {
  validOutputFifo_.push(data);
}

void Inferencer::pushValidInputNames(vector<string> images) {
  imagesFifo_.push(images);
}

vector<string> Inferencer::popValidInputNames() {
  return imagesFifo_.pop();
}

void Inferencer::notify() {
  running_ = false;
}

void Inferencer::run() {
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, 0));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  // func_type_ = CNRT_FUNC_TYPE_BLOCK;
  if (FLAGS_data_parallelism * FLAGS_model_parallelism <= 8) {
    cnrtSetCurrentChannel((cnrtChannelType_t)thread_id_);
    // func_type_ = CNRT_FUNC_TYPE_UNION2;
  }

  cnrtFunction_t function_;
  cnrtCreateFunction(&function_);
  cnrtCopyFunction(&function_, function);
  bool muta = false;
  cnrtInitFuncParam_t init_param;
  init_param.muta = &muta;
  init_param.data_parallelism = &FLAGS_data_parallelism;
  init_param.end = CNRT_PARAM_END;

  // 6. initialize function memory
  cnrtInitFunctionMemory_V2(function_, &init_param);

  cnrtCreateStream(&stream_);
  // initliaz function memory
  cnrtInitFunctionMemory_V2(function_, &init_param);
  // create start_event and end_event
  cnrtEvent_t event_start, event_end;
  cnrtCreateEvent(&event_start);
  cnrtCreateEvent(&event_end);
  float event_time_use;
  // void **param = (void **)malloc(sizeof(void *) * (inputNum + outputNum));
  void **param = reinterpret_cast<void **>(malloc(sizeof(void *) * (inputNum + outputNum)));
  while (running_ || validInputFifo_.size()) {
    void** mlu_input_data = validInputFifo_.pop();
    void** mlu_output_data = freeOutputFifo_.pop();
    // LOG(INFO) << "Invoke function ...";
    for (int i = 0; i < inputNum; i++) {
      param[i] = mlu_input_data[i];
    }
    for (int i = 0; i < outputNum; i++) {
      param[inputNum + i] = mlu_output_data[i];
    }
    cnrtPlaceEvent(event_start, stream_);
    // CNRT_CHECK(cnrtInvokeFunction(function_, dim_, param,
    //             func_type_, stream_, (void *)&invoke_param_));
    CNRT_CHECK(cnrtInvokeFunction(function_, dim_, param,
                       func_type_, stream_, reinterpret_cast<void*>(&invoke_param_)));
    cnrtPlaceEvent(event_end, stream_);
    if (cnrtSyncStream(stream_) == CNRT_RET_SUCCESS) {
      cnrtEventElapsedTime(event_start, event_end, &event_time_use);
      invoke_time += event_time_use;
      LOG(INFO) << " execution time: " << event_time_use << " us";
    } else {
      LOG(ERROR) << " SyncStream error";
    }
    // LOG(INFO) << "after Invoke function ...";
    // LOG(INFO) << "Invoke function input addr " << mlu_input_data[0];
    // LOG(INFO) << "Invoke function output addr " << mlu_output_data[0];

    pushValidOutputData(mlu_output_data);
    pushFreeInputData(mlu_input_data);
  }

  post_processor_->notify();

  free(param);
  cnrtDestroyEvent(&event_start);
  cnrtDestroyEvent(&event_end);
  cnrtDestroyFunction(function_);
}

PostProcessor::PostProcessor() {
  running_ = true;
  thread_id_ = 0;
  top1_ = 0;
  top5_ = 0;
  total_ = 0;
}

void PostProcessor::notify() {
  running_ = false;
}

string PostProcessor::get_string(string names, int index){
  std::ifstream files(names.c_str(), std::ios::in);
  string file;
  for (int i = 0; i < index; i++) {
    getline(files, file);
  }
  getline(files, file);
  files.close();
  return file;
}

void PostProcessor::top5_print(float* result_buffer, unsigned int count,
    const string& image) {
  unsigned int max_index[5] = { 0 };
  float max_num[5] = { 0 };
  for (unsigned int i = 0; i < count; i++) {
    float new_data = result_buffer[i];
    unsigned int new_index = i;
    for (unsigned int j = 0; j < 5; j++) {
      if (new_data > max_num[j]) {
        std::swap(max_num[j], new_data);
        std::swap(max_index[j], new_index);
      }
    }
  }

  if (FLAGS_debug == "true") {
    stringstream top5_infos;
    top5_infos << "------------classify " << image << " ------------\n";
    top5_infos << "the answer is " << get_string(FLAGS_synset, accurate_labels_[image]);
    top5_infos << "\ntop5       is ";
    for(int i = 0; i < 5; i++) {
      top5_infos << result_buffer[max_index[i]] << " - "
        << get_string(FLAGS_synset, max_index[i]);
      top5_infos << "\n              ";
    }
    std::cout << top5_infos.str();
  }

  if (max_index[0] == accurate_labels_[image]) {
    top1_++;
    top5_++;
  }

  for (int i = 1; i < 5; i++) {
    if (max_index[i] == accurate_labels_[image]) {
      top5_++;
    }
  }

  total_++;
}

void PostProcessor::get_label(const string& label_file) {
  int value, order;
  std::string key;
  std::string img_to_label;
  std::ifstream first(label_file.c_str(), std::ios::in);
  while (first) {
    std::getline(first, img_to_label);
    order = img_to_label.find(" ");
    key = img_to_label.substr(0, order);
    value = std::atoi(img_to_label.substr(order + 1, img_to_label.size()).c_str());
    auto ret_pr = accurate_labels_.insert(std::pair<std::string, int>(key, value));
    if (ret_pr.second == false) {
      std::cout << " keys of map are repeated !"<< std::endl;
      exit(0);
    }
  }
}

void PostProcessor::run() {
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, 0));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  if (FLAGS_data_parallelism * FLAGS_model_parallelism<= 8) {
    cnrtSetCurrentChannel((cnrtChannelType_t)thread_id_);
  }

  get_label(FLAGS_label_file);
  void** outputCpuPtrS;
  outputCpuPtrS = reinterpret_cast<void**>(malloc (sizeof(void*) * 1));
  outputCpuPtrS[0] = reinterpret_cast<void*>
      (malloc(sizeof(float) * inferencer_->out_count_));
  while (running_ || inferencer_->validOutputFifo_.size()) {
    void** mlu_output_data = inferencer_->validOutputFifo_.pop();
    // LOG(INFO) << "memcpy to host ...";
    // LOG(INFO) << "memcpy to host ... mlu addr " << mlu_output_data[0];
    CNRT_CHECK(cnrtMemcpyBatchByDescArray(
        outputCpuPtrS, mlu_output_data, inferencer_->outputDescS_,
        1, FLAGS_data_parallelism, CNRT_MEM_TRANS_DIR_DEV2HOST));

    inferencer_->pushFreeOutputData(mlu_output_data);

    float* result = reinterpret_cast<float*>(outputCpuPtrS[0]);
    unsigned int output_chw = inferencer_->output_chw();
    vector<string> origin_img = inferencer_->popValidInputNames();
    for (unsigned int i = 0; i < inferencer_->n(); i++) {
      if (origin_img[i] != "null") {
        top5_print(result, output_chw, origin_img[i]);
      }
      result += output_chw;
    }
  }

  free(outputCpuPtrS[0]);
  free(outputCpuPtrS);
  while (inferencer_->freeInputFifo_.size()) {
    cnrtFreeArray(inferencer_->freeInputFifo_.pop(), inferencer_->inputNum);
  }
  while (inferencer_->freeOutputFifo_.size()) {
    cnrtFreeArray(inferencer_->freeOutputFifo_.pop(), inferencer_->outputNum);
  }
  cnrtDestroyStream(inferencer_->stream_);
  cnrtDestroyFunction(inferencer_->function);
  cnrtUnloadModel(inferencer_->model_);
}

template<class datatype>
class Pipeline {
 public:
  Pipeline(const string& offlinemodel,
            const string& mean_file,
            const string& mean_value,
            const int& thread_id,
            const int& data_parallelism,
            queue<string> images,
            float scale = 1.0);
  ~Pipeline();

  DataProvider<datatype>* data_provider_;
  Inferencer* inferencer_;
  PostProcessor* post_processor_;
  void run();
};

template<class datatype>
Pipeline<datatype>::Pipeline(const string& offlinemodel,
                   const string& mean_file,
                   const string& mean_value,
                   const int& thread_id,
                   const int& data_parallelism,
                   queue<string> images,
                   float scale) {
  inferencer_ = new Inferencer(offlinemodel,
      thread_id,
      data_parallelism);
  data_provider_ = new DataProvider<datatype>(
      mean_file,
      mean_value,
      images,
      scale);
  post_processor_ = new PostProcessor();
  data_provider_->inferencer_ = inferencer_;
  post_processor_->inferencer_ = inferencer_;
  inferencer_->post_processor_ = post_processor_;
  data_provider_->thread_id_ = thread_id;
  post_processor_->thread_id_ = thread_id;
  inferencer_->thread_id_ = thread_id;
#ifdef PRE_READ
  data_provider_->preRead();
#endif
}

template<class datatype>
Pipeline<datatype>::~Pipeline() {
  if (data_provider_ != nullptr) {
    delete data_provider_;
    data_provider_ = nullptr;
  }
  if (inferencer_ != nullptr) {
    delete inferencer_;
    inferencer_ = nullptr;
  }
  if (post_processor_ != nullptr) {
    delete post_processor_;
    post_processor_ = nullptr;
  }
}

template<class datatype>
void Pipeline<datatype>::run() {
  vector<thread*> threads(3, nullptr);
  threads[0] = new thread(&DataProvider<datatype>::run, data_provider_);
  threads[1] = new thread(&Inferencer::run, inferencer_);
  threads[2] = new thread(&PostProcessor::run, post_processor_);
  for (auto th : threads)
    th->join();
  for (auto &th : threads) {
    if (th != nullptr) {
      delete th;
      th = nullptr;
    }
  }
}

void check_args(void) {
  if (FLAGS_use_mean != "on" && FLAGS_use_mean != "off") {
    LOG(ERROR) << "use_mean should be set on or off";
    exit(-1);
  }
  if (FLAGS_data_parallelism != 1 && FLAGS_data_parallelism != 2 && FLAGS_data_parallelism != 4 &&\
      FLAGS_data_parallelism != 8 && FLAGS_data_parallelism != 16 && FLAGS_data_parallelism != 32) {
    LOG(ERROR) << "data_parallelism should be set 1,2,4,8,16,32";
    exit(-1);
  }
  if (FLAGS_model_parallelism != 1 && FLAGS_model_parallelism != 2 && FLAGS_model_parallelism != 4 &&\
      FLAGS_model_parallelism != 8 && FLAGS_model_parallelism != 16 && FLAGS_model_parallelism != 32) {
    LOG(ERROR) << "model_parallelism should be set 1,2,4,8,16,32";
    exit(-1);
  }
  if (FLAGS_threads < 1 || FLAGS_threads > 4) {
    LOG(ERROR) << "threads should be set 1,2,3,4";
    exit(-1);
  }
  if (FLAGS_data_parallelism * FLAGS_threads * FLAGS_model_parallelism > 32) {
    LOG(ERROR) << "\nif model_parallelism * data_parallelism is 16, then mparallel must be in [1 or 2]\n"\
      << "if model_parallelism * data_parallelism is 32, then mparallel must be 1";
    exit(-1);
  }
  if (FLAGS_fix8 != 0 && FLAGS_fix8 != 1) {
    LOG(ERROR) << "fix8 should be set 0 or 1";
    exit(-1);
  }
  if (FLAGS_batch != 1 && FLAGS_batch != 2 && FLAGS_batch != 4) {
    LOG(ERROR) << "batch should be set 1,2,4";
    exit(-1);
  }
  return;
}

template<class datatype>
void Process(vector<queue<string>> img_list) {
  vector<thread*> pipelines;
  vector<Pipeline<datatype>*> pipeline_instances;

  for (int i = 0; i < FLAGS_threads; i++) {
    if (img_list[i].size()) {
      Pipeline<datatype>* pipeline = new Pipeline<datatype>(FLAGS_offlinemodel,
          FLAGS_mean_file,
          FLAGS_mean_value,
          i,
          FLAGS_data_parallelism,
          img_list[i],
          FLAGS_scale);
      pipeline_instances.push_back(pipeline);
      pipelines.push_back(new thread(&Pipeline<datatype>::run, pipeline));
    }
  }
  double time_use;
  struct timeval tpend, tpstart;
  gettimeofday(&tpstart, NULL);
  {
    std::lock_guard<std::mutex> lk(condition_m);
    ready_start = true;
    LOG(INFO) << "Notify to start ...";
  }
  condition.notify_all();
  for (int i = 0; i < FLAGS_threads; i++) {
    pipelines[i]->join();
  }
  gettimeofday(&tpend, NULL);

  unsigned int top1_hit = 0;
  unsigned int top5_hit = 0;
  unsigned int total = 0;
  double avg_invoke_time = 0.;
  for (int i = 0; i < FLAGS_threads; i++) {
    top1_hit += pipeline_instances[i]->post_processor_->top1_;
    top5_hit += pipeline_instances[i]->post_processor_->top5_;
    total += pipeline_instances[i]->post_processor_->total_;
    avg_invoke_time += pipeline_instances[i]->inferencer_->invoke_time;
  }

  std::cout << "---------------------" << std::endl;
  //  std::cout << "classifying " << total << " images, and top1_hit = "
  //  << top1_hit / (double)total << " , top5_hit = "
  //  << top5_hit / (double)total << std::endl;
  std::cout << "classifying " << total << " images, and top1_hit = "
    << top1_hit / static_cast<double>(total) << " , top5_hit = "
    << top5_hit / static_cast<double>(total) << std::endl;

  // avg_invoke_time = avg_invoke_time / (double)FLAGS_threads;
  avg_invoke_time = avg_invoke_time / static_cast<double>(FLAGS_threads);
  std::cout << "FLAGS_data_parallelism:" << FLAGS_data_parallelism
    << " and there are " << FLAGS_threads << " threads."<< std::endl;
  for (int i = 0; i < FLAGS_threads; i++) {
    std::cout << "event time sum = "
              << pipeline_instances[i]->inferencer_->invoke_time << std::endl;
  }
  std::cout << "for only inference, " << total * (1000000 / avg_invoke_time)
    << " fps" << std::endl;

  time_use = 1000000 * (tpend.tv_sec - tpstart.tv_sec)
    + tpend.tv_usec - tpstart.tv_usec;
  std::cout << "for end2end, " << total * (1000000 / time_use)
    << " fps" << std::endl;

  for (int i = 0; i < FLAGS_threads; i++) {
    if (pipelines[i] != nullptr) {
      delete pipelines[i];
      pipelines[i] = nullptr;
    }
    if (pipeline_instances[i] != nullptr) {
      delete pipeline_instances[i];
      pipeline_instances[i] = nullptr;
    }
  }
  cnrtDestroy();
  return;
}

int main(int argc, char* argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::ifstream files_tmp(FLAGS_images.c_str(), std::ios::in);
  int image_num = 0;
  vector<string> files;
  std::string line_tmp;
  vector<queue<string>> img_list(FLAGS_threads);
  if (files_tmp.fail()) {
    LOG(ERROR) << "open " << FLAGS_images  << " file fail!";
    return 1;
  } else {
    while (getline(files_tmp, line_tmp)) {
      img_list[image_num%FLAGS_threads].push(line_tmp);
      image_num++;
    }
  }
  files_tmp.close();
  std::cout << "there are " << image_num
    << " images in " << FLAGS_images << std::endl;

  check_args();
  cnrtInit(0);

  if (FLAGS_normalize == 0 || FLAGS_normalize == 1) {
    Process<uint8_t>(img_list);
  } else {
    Process<float>(img_list);
  }
}
#endif
