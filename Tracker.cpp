#include <iostream>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>

#include "NvInfer.h"
#include "NvInferRuntimeBase.h"
#include <chrono>

#include <vector>
#include <fstream>
#include <valarray>
#include <cmath>
#include <map>
#include <string>

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override;
};

void Logger::log(Severity severity, const char* msg) noexcept {
	// Only log Warnings or more important.
	//if (severity <= Severity::kWARNING) {
		std::cout << msg << std::endl;
	//}
}

std::vector<float> mat_to_chw(const cv::Mat& image)
{
	int height = image.rows;
	int width = image.cols;
	int channels = image.channels();
	
	std::vector<float> chw_data(channels * height * width);
	
	for (int c = 0; c < channels; ++c) {
		for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
				chw_data[c * height * width + h * width + w] = 
					static_cast<float>(image.at<cv::Vec3b>(h, w)[c]);
			}
		}
	}
	return chw_data;
}

cv::Mat chw_to_mat(const std::vector<float>& chw_data, int channels, int height, int width)
{
	cv::Mat image(height, width, CV_32FC3);
	for (int c = 0; c < channels; ++c) {
		for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
				image.at<cv::Vec3b>(h, w)[c] = static_cast<uchar>(chw_data[c * height * width + h * width + w]);
			}
		}
	}
	return image;
}

std::tuple<cv::Mat, std::valarray<float>> sample_patch_cv(
	const cv::Mat& im,
	const std::valarray<float>& pos,
	const std::valarray<float>& sample_sz,
	const std::valarray<float>& output_sz,
	const std::string& mode = "replicate")
{
	int tl_x = static_cast<int>(std::round(pos[1] - (sample_sz[1] - 1) / 2.0));
	int tl_y = static_cast<int>(std::round(pos[0] - (sample_sz[0] - 1) / 2.0));
	int br_x = static_cast<int>(std::round(pos[1] + sample_sz[1] / 2.0 + 1));
	int br_y = static_cast<int>(std::round(pos[0] + sample_sz[0] / 2.0 + 1));

	int pad_left = std::max(0, -tl_x);
	int pad_top = std::max(0, -tl_y);
	int pad_right = std::max(0, br_x - im.cols);
	int pad_bottom = std::max(0, br_y - im.rows);

	int valid_tl_x = std::max(0, tl_x);
	int valid_tl_y = std::max(0, tl_y);
	int valid_br_x = std::min(im.cols, br_x);
	int valid_br_y = std::min(im.rows, br_y);

	cv::Rect roi(valid_tl_x, valid_tl_y, valid_br_x - valid_tl_x, valid_br_y - valid_tl_y);
	cv::Mat cropped = im(roi).clone();

	int borderType = (mode == "replicate") ? cv::BORDER_REPLICATE : cv::BORDER_CONSTANT;
	if (pad_left > 0 || pad_top > 0 || pad_right > 0 || pad_bottom > 0) {
		cv::copyMakeBorder(cropped, cropped, pad_top, pad_bottom, pad_left, pad_right, borderType, cv::Scalar(0, 0, 0));
	}

	cv::Mat final_patch;
	if (output_sz.size() >= 2) {
		cv::Size out_size(static_cast<int>(output_sz[1]), static_cast<int>(output_sz[0]));
		cv::resize(cropped, final_patch, out_size, 0, 0, cv::INTER_LINEAR);
	} else {
		final_patch = cropped;
	}

	std::valarray<float> patch_coord = { (float)tl_y, (float)tl_x, (float)br_y, (float)br_x };
	return std::make_tuple(final_patch, patch_coord);
}

struct Transform {
	std::vector<int64_t> output_sz;
	std::vector<int64_t> shift;

	Transform(const std::vector<int64_t> output_sz, const std::vector<int64_t> shift = { 0,0 }) : output_sz(output_sz), shift(shift) {}
};

std::tuple<float, std::valarray<float>> max2dd_val(const std::valarray<std::valarray<float>>& a) {
	std::valarray<float> max_val_col(a.size());
	std::valarray<int64_t> argmax_col(a.size());
	for (size_t i = 0; i < a.size(); i++) {
		max_val_col[i] = *std::max_element(std::begin(a[i]), std::end(a[i]));
		argmax_col[i] = std::distance(std::begin(a[i]), std::max_element(std::begin(a[i]), std::end(a[i])));
	}

	float max_val = *std::max_element(std::begin(max_val_col), std::end(max_val_col));
	int64_t argmax_row = std::distance(std::begin(max_val_col), std::max_element(std::begin(max_val_col), std::end(max_val_col)));

	int64_t argmax_col_col = argmax_col[argmax_row];

	std::valarray<float> argmax_out({ static_cast<float>(argmax_row), static_cast<float>(argmax_col_col) });

	return std::make_tuple(max_val, argmax_out);
}

std::valarray<float> clip_bbox_to_image_area_val(std::valarray<float> bbox, const cv::Mat& image, float minwidth = 10, float minheight = 10)
{
	float H = static_cast<float>(image.rows);
	float W = static_cast<float>(image.cols);

	auto x1 = std::max(float(0), std::min((bbox[0]), W - minwidth));
	auto y1 = std::max(float(0), std::min((bbox[1]), H - minheight));
	auto x2 = std::max(x1 + minwidth, std::min((bbox[0]) + (bbox[2]), W));
	auto y2 = std::max(y1 + minheight, std::min((bbox[1]) + (bbox[3]), H));

	std::valarray<float> clipped_box({ x1, y1, x2 - x1, y2 - y1 });

	return clipped_box;
}

std::string mode = "init";
std::vector<int> coord(4, 0);
bool new_init = false;

static void MouseCallback(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN && mode == "init") {
		coord[0] = x;
		coord[1] = y;
		coord[2] = x;
		coord[3] = y;
		mode = "select";
	}
	else if (event == cv::EVENT_MOUSEMOVE && mode == "select") {
		coord[2] = x;
		coord[3] = y;
	}
	else if (event == cv::EVENT_LBUTTONDOWN && mode == "select") {
		if (coord[0] > x) {
			coord[2] = coord[0];
			coord[0] = x;
		}
		else {
			coord[2] = x;
		}
		if (coord[1] > y) {
			coord[3] = coord[1];
			coord[1] = y;
		}
		else {
			coord[3] = y;
		}
		mode = "init";
		new_init = true;
	}
}

cv::Rect getCoordRect() {
	return cv::Rect(coord[0], coord[1], coord[2] - coord[0], coord[3] - coord[1]);
}

std::vector<int> get_bb()
{
	std::vector<int> bb(4);
	bb[0] = std::min(coord[0], coord[2]);      // x (left)
	bb[1] = std::min(coord[1], coord[3]);      // y (top)
	bb[2] = std::abs(coord[2] - coord[0]);     // width
	bb[3] = std::abs(coord[3] - coord[1]);     // height
	return bb;
}

auto train_feature_size = 18;
auto feature_stride = 16;
auto image_sample_size = train_feature_size * feature_stride;
int search_area_scale = 5;
int augmentation_expansion_factor = 2;
double random_shift_factor = 1 / 3;
int sample_memory_size = 2;

struct trte {
	nvinfer1::ICudaEngine* fe_engine;
	nvinfer1::IExecutionContext* fe_context;
	nvinfer1::IRuntime* fe_runtime;

	nvinfer1::ICudaEngine* hfe_engine;
	nvinfer1::IExecutionContext* hfe_context;
	nvinfer1::IRuntime* hfe_runtime;

	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;
	nvinfer1::IRuntime* runtime;
	cudaStream_t stream;

	std::valarray<float> pos;
	std::valarray<float> feature_sz;
	std::valarray<int64_t> kernel_size;
	float target_scale;
	std::valarray<float> img_support_sz;
	std::valarray<float> scale_factors;
	std::valarray<float> img_sample_sz;
	std::valarray<int64_t> img_sample_sz_val_int;
	std::string border_mode;
	float max_scale_change;

	float* train_feat_d = nullptr;
	float* target_labels_d = nullptr;
	float* train_ltrb_d = nullptr;
	float* scores_raw_d = nullptr;
	std::valarray<float> scores_raw_val;
	float* bbox_preds_d = nullptr;

	std::valarray<float> train_img_sample_sz;

	std::valarray<float> target_sz;
	std::valarray<float> base_target_sz;

	std::valarray<float> init_sample_pos;

	Transform* transform = nullptr;

	float* target_boxes_d = nullptr;
	float* training_samples_d = nullptr;
	float* im_patch_d = nullptr;

	std::valarray<float> precomputed_get_centered_sample_pos_val;
	std::vector<float> target_boxes_host;
	std::vector<float> train_ltrb_host;

	~trte() {
		if(train_feat_d) cudaFree(train_feat_d);
		if(target_labels_d) cudaFree(target_labels_d);
		if(train_ltrb_d) cudaFree(train_ltrb_d);
		if(scores_raw_d) cudaFree(scores_raw_d);
		if(bbox_preds_d) cudaFree(bbox_preds_d);
		if(training_samples_d) cudaFree(training_samples_d);
		if(im_patch_d) cudaFree(im_patch_d);
		if(target_boxes_d) cudaFree(target_boxes_d);
		if(transform) delete transform;
	}

	void initialize(cv::Mat image, std::vector<int> state)
	{
		std::cout << "--------------initialize start----------------------" << std::endl;

		this->pos = { static_cast<float>(state[1]) + (static_cast<float>(state[3]) - 1.0f) / 2.0f, 
		              static_cast<float>(state[0]) + (static_cast<float>(state[2]) - 1.0f) / 2.0f };
		this->target_sz = { static_cast<float>(state[3]), static_cast<float>(state[2]) };

		this->img_sample_sz = { static_cast<float>(image_sample_size), static_cast<float>(image_sample_size) };
		this->img_support_sz = this->img_sample_sz;
		this->img_sample_sz_val_int = std::valarray<int64_t>({ (int64_t)image_sample_size, (int64_t)image_sample_size });

		this->train_img_sample_sz = { (float)(train_feature_size * feature_stride), (float)(train_feature_size * feature_stride) };

		float search_area = (this->target_sz[0] * search_area_scale) * (this->target_sz[1] * search_area_scale);
		this->target_scale = std::sqrt(search_area / (this->img_sample_sz[0] * this->img_sample_sz[1]));
		this->base_target_sz = this->target_sz / this->target_scale;

		std::vector<float> init_backbone_feat;
		generate_init_samples(image, init_backbone_feat);

		init_classifier();

		run_infer_on_engine({ this->training_samples_d, this->train_feat_d }, this->hfe_engine, this->hfe_context);

		this->train_ltrb_host = encode_bbox(this->target_boxes_host);
		cudaMemcpy(this->train_ltrb_d, this->train_ltrb_host.data(), 4 * 18 * 18 * sizeof(float), cudaMemcpyHostToDevice);

		std::valarray<float> ksz_even = { (float)((this->kernel_size[0] + 1) % 2), (float)((this->kernel_size[1] + 1) % 2) };
		this->precomputed_get_centered_sample_pos_val = ksz_even * target_scale * img_support_sz / (2.0f * feature_sz);
		
		this->scale_factors = { 1.0f };

		std::cout << "-----------------initialize end---------------------" << std::endl;
	}

	std::vector<float> encode_bbox(const std::vector<float>& bbox) {
		int stride = feature_stride; 
		int output_sz = image_sample_size; 
		int sz = output_sz / stride; 
		
		std::vector<float> reg_targets(4 * sz * sz);
		float x1 = bbox[0];
		float y1 = bbox[1];
		float x2 = bbox[0] + bbox[2];
		float y2 = bbox[1] + bbox[3];

		for (int i = 0; i < sz; i++) {
			for (int j = 0; j < sz; j++) {
				float px = j * stride + stride / 2.0f;
				float py = i * stride + stride / 2.0f;
				
				float l = (px - x1) / output_sz;
				float t = (py - y1) / output_sz;
				float r = (x2 - px) / output_sz;
				float b = (y2 - py) / output_sz;
				
				reg_targets[0 * sz * sz + i * sz + j] = l;
				reg_targets[1 * sz * sz + i * sz + j] = t;
				reg_targets[2 * sz * sz + i * sz + j] = r;
				reg_targets[3 * sz * sz + i * sz + j] = b;
			}
		}
		return reg_targets;
	}

	void generate_init_samples(const cv::Mat& im, std::vector<float>& output_feat)
	{
		this->init_sample_pos = { std::round(this->pos[0]), std::round(this->pos[1]) };

		std::valarray<float> aug_expansion_sz = this->img_sample_sz * (float)augmentation_expansion_factor;
		std::valarray<int64_t> aug_expansion_sz_int(2);
		aug_expansion_sz_int[0] = static_cast<int64_t>(std::round(aug_expansion_sz[0]));
		aug_expansion_sz_int[1] = static_cast<int64_t>(std::round(aug_expansion_sz[1]));
		
		std::valarray<int64_t> img_sample_sz_int(2);
		img_sample_sz_int[0] = static_cast<int64_t>(std::round(this->img_sample_sz[0]));
		img_sample_sz_int[1] = static_cast<int64_t>(std::round(this->img_sample_sz[1]));

		aug_expansion_sz_int += (aug_expansion_sz_int - img_sample_sz_int) % (int64_t)2;
		
		std::vector<int64_t> aug_output_sz_vec = { img_sample_sz_int[0], img_sample_sz_int[1] };
		if (this->transform != nullptr) delete this->transform;
		this->transform = new Transform(aug_output_sz_vec);

		std::valarray<float> scaled_image_sz = this->img_sample_sz * this->target_scale;
		auto [im_patch_mat, sample_coords_val] = sample_patch_cv(im, this->init_sample_pos, scaled_image_sz, this->img_sample_sz, "inside_major");

		std::vector<float> im_patch_chw = mat_to_chw(im_patch_mat);
		cudaMemcpy(this->im_patch_d, im_patch_chw.data(), im_patch_chw.size() * sizeof(float), cudaMemcpyHostToDevice);

		output_feat.resize(1024 * 18 * 18);
		run_infer_on_engine({ this->im_patch_d, this->training_samples_d }, this->fe_engine, this->fe_context);
		cudaMemcpy(output_feat.data(), this->training_samples_d, 1024 * 18 * 18 * sizeof(float), cudaMemcpyDeviceToHost);
	}

	void init_classifier()
	{
		this->feature_sz = { 18.0f, 18.0f };
		int64_t ksz = 1;
		this->kernel_size = { ksz, ksz };

		init_target_boxes();
		init_target_labels();
	}

	void init_target_boxes()
	{
		std::valarray<float> box_center = (this->pos - this->init_sample_pos) / this->target_scale + (this->img_sample_sz - 1.0f) / 2.0f;
		std::valarray<float> box_sz = this->target_sz / this->target_scale;
		std::valarray<float> target_ul = box_center - (box_sz - 1.0f) / 2.0f;
		
		std::vector<float> classifier_target_box = { target_ul[1], target_ul[0], box_sz[1], box_sz[0] };
		classifier_target_box[0] += this->transform->shift[1];
		classifier_target_box[1] += this->transform->shift[0];
		
		this->target_boxes_host = classifier_target_box;
	}

	std::vector<float> gauss_spatial(float sz, float sigma, float center = 0.0f, float end_pad = 0.0f)
	{
		float start = -(sz - 1) / 2.0f;
		float end = (sz + 1) / 2.0f + end_pad;
		int n = static_cast<int>(std::ceil(end - start));
		std::vector<float> ret(n);
		for (int i = 0; i < n; ++i) {
			float k = start + i;
			ret[i] = std::exp(-1.0f / (2 * sigma * sigma) * (k - center) * (k - center));
		}
		return ret;
	}

	void init_target_labels()
	{
		float feature_sz_prod = feature_sz[0] * feature_sz[1];
		float img_support_sz_prod = img_support_sz[0] * img_support_sz[1];
		float base_target_sz_prod = base_target_sz[0] * base_target_sz[1];

		float sigma_scalar = std::sqrt((feature_sz_prod / img_support_sz_prod) * base_target_sz_prod) * 0.25f;
		
		std::valarray<float> target_center_norm = (this->pos - this->init_sample_pos) / (this->target_scale * this->img_support_sz);
		std::valarray<float> ksz_even = { (float)((this->kernel_size[0] + 1) % 2), (float)((this->kernel_size[1] + 1) % 2) };
		std::valarray<float> center_pos = this->feature_sz * target_center_norm + 0.5f * ksz_even;
		std::valarray<float> transform_shift = { (float)this->transform->shift[0], (float)this->transform->shift[1] };
		std::valarray<float> sample_center = center_pos + (transform_shift / this->img_support_sz * this->feature_sz);

		std::vector<float> gauss_x = gauss_spatial(this->feature_sz[0], sigma_scalar, sample_center[0], ksz_even[0]);
		std::vector<float> gauss_y = gauss_spatial(this->feature_sz[1], sigma_scalar, sample_center[1], ksz_even[1]);
		
		int h = gauss_x.size();
		int w = gauss_y.size();
		std::vector<float> target_labels_host(h * w);
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				target_labels_host[i * w + j] = gauss_x[i] * gauss_y[j];
			}
		}
		
		cudaMemcpy(this->target_labels_d, target_labels_host.data(), h * w * sizeof(float), cudaMemcpyHostToDevice);
	}

	std::tuple<nvinfer1::ICudaEngine*, nvinfer1::IExecutionContext*, nvinfer1::IRuntime*> load_engine(std::string engine_file_name)
	{
		std::ifstream file(engine_file_name, std::ios::binary | std::ios::ate);
		if (!file) {
			throw std::runtime_error("failed to load engine");
		}
		std::streamsize size = file.tellg();
		file.seekg(0, std::ios::beg);

		std::vector<char> buffer(size);
		if (!file.read(buffer.data(), size)) {
			throw std::runtime_error("unable to read engine");
		}

		Logger m_l = Logger();
		nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(m_l);
		nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size());
		nvinfer1::IExecutionContext* context = engine->createExecutionContext();

		return std::make_tuple(engine, context, runtime);
	}

	trte(std::string engine_file_name, std::string fe_engine_file_name, std::string hfe_engine_file_name)
	{
		std::ifstream file(engine_file_name, std::ios::binary | std::ios::ate);
		if (!file) throw std::runtime_error("failed to load engine");
		
		std::streamsize size = file.tellg();
		file.seekg(0, std::ios::beg);

		std::vector<char> buffer(size);
		if (!file.read(buffer.data(), size)) throw std::runtime_error("unable to read engine");

		Logger m_l = Logger();
		this->runtime = nvinfer1::createInferRuntime(m_l);
		this->engine = this->runtime->deserializeCudaEngine(buffer.data(), buffer.size());
		this->context = this->engine->createExecutionContext();
		
		cudaStreamCreate(&stream);

		border_mode = "inside_major";
		max_scale_change = 1.5;
		
		cudaMalloc((void**)&scores_raw_d, 18 * 18 * sizeof(float));
		cudaMalloc((void**)&bbox_preds_d, 4 * 18 * 18 * sizeof(float));
		cudaMalloc((void**)&train_feat_d, 256 * 18 * 18 * sizeof(float));
		cudaMalloc((void**)&target_labels_d, 18 * 18 * sizeof(float));
		cudaMalloc((void**)&train_ltrb_d, 4 * 18 * 18 * sizeof(float));
		cudaMalloc((void**)&training_samples_d, 1024 * 18 * 18 * sizeof(float));
		cudaMalloc((void**)&im_patch_d, 3 * image_sample_size * image_sample_size * sizeof(float));

		std::tie(this->fe_engine, this->fe_context, this->fe_runtime) = load_engine(fe_engine_file_name);
		std::tie(this->hfe_engine, this->hfe_context, this->hfe_runtime) = load_engine(hfe_engine_file_name);
	}

	std::valarray<float> get_centered_sample_pos()
	{
		return this->pos + this->precomputed_get_centered_sample_pos_val;
	}

	std::tuple<std::valarray<float>, float> get_sample_location(const std::valarray<float>& sample_coord)
	{
		std::valarray<float> sample_pos = 0.5f * (std::valarray<float>(sample_coord[std::slice(0, 2, 1)]) + std::valarray<float>(sample_coord[std::slice(2, 2, 1)]) - 1.0f);
		std::valarray<float> sample_sizes = std::valarray<float>(sample_coord[std::slice(2, 2, 1)]) - std::valarray<float>(sample_coord[std::slice(0, 2, 1)]);
		std::valarray<float> sc = sample_sizes / this->img_sample_sz;
		float sample_scales = std::sqrt(sc[0] * sc[1]);
		return std::make_tuple(sample_pos, sample_scales);
	}

	std::tuple<std::valarray<float>, int64_t, std::valarray<std::valarray<float>>, std::valarray<float>> localize_target_val(const std::valarray<float>& sample_pos, float sample_scales)
	{
		std::valarray<float> scores_val2(18 * 18);
		cudaMemcpy(&scores_val2[0], scores_raw_d, 18 * 18 * sizeof(float), cudaMemcpyDeviceToHost);

		std::valarray<std::valarray<float>> scores_val_val(18);
		for (int i = 0; i < 18; i++)
		{
			scores_val_val[i] = std::valarray<float>(&scores_val2[i * 18], 18);
		}

		std::valarray<float> score_sz = { 18.0f, 18.0f };
		std::valarray<float> score_center = (score_sz - 1.0f) / 2.0f;

		auto [max_score_val, max_disp_val_val] = max2dd_val(scores_val_val);

		std::valarray<float> target_disp = max_disp_val_val - score_center;

		std::valarray<float> ksz_even = { (float)((this->kernel_size[0] + 1) % 2), (float)((this->kernel_size[1] + 1) % 2) };
		std::valarray<float> output_sz_val_float = score_sz - ksz_even;
		
		std::valarray<float> img_sample_sz_float = { static_cast<float>(this->img_sample_sz[0]), static_cast<float>(this->img_sample_sz[1]) };
		std::valarray<float> translation_vec_val = target_disp * (img_sample_sz_float / output_sz_val_float * sample_scales);

		return std::make_tuple(translation_vec_val, (int64_t)0, scores_val_val, max_disp_val_val);
	}

	std::valarray<float> direct_bbox_regression(const std::valarray<float>& sample_coords, const std::valarray<float>& score_loc)
	{
		const int64_t img_width = static_cast<int64_t>(std::round(img_sample_sz[0]));
		const int64_t img_height = static_cast<int64_t>(std::round(img_sample_sz[1]));
		const int stride = 16;
		const int half_stride = stride / 2;

		const float x_scale_val = (sample_coords[3] - sample_coords[1]) / img_width;
		const float y_scale_val = (sample_coords[2] - sample_coords[0]) / img_height;
		const float x_offset_val = sample_coords[1];
		const float y_offset_val = sample_coords[0];

		int s1 = 18;
		int s2 = 18;

		std::valarray<float> xs_val(s1 * s2);
		std::valarray<float> ys_val(s1 * s2);
		for (int i = 0; i < s1; i++) {
			for (int j = 0; j < s2; j++) {
				ys_val[i * s2 + j] = i * stride + half_stride;
				xs_val[i * s2 + j] = j * stride + half_stride;
			}
		}

		std::valarray<float> multen_val({ train_img_sample_sz[0], train_img_sample_sz[1], train_img_sample_sz[0], train_img_sample_sz[1] });

		std::valarray<float> ltrb_cpu(4 * 18 * 18);
		cudaMemcpy(&ltrb_cpu[0], bbox_preds_d, 4 * 18 * 18 * sizeof(float), cudaMemcpyDeviceToHost);
		
		std::valarray<std::valarray<float>> ltrb_val(4);
		for (int i = 0; i < 4; i++) {
			ltrb_val[i] = std::valarray<float>(&ltrb_cpu[i * 18 * 18], 18 * 18);
			ltrb_val[i] *= multen_val[i];
		}

		std::valarray<float> xs1_val = xs_val - ltrb_val[0];
		std::valarray<float> ys1_val = ys_val - ltrb_val[1];
		std::valarray<float> xs2_val = xs_val + ltrb_val[2];
		std::valarray<float> ys2_val = ys_val + ltrb_val[3];

		int y_idx = static_cast<int>(std::round(score_loc[0]));
		int x_idx = static_cast<int>(std::round(score_loc[1]));

		float x1_out = xs1_val[y_idx * s2 + x_idx] * x_scale_val + x_offset_val;
		float y1_out = ys1_val[y_idx * s2 + x_idx] * y_scale_val + y_offset_val;
		float x2_out = xs2_val[y_idx * s2 + x_idx] * x_scale_val + x_offset_val;
		float y2_out = ys2_val[y_idx * s2 + x_idx] * y_scale_val + y_offset_val;

		return { x1_out, y1_out, x2_out - x1_out, y2_out - y1_out };
	}

	bool run_infer(const std::vector<float*>& tinps)
	{
		for (int i = 0; i < engine->getNbIOTensors(); i++) {
			context->setTensorAddress(engine->getIOTensorName(i), tinps[i]);
		}

		cudaStreamSynchronize(stream);
		bool status = context->enqueueV3(stream);
		cudaStreamSynchronize(stream);

		return status;
	}

	bool run_infer_on_engine(const std::vector<float*>& t_inps_outs, nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context)
	{
		for (int i = 0; i < engine->getNbIOTensors(); i++) {
			context->setTensorAddress(engine->getIOTensorName(i), t_inps_outs[i]);
		}
		cudaStreamSynchronize(stream);
		bool status = context->enqueueV3(stream);
		cudaStreamSynchronize(stream);

		return status;
	}

	std::tuple<std::vector<float>, float> track(cv::Mat image)
	{
		auto start = std::chrono::high_resolution_clock::now();
		
		std::valarray<float> inp_pos = this->get_centered_sample_pos();
		std::valarray<float> scaled_image_sz = this->img_sample_sz * this->target_scale;
		
		auto [im_patch_mat, sample_coords_val] = sample_patch_cv(image, inp_pos, scaled_image_sz, this->img_sample_sz, border_mode);

		std::vector<float> im_patch_chw = mat_to_chw(im_patch_mat);
		cudaMemcpy(im_patch_d, im_patch_chw.data(), im_patch_chw.size() * sizeof(float), cudaMemcpyHostToDevice);

		bool status = run_infer({ im_patch_d, train_feat_d, target_labels_d, train_ltrb_d, scores_raw_d, bbox_preds_d });

		if (!status) { throw std::runtime_error("FAIL"); }

		auto [sample_pos_val, sample_scales_val] = get_sample_location(sample_coords_val);
		auto [translation_vec_val, scale_ind_val, s_val, score_loc_val] = localize_target_val(sample_pos_val, sample_scales_val);

		std::valarray<float> bbox_raw_val = direct_bbox_regression(sample_coords_val, score_loc_val);
		std::valarray<float> bbox_val = clip_bbox_to_image_area_val(bbox_raw_val, image);

		std::valarray<float> pos_val1(bbox_val[std::slice(0, 2, 1)]);
		std::valarray<float> pos_val2(bbox_val[std::slice(2, 2, 1)]);

		std::reverse(std::begin(pos_val1), std::end(pos_val1));
		std::reverse(std::begin(pos_val2), std::end(pos_val2));

		this->pos = pos_val1 + pos_val2 / 2.0f;
		std::valarray<float> target_sz_val = pos_val2;

		float out2 = 0; 
		if (s_val.size() > 0) {
			out2 = std::get<0>(max2dd_val(s_val));
		}

		std::valarray<float> pos_yx_val({ this->pos[1], this->pos[0] });
		std::valarray<float> target_sz_yx_val({ target_sz_val[1], target_sz_val[0] });

		std::valarray<float> new_state_val1 = pos_yx_val - (target_sz_yx_val - 1.0f) / 2.0f;
		std::valarray<float> new_state_val2 = target_sz_yx_val;
		std::valarray<float> new_state_val(4);
		std::copy(std::begin(new_state_val1), std::end(new_state_val1), std::begin(new_state_val));
		std::copy(std::begin(new_state_val2), std::end(new_state_val2), std::begin(new_state_val) + 2);

		std::vector<float> output_state_val = { new_state_val[0], new_state_val[1], new_state_val[2], new_state_val[3] };
		auto ret = std::make_tuple(output_state_val, out2);

		return ret;
	}
};

std::map<std::string, std::string> read_config(const std::string& filename) {
	std::map<std::string, std::string> config;
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Warning: Could not open " << filename << ". Using defaults." << std::endl;
		return config;
	}
	std::string line;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#' || line[0] == ';') continue;
		auto delimiterPos = line.find("=");
		if (delimiterPos != std::string::npos) {
			auto key = line.substr(0, delimiterPos);
			auto value = line.substr(delimiterPos + 1);
			config[key] = value;
		}
	}
	return config;
}

int main()
{
	auto config = read_config("config.ini");
	
	std::string main_engine = config.count("main_engine") ? config["main_engine"] : "new_full_implicit_batch1_50_sanitized_calibrated.engine";
	std::string fe_engine = config.count("fe_engine") ? config["fe_engine"] : "feature_extractor_tompnet_50.engine";
	std::string hfe_engine = config.count("hfe_engine") ? config["hfe_engine"] : "head_feature_extractor_50.engine";
	std::string camera_input = config.count("camera_input") ? config["camera_input"] : "0";
	std::string output_mode = config.count("output_mode") ? config["output_mode"] : "display";
	std::string stream_url = config.count("stream_url") ? config["stream_url"] : "";

	trte trt_engine(main_engine, fe_engine, hfe_engine);

	bool bbox_set = false;

	cv::VideoCapture camera;
	if (camera_input.length() == 1 && std::isdigit(camera_input[0])) {
		camera.open(std::stoi(camera_input), cv::CAP_MSMF);
	} else {
		camera.open(camera_input);
	}

	if (!camera.isOpened()) {
		std::cerr << "ERROR: Could not open camera/stream: " << camera_input << std::endl;
		return 1;
	}

	camera.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
	camera.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
	camera.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
	camera.set(cv::CAP_PROP_EXPOSURE, -6);

	std::string CamName = "Webcam";
	if (output_mode == "display") {
		cv::namedWindow(CamName, cv::WINDOW_AUTOSIZE);
		cv::setMouseCallback(CamName, MouseCallback);
	}

	cv::VideoWriter writer;
	if (output_mode == "stream" && !stream_url.empty()) {
		int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
		writer.open(stream_url, cv::CAP_FFMPEG, fourcc, 30, cv::Size(1920, 1080), true);
		if (!writer.isOpened()) {
			std::cerr << "Warning: Could not open VideoWriter for stream." << std::endl;
		}
	}

	cv::Mat frame;
	int font = cv::FONT_HERSHEY_COMPLEX_SMALL;

	auto stop = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::high_resolution_clock::now();
	std::string time_str = "20";

	while (1) {
		start = std::chrono::high_resolution_clock::now();

		double time = 1 / std::chrono::duration<double>(start - stop).count();
		stop = start;
		time_str = std::to_string(time);

		camera.read(frame);
		if (frame.empty()) break;

		if (!bbox_set && new_init) {
			bbox_set = true;
			trt_engine.initialize(frame, get_bb());
		}
		if (bbox_set) {
			std::vector<float> track_out;
			float score;
			std::tie(track_out, score) = trt_engine.track(frame);
			if(score > 0.3)
				cv::rectangle(frame,
					cv::Point(track_out[0], track_out[1]),
					cv::Point(track_out[0] + track_out[2], track_out[1] + track_out[3]), cv::Scalar(0, 255, 0));
		}

		time_str = std::to_string(camera.get(cv::CAP_PROP_FPS));
		if (mode == "select" && output_mode == "display") {
			cv::rectangle(frame, getCoordRect(), cv::Scalar(0, 255, 0));
		}
		cv::putText(frame, time_str, cv::Point(20, 100), font, 3, cv::Scalar(100, 255, 0), 3);
		
		if (output_mode == "display") {
			cv::imshow("Webcam", frame);
			if (cv::waitKey(1) == 27) break;
		} else if (output_mode == "stream" && writer.isOpened()) {
			writer.write(frame);
		}
	}
	return 0;
}