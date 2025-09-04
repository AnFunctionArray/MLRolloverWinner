#include "pch.h"
#include <locale>
#include <fstream>
#include <thread>
#include <atomic>
#include <filesystem>
#include <mutex>
#include <condition_variable>
//#include <barrier>
#include "lbfgs1.h"
#include <iostream>
#include <sstream>
#include <ppl.h>
#include "nadam.h"
#include <iomanip>
#include <functional>
#include <tlhelp32.h>
#include <random>
#include <barrier>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGraph.h>
#include <torch/nn/parallel/data_parallel.h>
#include <torch/data/datasets/mnist.h>
#include <torch/optim/schedulers/lr_scheduler.h>
#include <torch/optim/schedulers/reduce_on_plateau_scheduler.h>
const float clearColor[] = { 0.0f, 0.7f, 0.7f, 1.0f };
#define N_MODES 1
//#define float double
#include "lbfgs1.h"
//#define weight_decay lr
#define device torch::Device(torch::kCPU, -1)
#define devicecpu torch::Device(torch::kCPU, -1)
#define cuda cpu
#define LCM_CUDA LCM_NO_ACCELERATION
static torch::Tensor prabs = torch::empty({ 3 });
static torch::Tensor todrawres;
typedef torch::optim::LBFGSOptions tgt_optim_opts_t;
typedef torch::optim::LBFGS tgt_optim_t;
static float lssdifovg = FLT_MAX;
static int itesrwin = 0;
static float acccoef = 0.;
static int modes = 0;
static float lossdelta = 0.;
static bool predrightg;
static bool lcctotrain;
static int defb = 1;
static float vbal2plstvbal2av = INFINITY;
static float losslstg = 0.;
static float losslstgr = FLT_MAX;
static float losslstgor = FLT_MAX;
static int betindexa = 0;
static int betindexb = 0;
static float losslstgorig = FLT_MAX;
static float losslstgorig1 = FLT_MAX;
static float losslstgt = FLT_MAX;
static size_t itesrempty = 0;
static float losslstgd = 0.;
static torch::Tensor resallcur;
static float biasseses[20]{ INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, };
static bool predaboves[20];
static torch::Tensor vbal2s = torch::zeros({ 3, 20 }, torch::dtype(torch::ScalarType::Int)).cuda();
static float betamntflss = 0.;
static bool predsperrow[20];
static float resallmlst;
static int sincelastwin;
static bool hitted = false;
static float avitesrperwin = INFINITY;
static int toinfermiisdiagcenter[2];
static uint32_t timesresetted = 0;
static float dirmean;
static float runlr = 0.00001;
static float runlradv = 100.;
static float runlr2 = 0.00001;
static float runlr3 = 0.00001;
static float runlr2b = 1e-4;
#define RUNLRBD 0.166666666
static float runlrb = RUNLRBD;
static float resallm;
static float lssdiv;
static float lossorigin;
static float lossoriignm = FLT_MAX;
static bool lossorignmb;
static bool swtolrn = false;
static bool swto50 = false;
static bool dirwlaloss2ab = false;
static bool dirwlaloss2abf = false;
static double gain = 0.0;
static bool firstgain = false;
static bool defbetabove = false;
static bool dobetr = 0;
static float rollostismmin = FLT_MAX;
static float rollostismmax = FLT_MIN;
static torch::Tensor allw = torch::empty({ 0 }).cuda();
static torch::Tensor toinfetotrain = torch::tensor({ 0 }).cuda();
static torch::Tensor toinfetotrainto = torch::tensor({ 0 }).cuda();
static torch::Tensor ressesvbals = torch::zeros({ 1 });
static torch::Tensor resallmenall = torch::zeros({ 20, 20 }).cuda();
static torch::Tensor totraintol = torch::zeros({ N_MODES, 20, 20 }).cuda();
static torch::Tensor totraintoll = torch::zeros({ N_MODES, 20, 20 }).cuda();
static torch::Tensor resall3o = torch::zeros({ 20, 20 });
static bool resallmeanfirst = true;
static bool vbalactualonly = false;
static uint64_t winsov;
static uint64_t lossov;
static bool dinferlrnsw = false;
static std::barrier inrec{ 2 };
static float avmaxiter = 0.;
static bool avmaxiterfirst = true;
static bool bal_gen = false;
static bool iterslostfirst = true;
static int iterslost = 0, * piterslostlst = &iterslost;
static float iterslostav;
static torch::Tensor ten = torch::zeros({ 20 }).cuda();
static torch::Tensor ten2 = torch::zeros({ 20 }).cuda();
static int dirwlac = 0;
static bool checkedall = false;
static float decayfe = 0.01;
static float avm;
static float avm2;
static int betindex = 0;
static bool betindexdir = 0;
static torch::Tensor fwdhlbl = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda().requires_grad_(false), fwdhlb2 = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda(), resall2o, resallo, reswillwino, reswillwino1, reswillwinotr,
fwdhlblout = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda(), fwdhlbl2 = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda(), fwdhlbl2o = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda(),
fwdhlbl2w = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda(), fwdhlbl2l = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda(), fwdhlbloutst = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
static torch::Tensor reslst = torch::zeros({ 20,20 }).cuda();
static bool dirw = false;
static bool contiw = false;
static int itesr = 0;
static int itesrlst = 0;
static int itesr2 = 0;
static int itesrg = 0;
static bool optswitch = false;
static bool meanswitch = false;
static int targetiters = 1;
static bool halttrain = true;
static std::barrier contrain{ 2 };
static bool shrink = false;
static bool above10 = false;
static bool above10reqend = true;
static bool above10reqstart = true;
static bool expand = true;
static int dirsh = 1;
static int dirsh2 = 1;
static bool netted = false;
static float glcoef = 0.;
static bool predicts2[20];
static bool predictsg[20];
static float avmeans[20];
static float avmeans2[20];
static float resallmena;
static bool avmed = false;
static bool avmed2 = false;
std::streambuf* orig_buf;
torch::Tensor supervised_loss(torch::Tensor model_output, torch::Tensor target) {
	return torch::binary_cross_entropy_with_logits(
		model_output,  // Raw logits (20x20)
		target        // Ground truth "almost sure" matrix (20x20)
	);
}
torch::Tensor compute_reward(torch::Tensor pred_logits, torch::Tensor validation_matrix) {
	torch::Tensor preds = torch::sigmoid(pred_logits) > 0.5;  // Binarize
	torch::Tensor intersection = (preds * validation_matrix).sum();
	torch::Tensor union_ = (preds + validation_matrix).sum() - intersection;
	return intersection / (union_ + 1e-6);  // IoU reward
}
torch::Tensor hybrid_loss(
	torch::Tensor model_output,      // Raw logits (20x20)
	torch::Tensor sl_target,         // "Almost sure" matrix (20x20)
	torch::Tensor validation_matrix  // Actual validation matrix (20x20)
) {


	// Reinforcement reward (RL)
	//torch::Tensor reward = compute_reward(model_output, validation_matrix);

	// Policy gradient: Maximize reward * log_prob of chosen actions
	//torch::Tensor probs = torch::sigmoid(model_output);
	//torch::Tensor log_probs = torch::log(probs + 1e-6);  // Avoid log(0)
	torch::Tensor rl_weights = sl_target.matmul(validation_matrix);//reward / (reward.max() + 1e-6);//-log_probs * reward;  // Minimize negative reward
	//sl_target = sl_target * rl_loss;
	// Supervised loss (SL)
	rl_weights = rl_weights / (rl_weights.max() + 1e-6);
	torch::Tensor sl_loss = torch::binary_cross_entropy_with_logits(
		model_output,  // Raw logits (20x20)
		sl_target//,        // Ground truth "almost sure" matrix (20x20)
		//{},
		//rl_weights.squeeze(0)
	);
	//std::stringstream ss;
	//ss << "rl_weights " << rl_weights << std::endl;
	//ss << "triggered at: " << res[2] << std::endl;
	//ss << "pred is: " << (res[0] > res[1]) << std::endl;

	//orig_buf->sputn(ss.str().c_str(), ss.str().size());
	// Combine losses
	float lambda = 0.5;  // Weight for SL
	return sl_loss;
}
#if 0
inline double clip_norm_(
	const std::vector<torch::Tensor>& parameters,
	double max_norm,
	double norm_type = 2.0,
	bool error_if_nonfinite = false) {
	using namespace torch;
	std::vector<Tensor> params_with_grad;

	for (const auto& param : parameters) {
		auto& grad = param.grad();
		if (grad.defined(), true) {
			params_with_grad.push_back(param);
		}
	}

	if (params_with_grad.empty()) {
		return 0.0;
	}

	Tensor total_norm_tensor;
	if (norm_type == std::numeric_limits<double>::infinity()) {
		std::vector<Tensor> norms;
		norms.reserve(params_with_grad.size());

		for (const auto& param : params_with_grad) {
			norms.emplace_back(param.data().abs().max());
		}
		total_norm_tensor =
			(norms.size() == 1) ? norms[0] : torch::max(torch::stack(norms));
	}
	else if (norm_type == 0) {
		total_norm_tensor =
			torch::full({}, static_cast<double>(params_with_grad.size()));
	}
	else {
		std::vector<Tensor> norms;
		norms.reserve(params_with_grad.size());

		for (const auto& param : params_with_grad) {
			norms.emplace_back(param.data().norm(norm_type));
		}
		total_norm_tensor =
			(norms.size() == 1) ? norms[0] : torch::stack(norms).norm(norm_type);
	}

	// When possible (ie when skipping the finiteness check), we avoid
	// synchronizing the CPU and the gradients' device until the very end to
	// preserve async execution on the device. When checking for finite-ness, this
	// optional ensures we only sync once.
	std::optional<double> total_norm = std::nullopt;
	if (error_if_nonfinite) {
		total_norm = total_norm_tensor.item().toDouble();
		TORCH_CHECK(
			std::isfinite(*total_norm),
			"The total norm of order ",
			norm_type,
			" for gradients from `parameters` ",
			"is non-finite, so it cannot be clipped. To disable this error and scale ",
			"the gradients with the non-finite norm anyway, set ",
			"`error_if_nonfinite=false`");
	}

	auto clip_coef = max_norm * (total_norm_tensor + 1e-6);
	auto clip_coef_clamped =
		torch::clamp(clip_coef, std::nullopt /* min */, 1.0 /* max */);
	for (auto& param : params_with_grad) {
		param.data().mul_(clip_coef_clamped);
	}

	if (!total_norm.has_value()) {
		total_norm = total_norm_tensor.item().toDouble();
	}
	return *total_norm;
}
#endif

torch::Tensor prune_weights(torch::Tensor weights, float prune_ratio) {
	torch::Tensor abs_weights = weights.abs();
	torch::Tensor sorted_values, sorted_indices;

	// Sort absolute weights
	std::tie(sorted_values, sorted_indices) = abs_weights.view(-1).sort();

	// Find threshold for pruning
	int num_to_prune = static_cast<int>(sorted_values.size(0) * prune_ratio);
	float threshold = sorted_values[num_to_prune].item<float>();

	// Apply pruning mask
	torch::Tensor mask = abs_weights >= threshold;
	return weights.mul_(mask); // Zero out pruned weights
}

//#define tensor(...) tensor(__VA_ARGS__, device)

//#define Float Half
#define Half Float
//#define Double Half
//#define toFloat toDouble
//#define Float Float
#define BFLoat16 Float
#define kCUDA kCUDA
//#define cuda() to(device)
static int rolllosti = 0;
static bool subadd[20];
static torch::Tensor rolllostiscr = torch::zeros({ 20 }, torch::dtype(torch::ScalarType::Float)).cuda();//[20] = { 0 };
static torch::Tensor rolllostis = torch::zeros({ 3, 20 }, torch::dtype(torch::ScalarType::Float)).cuda();//[20] = { 0 };
static torch::Tensor rolllostisinf = torch::zeros({ 20 }, torch::dtype(torch::ScalarType::Float)).cuda();//[20] = { 0 };
static torch::Tensor rolllostisw = torch::zeros({ 20 }, torch::dtype(torch::ScalarType::Int)).cuda();//[20] = { 0 };
static torch::Tensor rolllostisl = torch::zeros({ 20 }, torch::dtype(torch::ScalarType::Int)).cuda();//[20] = { 0 };
static torch::Tensor rolllostisov = torch::zeros({ 20 }, torch::dtype(torch::ScalarType::Float)).cuda();
static torch::Tensor rolllostisovinf = torch::zeros({ 20 }, torch::dtype(torch::ScalarType::Float)).cuda();
static unsigned long long rndvalb;
static char preds[20];
static bool lststate[20];
static float loss1gf = FLT_MAX;
static float loss2gf = FLT_MAX;
static bool currloss = false;
static float lossg = FLT_MAX;
static float stepmaxg = FLT_MIN;
static float lstlossg = FLT_MAX;
static float lossav = FLT_MAX;
static float lstlossc = FLT_MAX;
static std::atomic_int ovrolls = 0;
static torch::Tensor tolrnmsk = torch::zeros({ 20, 20 }).cuda();
static torch::Tensor toinfermsk = torch::zeros({ 20, 20 }).cuda();
static torch::Tensor totrain = torch::tensor({ 0 }).cuda();
static torch::Tensor toinfer = torch::tensor({ 0 }).cuda();
static torch::Tensor toinferm = torch::tensor({ 0 }).cuda();
torch::Tensor totrainl = torch::ones({ 1, 20, 20 }).cuda();
torch::Tensor totrainllst = torch::zeros({ 1, 1, 20 }).cuda();
torch::Tensor totraincur = torch::empty({}).cuda();
torch::Tensor tolrnll2 = torch::zeros({ 1, 20, 20 }).cuda();
torch::Tensor abvsgrids = torch::ones({ 1, 20, 20 }).cuda();
torch::Tensor abvsgridsvals = torch::zeros({ 1, 20, 20 }).to(dtype(torch::ScalarType::Int)).cuda();
torch::Tensor abvsgridslst = torch::zeros({ 1, 20, 20 }).cuda();
torch::Tensor totrainll = torch::ones({ 1, 20, 20 }).cuda();
torch::Tensor totrainlw = torch::ones({ 1, 20, 20 }).cuda();
torch::Tensor totrainlm;//torch::tensor({}).cuda();
torch::Tensor totrainlm2 = torch::ones({ 1, 1, 1 }).cuda();
//torch::Tensor totrainll = torch::zeros({ 1, 20, 20 }).cuda();
torch::Tensor totrainlr = torch::zeros({ 1, 20, 20 }).cuda();
static torch::Tensor toinferm2 = torch::tensor({ 0 }).cuda();
static torch::Tensor toinferm2lst = torch::tensor({ 0 }).cuda();
static torch::Tensor toinferm3 = torch::tensor({ 0 }).cuda();
static torch::Tensor toinferm3mask = torch::tensor({ 0 }).cuda();
static torch::Tensor toinferm4 = torch::tensor({ 0 }).cuda();
static torch::Tensor toinferm3r = torch::tensor({ 0 }).cuda();
static int toinfermii[20][2];
static torch::Tensor toinfermi = torch::tensor({ 0 }).cuda();
static torch::Tensor toinfermi2 = torch::tensor({ 0 }).cuda();
static torch::Tensor toinfermio = torch::tensor({ 0 }).cuda();
static torch::Tensor toinfermio2 = torch::tensor({ 0 }).cuda();
static torch::Tensor toinferin = torch::tensor({ 0 }).cuda();
static torch::Tensor toinferlst = torch::tensor({ 0 }).cuda();
static torch::Tensor toinferwon = torch::tensor({ 0 }).cuda();
static torch::Tensor toinferlost = torch::tensor({ 0 }).cuda();
static torch::Tensor toinfersum = torch::zeros({ 20, 20 }).cuda();
static torch::Tensor toinfero = torch::ones({ 20, 20 }).cuda();;
static torch::Tensor tolrn = torch::tensor({ 0 }).cuda();
static torch::Tensor tolrnl = torch::tensor({ 0 }).cuda();
static torch::Tensor tolrnl2 = torch::zeros({ 0 }).cuda();
static torch::Tensor tolrnl2m = torch::zeros({ 0 }).cuda();
static torch::Tensor tolrnl52m;
static torch::Tensor tolrnl3 = torch::tensor({ 0 }).cuda();
static torch::Tensor tolrnl4 = torch::zeros({ 20, 1 }).cuda();
static torch::Tensor tolrnl52 = torch::zeros({ 1 }).cuda();
static torch::Tensor tolrnl52r = torch::zeros({ 2 }).cuda();
static torch::Tensor tolrn0 = torch::ones({ 20, 20 }).cuda();
static torch::Tensor tolrn1 = torch::ones({ 20, 20 }).cuda();
static torch::Tensor tolrn3 = torch::ones({ 20, 20 }).cuda();
static torch::Tensor tolrnrs = torch::ones({ 20, 20 }).cuda();
static torch::Tensor tolrn3sz = torch::ones({ 20, 20 }).cuda();
static torch::Tensor totrainto = torch::tensor({ 0 }).cuda();
static torch::Tensor totrainto2 = torch::tensor({ 0 }).cuda();
static torch::Tensor lstrnnresh = torch::zeros({ 2, 2 * 6, 20, 20 }).cuda();
static std::atomic_bool chngsmpls = false;
static std::atomic_bool rndtrain = false;
static std::atomic_bool bad_exc = false;
static int ovwins;
#define relu relu
//#define tanh relu
static bool abovereswl = false;
static bool donetrained = 0;
static bool donetrainedbet = 0;
static bool ressswon = false;
static bool doacc = 1;
static bool vbalactualacc = false;
static bool diirseek = false;
static std::atomic_int statlessiters = 0;
static enum TRAINMODE {
	general,
	haswon,
	unknown
} trainmode;

static bool traingb = false;





//static torch::Device device(torch::kCUDA, 0);

//#define REAL_BAL

static bool REAL_BAL = true;

static uint16_t triangleIndices[50] = { 0, 1, 2, 3 };
static std::atomic_int64_t resrttsedg = 1;
static std::atomic_int64_t resrttsedgmore = 0;
static std::mutex incnonce;
static std::atomic_uint32_t betsmade;
static int64_t betsitesrmade;
static int64_t betsitesrmade400g;

static std::atomic_int32_t nbetthreads = 0;
static std::atomic_int32_t ntrainsfinished = 0, nontrains = 0;

static std::atomic_bool pseudoswitch = false,
doneiter = false, haslost = false, switchswitchb = false, switchswitch = false, pseudoswitchb = false, modeswitch = false, bethigh = false, modeswitch2 = false, trainstate = false, dirlr = false,
hasstarted = false, modeswitchswitch = false, mdsw = false, haslostg = false;
static std::atomic_int iterscltrain = 0;

static std::mt19937 mstsrvrng;
static unsigned int in32srvmseed;
static std::uniform_int_distribution<std::mt19937::result_type> mstsrvdist(0, UINT_MAX);

static std::mt19937 mstclrng;
static unsigned int in32clmseed;
static std::uniform_int_distribution<std::mt19937::result_type> mstcldist(0, UINT_MAX);

static std::mt19937 mstrng;
static unsigned int in32mseed;
static std::uniform_int_distribution<std::mt19937::result_type> mstdist(0, UINT_MAX);

extern "C" unsigned long long getrngterry();

template<class T>
int terryrnd64(T* u64, int bits = 64, bool inf = false) {
	volatile unsigned long long u = 0, u1 = 0, res = 0;
	for (int y = 0; !inf ? (u1 - u) < UINT8_MAX : true; res ^= (u1 - u) << (y % bits), ++y) {
		do {
			u = getrngterry();
			u1 = getrngterry();
		} while (u >= u1);
		//std::stringstream ss;
		//ss << " " << std::hex << res << std::endl;
	//ss << "triggered at: " << res[2] << std::endl;
	//ss << "pred is: " << (res[0] > res[1]) << std::endl;

		//orig_buf->sputn(ss.str().c_str(), ss.str().size());
		if (inf)
			*u64 = res;
	}
	*u64 = res;
	//std::cout << std::endl << *u64 << std::endl;
	//std::stringstream ss;
	//ss << " " << std::hex << *u64 << std::endl;
	//ss << "triggered at: " << res[2] << std::endl;
	//ss << "pred is: " << (res[0] > res[1]) << std::endl;

	//orig_buf->sputn(ss.str().c_str(), ss.str().size());
	return 1;
}

#define _rdseed64_step terryrnd64

#define getrng(var) while (!(_rdseed64_step(&var)))
#define getrngt(var)  while (!( terryrnd64(&var)))
#define getrngps(var) var = (((unsigned long long)mstsrvdist(mstsrvrng) << 32) | (unsigned long long)mstsrvdist(mstsrvrng))//while (!terryrnd64(&var))
#define getrngpc(var) var = (((unsigned long long)mstcldist(mstclrng) << 32) | (unsigned long long)mstcldist(mstclrng))//while (!terryrnd64(&var))
#define getrngp(var) var = (((unsigned long long)mstdist(mstrng) << 32) | (unsigned long long)mstdist(mstrng))//while (!terryrnd64(&var))

#define getrngtp(var, cond)  if (cond) getrngt(var); else getrngp(var)

//9449bf57b4e13b6cc555e4ca4ea908336f16acc3290c3e791670337bece49cf1
//FFFFFFFFFFFFFFFF

// x4 64bit
static std::thread lstthread = std::thread{ []() {} };
static std::atomic_int iterstrain = 1;

static std::function<void(bool, const class torch::nn::Module&, const char*)> savestuff;
static long long traintime = 0;
static std::atomic_int resitl2 = 0;
static std::atomic_int resitl = 0;
static std::atomic_int iterb = 0;
static std::atomic_bool dobets = true;
static std::atomic_bool switchchecklss = false;
static std::condition_variable condvartrain;
static std::condition_variable condvartrainreset;
static std::atomic_bool condt4 = true, condtb = false, traininginp = false, endtrain = false, switchtrain = false, condt5 = false;
static std::chrono::nanoseconds startt, endt;
static std::atomic_int ntrains = 1;
static std::atomic_int resir = 0;
static int trainedbend = -1;
static int trainedbstart = -1;
static std::atomic_int trainiter = 0, classesn = 0;
static torch::Tensor inpb = torch::ones({ 20 }).cuda();;
static DWORD mainthreadid;
static DWORD learnthreadid;
static std::atomic_bool condtll3 = false, condtll4, divb = true;
static int globaliters = 0;
std::atomic_bool condtll = false, trainedb = 0, trainedb2 = true, trainedb3 = true, trainedb4 = true, trainedb5 = true, hastrainedb = false, trainedba;
static std::atomic_bool switchdir = false;
static bool canchangeswitchdir = false;
static bool wantchangeswitchdir = false;
static std::atomic_bool actualdir = false;
static std::atomic_bool actualdir2 = 1;
static std::atomic_bool lstactualdir3;
static std::atomic_bool actualdir3 = 0;
static std::atomic_bool actualdir4 = 1;
static bool actualdirs5[20];
static int64_t vbals[20];
static int64_t vbalssum[20];
static bool vbalssumdir[20];
static int64_t vbalsmin[20];
static int64_t vbalsmax[20];
static std::atomic_bool actualdird = 1;
static std::atomic_bool actualdirset = false;
static std::atomic_bool switchdirroll = false;
static bool switchdirs[N_MODES][20] = { 0 };
static bool lstpreds[N_MODES][20] = { 0 };
static bool lstpredsright[20] = { 0 };
static int lstpredsrightc[20] = { 0 };
static bool switchdirrolls2[20] = { 0 };
static bool switchdirs2[20] = { 0 };
static bool switchdirrolls[20] = { 0 };
static std::atomic_int nthreads = 0;
static std::mutex optimlk;
static std::mutex mdllk, trainreslk, mdllkb, mdllkt, mtend;
static std::atomic_bool alttrain = false;
//static std::barrier syncb{ 2 };
static float minlss = 1.;
static std::atomic_bool against;
static std::atomic_int againstg = -1;
static std::atomic_int againstgav = -1;
static std::atomic_int againstgu = 0;
static std::atomic_int againstgtotal = 0;
static std::atomic_int againstgnext = -1;
static std::atomic_bool against1;
static std::atomic_bool against2;
static std::atomic_bool fliptrain;
static torch::Tensor lstrnntrainh;
static double lrf = 0.0000000000002;
static POINT p1, p2, p3;
__declspec(align(32))
struct Vertex
{
	float vals[8];
};
static Vertex triangleVertices[50] =
{
	/*{{0.0f, 0.25f , 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}},
	{ { 0.25f, -0.25f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f } },
	{ { -0.25f, -0.25f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f } },
	{ { -0.25f, 0.f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f } },*/
};

#define SYNC_TRAIN_BET

// Pass 0 as the targetProcessId to suspend threads in the current process
void DoSuspendThread()
{
	HANDLE h = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (h != INVALID_HANDLE_VALUE)
	{
		THREADENTRY32 te;
		te.dwSize = sizeof(te);
		if (Thread32First(h, &te))
		{
			do
			{
				if (te.dwSize >= FIELD_OFFSET(THREADENTRY32, th32OwnerProcessID) + sizeof(te.th32OwnerProcessID))
				{
					// Suspend all threads EXCEPT the one we want to keep running
					if (te.th32ThreadID != mainthreadid && te.th32OwnerProcessID == 0 && te.th32ThreadID != GetCurrentThreadId() && te.th32ThreadID != learnthreadid)
					{
						HANDLE thread = ::OpenThread(THREAD_ALL_ACCESS, FALSE, te.th32ThreadID);
						if (thread != NULL)
						{
							SuspendThread(thread);
							CloseHandle(thread);
						}
					}
				}
				te.dwSize = sizeof(te);
			} while (Thread32Next(h, &te));
		}
		CloseHandle(h);
	}
}



struct NetImpl : torch::nn::Cloneable<NetImpl> {
	void reset() override {

	}
	NetImpl() {
		//reset();
	}
	NetImpl(std::nullptr_t) {
		//reset();
	}
	void printw(int i = 0) {
		NetImpl& mdl = *this;
		auto updwp = [=](torch::Tensor& w, const torch::Tensor& w1) {
			std::stringstream ss;
			//ss << " " << std::hex << res << std::endl;
		//ss << "triggered at: " << res[2] << std::endl;
		//ss << "pred is: " << (res[0] > res[1]) << std::endl;
			ss << w << std::endl;
			orig_buf->sputn(ss.str().c_str(), ss.str().size());
			};
		auto updw = [=](torch::Tensor& w, const torch::Tensor& w1) {
			std::stringstream ss;
			//ss << " " << std::hex << res << std::endl;
		//ss << "triggered at: " << res[2] << std::endl;
		//ss << "pred is: " << (res[0] > res[1]) << std::endl;
			ss << w << std::endl;
			orig_buf->sputn(ss.str().c_str(), ss.str().size());
			};
		updw(layers[i].cnvtr1->weight, mdl.layers[i].cnvtr1->weight);
		updw(layers[i].lin1->weight, mdl.layers[i].lin1->weight);
		updw(layers[i].lin2->weight, mdl.layers[i].lin2->weight);
		updw(layers[i].cnv1->weight, mdl.layers[i].cnv1->weight);
		updw(layers[i].cnv2->weight, mdl.layers[i].cnv2->weight);
		updw(layers[i].cnv3->weight, mdl.layers[i].cnv3->weight);
		updw(layers[i].rnnresh, mdl.layers[i].rnnresh);
		if (layers[i].mha->in_proj_weight.defined())
			updw(layers[i].mha->in_proj_weight, mdl.layers[i].mha->in_proj_weight);
		if (layers[i].mha->k_proj_weight.defined())
			updw(layers[i].mha->k_proj_weight, mdl.layers[i].mha->k_proj_weight);
		if (layers[i].mha->q_proj_weight.defined())
			updw(layers[i].mha->q_proj_weight, mdl.layers[i].mha->q_proj_weight);
		if (layers[i].mha->v_proj_weight.defined())
			updw(layers[i].mha->v_proj_weight, mdl.layers[i].mha->v_proj_weight);
		for (size_t pi = 0; pi < layers[i].trans->decoder.ptr()->parameters().size(); ++pi)
			updwp(layers[i].trans->decoder.ptr()->parameters()[pi], mdl.layers[i].trans->decoder.ptr()->parameters()[pi].data());//torch::nn::init::xavier_uniform_(p).cuda();
		for (size_t pi = 0; pi < layers[i].trans->encoder.ptr()->parameters().size(); ++pi)
			updwp(layers[i].trans->encoder.ptr()->parameters()[pi], mdl.layers[i].trans->encoder.ptr()->parameters()[pi].data());
		for (size_t pi = 0; pi < layers[i].rnn1->all_weights().size(); ++pi)
			updwp(layers[i].rnn1->all_weights()[pi], mdl.layers[i].rnn1->all_weights()[pi].cuda().data());
	}
	torch::Tensor gather_flat_grad() {
		std::vector<torch::Tensor> views;
		views.reserve(parameters().size());
		for (const auto& p : parameters()) {
			if (!p.grad().defined()) {
				views.emplace_back(p.new_empty({ p.numel() }).zero_());
			}
			else if (p.grad().is_sparse()) {
				views.emplace_back(p.grad().to_dense().view(-1));
			}
			else {
				views.emplace_back(p.grad().view(-1));
			}
		}
		return torch::cat(views, 0);
	}

	torch::Tensor gather_flat_grad_uniform() {
		std::vector<torch::Tensor> views;
		views.reserve(parameters().size());
		for (const auto& p : parameters()) {
			//auto newt = torch::zeros(p.data().sizes()).cuda();
			//if (newt.ndimension() > 1)
			//	torch::nn::init::xavier_uniform_(newt);
			views.emplace_back(p.data().view(-1));
		}
		return torch::cat(views, 0);
	}

	void add_grad(const double step_size, const torch::Tensor& update) {
		auto offset = 0;
		for (auto& p : parameters()) {
			auto numel = p.numel();
			// view as to avoid deprecated pointwise semantics
			//p.mul_(0.9).add_(p, 1.0 - 0.9);
			p.data().add_(
				update.index({ at::indexing::Slice(offset, offset + numel) }).view_as(p),
				step_size);
			offset += numel;
		}
	}

	void set_grad(const double step_size, const torch::Tensor& update) {
		auto offset = 0;
		for (auto& p : parameters()) {
			auto numel = p.numel();
			// view as to avoid deprecated pointwise semantics
			//p.mul_(0.9).add_(p, 1.0 - 0.9);
			//p.data().copy_(
			p.data().mutable_grad() = p.grad().defined() ? p.grad() +
				update.index({ at::indexing::Slice(offset, offset + numel) }).view_as(p)
				:
				update.index({ at::indexing::Slice(offset, offset + numel) }).view_as(p);
			offset += numel;
		}
	}

	int nels = 0;

	int calc_nels() {
		for (auto& p : parameters()) {
			auto numel = p.numel();

			nels += numel;
		}
		return nels;
	}

	//static constexpr double decayema = 0.999;
	void ema_update(int i, const NetImpl& mdl, double decay = 0.999, std::function<void(torch::Tensor& w, const torch::Tensor& w1, double decay)> updwpa = [](torch::Tensor& w, const torch::Tensor& w1, double decay) {
		w.set_data(w.data().detach() * decay + w1.detach() * (1. - decay));
		}, std::function<void(torch::Tensor& w, const torch::Tensor& w1, double decay)> updwa = [](torch::Tensor& w, const torch::Tensor& w1, double decay) {
			w = w.detach() * decay;
			w += w1.detach() * (1. - decay);
			}) {
		//return;
		auto updw = [decay, updwa](torch::Tensor& w, const torch::Tensor& w1) {updwa(w, w1, decay); };
		auto updwp = [decay, updwpa](torch::Tensor& w, const torch::Tensor& w1) {updwpa(w, w1, decay); };
		//for (int i = 0; i < 3; ++i) {
		for (size_t i = 0; i < parameters().size(); ++i) {
			//concurrency::parallel_for(size_t(0), parameters().size(), [&](size_t i) {
				//auto p = named_parameters()[i].value();
				//auto numel = p.numel();
				// view as to avoid deprecated pointwise semantics
				//p.mul_(0.9).add_(p, 1.0 - 0.9);
			updw(parameters()[i], mdl.parameters()[i]);
		}//);
		//layers[0].rnn1->flatten_parameters();
		return;
		//updw(cnvf->weight, mdl.cnvf->weight);
		for (int i = 0; i < 1; ++i) {
			//concurrency::parallel_for(int(0), 2, [&](int i) {
				//updw(layers[i].cnvtr1->weight, mdl.layers[i].cnvtr1->weight);
				//updw(layers[i].cnvtr2->weight, mdl.layers[i].cnvtr2->weight);
				//updw(layers[i].cnvtr3->weight, mdl.layers[i].cnvtr3->weight);
				//updw(layers[i].lin1->weight, mdl.layers[i].lin1->weight);
				//updw(layers[i].lin2->weight, mdl.layers[i].lin2->weight);
				//updw(layers[i].lin3->weight, mdl.layers[i].lin3->weight);
				//updw(layers[i].lin4->weight, mdl.layers[i].lin4->weight);
			updw(layers[i].cnv1->weight, mdl.layers[i].cnv1->weight);
			updw(layers[i].cnv2->weight, mdl.layers[i].cnv2->weight);
			updw(layers[i].cnv3->weight, mdl.layers[i].cnv3->weight);

			updw(layers[i].cnv11->weight, mdl.layers[i].cnv11->weight);
			updw(layers[i].cnv21->weight, mdl.layers[i].cnv21->weight);
			updw(layers[i].cnv31->weight, mdl.layers[i].cnv31->weight);

			updw(layers[i].cnv12->weight, mdl.layers[i].cnv12->weight);
			updw(layers[i].cnv22->weight, mdl.layers[i].cnv22->weight);
			updw(layers[i].cnv32->weight, mdl.layers[i].cnv32->weight);
			//updw(layers[i].cnv4->weight, mdl.layers[i].cnv4->weight);
			//updw(layers[i].rnnresh, mdl.layers[i].rnnresh);
			//updw(layers[i].norm11->weight, mdl.layers[i].norm11->weight);
			//updw(layers[i].norm21->weight, mdl.layers[i].norm21->weight);
			//updw(layers[i].norm31->weight, mdl.layers[i].norm31->weight);

			//updw(layers[i].norm1->weight, mdl.layers[i].norm1->weight);
			//updw(layers[i].norm2->weight, mdl.layers[i].norm2->weight);
			//updw(layers[i].norm3->weight, mdl.layers[i].norm2->weight);

			//updw(layers[i].cnv4->weight, mdl.layers[i].cnv4->weight);
			/*if (layers[i].mha->in_proj_weight.defined())
				updw(layers[i].mha->in_proj_weight, mdl.layers[i].mha->in_proj_weight);
			if (layers[i].mha->k_proj_weight.defined())
				updw(layers[i].mha->k_proj_weight, mdl.layers[i].mha->k_proj_weight);
			if (layers[i].mha->q_proj_weight.defined())
				updw(layers[i].mha->q_proj_weight, mdl.layers[i].mha->q_proj_weight);
			if (layers[i].mha->v_proj_weight.defined())
				updw(layers[i].mha->v_proj_weight, mdl.layers[i].mha->v_proj_weight);
			for (size_t pi = 0; pi < layers[i].trans->decoder.ptr()->parameters().size(); ++pi)
				updwp(layers[i].trans->decoder.ptr()->parameters()[pi], mdl.layers[i].trans->decoder.ptr()->parameters()[pi].data());//torch::nn::init::xavier_uniform_(p).cuda();
			for (size_t pi = 0; pi < layers[i].trans->encoder.ptr()->parameters().size(); ++pi)
				updwp(layers[i].trans->encoder.ptr()->parameters()[pi], mdl.layers[i].trans->encoder.ptr()->parameters()[pi].data());*/
				//for (size_t pi = 0; pi < layers[i].rnn1->all_weights().size(); ++pi)
				//	updwp(layers[i].rnn1->all_weights()[pi], mdl.layers[i].rnn1->all_weights()[pi].data());
		}//);
	}

	struct {
		torch::nn::LSTM rnn1 = nullptr;
		torch::nn::RNN rnn2 = nullptr;
		torch::nn::ConvTranspose1d cnvtr1 = nullptr;
		torch::nn::ConvTranspose1d cnvtr2 = nullptr;
		torch::nn::ConvTranspose1d cnvtr3 = nullptr;
		torch::nn::Linear lin1 = nullptr;
		torch::nn::Linear lin2 = nullptr;
		torch::nn::Linear lin3 = nullptr;
		torch::nn::Linear lin3_1 = nullptr;
		torch::nn::Linear lin4 = nullptr;
		torch::nn::Linear lin4_1 = nullptr;
		torch::nn::Linear lin5 = nullptr;
		torch::nn::Linear lin6 = nullptr;
		torch::nn::Linear lin7 = nullptr;
		torch::nn::Linear lin8 = nullptr;
		torch::nn::Conv1d cnv1 = nullptr;
		torch::nn::Conv1d cnv2 = nullptr;
		torch::nn::Conv1d cnv3 = nullptr;
		torch::nn::Conv1d cnv11 = nullptr;
		torch::nn::Conv1d cnv21 = nullptr;
		torch::nn::Conv1d cnv31 = nullptr;
		torch::nn::Conv1d cnv12 = nullptr;
		torch::nn::Conv1d cnv22 = nullptr;
		torch::nn::Conv1d cnv32 = nullptr;
		struct { torch::nn::Conv1d cnv = nullptr; } cnvs1d[8 * 5];
		struct { torch::nn::Conv1d cnv = nullptr; } cnvs1d1[8 * 5];
		//struct { torch::nn::LayerNorm no = nullptr; } lnorm[9];
		torch::nn::Conv1d cnv4 = nullptr;
		torch::nn::MultiheadAttention mha = nullptr;
		torch::nn::MultiheadAttention mha1 = nullptr;
		torch::Tensor rnnresh;
		torch::nn::BatchNorm1d batchntom1d = nullptr;
		torch::nn::Transformer trans = nullptr;
		torch::nn::Transformer trans1 = nullptr;
		torch::nn::Transformer trans2 = nullptr;
		torch::nn::Transformer trans3 = nullptr;
		torch::nn::Transformer trans4 = nullptr;
		torch::nn::TransformerDecoder transdec = nullptr;
		torch::nn::TransformerEncoder transenc = nullptr;
		torch::nn::LayerNorm norm1 = nullptr;
		torch::nn::LayerNorm norm2 = nullptr;
		torch::nn::LayerNorm norm2t = nullptr;
		torch::nn::LayerNorm norm3 = nullptr;
		torch::nn::LayerNorm norm11 = nullptr;
		torch::nn::LayerNorm norm21 = nullptr;
		torch::nn::LayerNorm norm31 = nullptr;
		torch::nn::LayerNorm norm4 = nullptr;
		torch::nn::LayerNorm norm5 = nullptr;
		torch::nn::MaxPool1d pool1 = nullptr;
		torch::nn::MaxPool1d pool2 = nullptr;
		struct { torch::nn::ConvTranspose1d cnv = nullptr; } tcnvs1d[8 * 5];
	} layers[5];
	torch::Tensor mem = torch::zeros({ 1, 20, 20 }).cuda();

	struct { torch::nn::Conv1d cnv = nullptr; } cnvs1df[8 * 5];
	torch::nn::Conv2d cnv2df = nullptr;
	torch::nn::Conv1d cnv1df = nullptr;

	torch::nn::Linear linf = nullptr;
};
TORCH_MODULE(Net);

struct Net2Impl : NetImpl {
	std::vector<torch::optim::OptimizerParamGroup> param_groups;
	Net2Impl() : NetImpl(nullptr) {
		reset();
	}
	void reset() override {
		//cnvf = torch::nn::Conv1d{ torch::nn::Conv1dOptions(40, 1, 6).bias(false)};
		//register_module("cnvf", cnvf);


		//cnv2df = torch::nn::Conv2d{ torch::nn::Conv2dOptions(1, 1, {20, 16}).bias(false)};
		//linf = torch::nn::Linear{ torch::nn::LinearOptions(400, 400) };
		//register_module("linf", linf);

		//cnv1df = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 21) };
		//linf = torch::nn::Linear{ torch::nn::LinearOptions(2, 1).bias(false) };
		//register_module("cnv1df", cnv1df);

		///tcnv1df = torch::nn::ConvTranspose1d{ torch::nn::ConvTranspose1dOptions(20, 20, 11) };
		//linf = torch::nn::Linear{ torch::nn::LinearOptions(4, 1).bias(false) };
		//register_module("tcnv1df", tcnv1df);
		std::vector<torch::Tensor> cnvpars;
		for (int i = 0; i < 1; ++i) {
			layers[i].rnn1 = torch::nn::LSTM(torch::nn::LSTMOptions(20, 20).num_layers(6).bidirectional(true));
			//layers[i].rnn2 = register_module("rnn2" + std::to_string(i), torch::nn::RNN(torch::nn::RNNOptions(40, 1).nonlinearity(torch::kReLU)));
			//layers[i].cnvtr1 = torch::nn::ConvTranspose1d{ torch::nn::ConvTranspose1dOptions(1, 20, 1) };
			//layers[i].cnvtr2 = torch::nn::ConvTranspose1d{ torch::nn::ConvTranspose1dOptions(20, 20, 4) };
			//layers[i].cnvtr3 = torch::nn::ConvTranspose1d{ torch::nn::ConvTranspose1dOptions(20, 20, 20).bias(false).dilation(161) };//register_module("cnvtr2" + std::to_string(i), torch::nn::ConvTranspose1d{ torch::nn::ConvTranspose1dOptions(20, 40, 100) });
			//layers[i].cnvtr3 = register_module("cnvtr3" + std::to_string(i), torch::nn::ConvTranspose1d{ torch::nn::ConvTranspose1dOptions(20, 40, 100) });
			//layers[i].lin2 = torch::nn::Linear{ torch::nn::LinearOptions(40, 20) };
			//layers[i].lin3 = torch::nn::Linear{ torch::nn::LinearOptions(1, 400).bias(false)};
			//layers[i].lin4 = register_module("lin4" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(40, 1) });
			//layers[i].lin4_1 = register_module("lin4_1" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(40, 40) });
			//layers[i].lin3_1 = register_module("lin3_1" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(40, 20) });
			//layers[i].lin6 = register_module("lin6" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(100, 40) });
			//layers[i].lin7 = register_module("lin7" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(40, 20) });
			//layers[i].cnv1 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(40, 1, 780).bias(false) };
			//layers[i].lin5 = register_module("lin5" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(20, 20) });

			if (i == 1, 0) {
				layers[i].cnv2 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(1, 1, 3).bias(false).dilation(1) };
				//layers[i].cnv1 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(1, 1, 3).bias(false).dilation(2).padding(1) };
				layers[i].cnv3 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(1, 1, 3).bias(false).dilation(4).padding(3) };

				layers[i].cnv21 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(1, 1, 3).bias(false).dilation(1) };
				layers[i].cnv11 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(1, 1, 3).bias(false).dilation(2).padding(1) };
				layers[i].cnv31 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(1, 1, 3).bias(false).dilation(4).padding(3) };

				layers[i].cnv22 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(1, 1, 5).bias(false) };
				layers[i].cnv12 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(1, 1, 5).bias(false) };
				layers[i].cnv32 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(1, 1, 5).bias(false) };
				//layers[i].cnv4 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(1, 1, 7).bias(false) };

				//layers[i].mha = torch::nn::MultiheadAttention{ torch::nn::MultiheadAttentionOptions(400, 8) };
				//layers[i].batchntom1d = torch::nn::BatchNorm1d{ torch::nn::BatchNorm1dOptions(20) };
				//layers[i].rnnresh = torch::zeros({ 2, 2 * 6, 20, 20 });
				//layers[i].trans = torch::nn::Transformer{ torch::nn::TransformerOptions().d_model(16).nhead(8).dropout(0.0) };
				//layers[i].trans1 = torch::nn::Transformer{ torch::nn::TransformerOptions().d_model(12).nhead(4).dropout(0.0) };
				//layers[i].trans2 = torch::nn::Transformer{ torch::nn::TransformerOptions().d_model(8).nhead(2).dropout(0.0) };

				//layers[i].norm1 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 20, 1}) };
				//layers[i].norm2 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({1, 400, 800}) };
				//layers[i].norm3 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 20, 1}) };
				//layers[i].norm4 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({1, 20, 1}) };
				//layers[i].norm5 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({1, 20, 20}) };

				layers[i].norm1 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 1, 18}) };
				layers[i].norm2 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 1, 16}) };
				layers[i].norm3 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 1, 14}) };

				layers[i].norm11 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 1, 12}) };
				layers[i].norm21 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 1, 10}) };
				layers[i].norm31 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 1, 8}) };
			}
			else {
				//layers[i].lin5 = register_module("lin5" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(20, 20) });
				//layers[i].cnv2 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 3) };
				//layers[i].cnv1 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 3) };
				//layers[i].cnv3 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 3) };

				//layers[i].cnv21 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 3).bias(false).dilation(1) };
				//layers[i].cnv11 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 3).bias(false).dilation(2).padding(1) };
				//layers[i].cnv31 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 3).bias(false).dilation(4).padding(3) };

				//layers[i].cnv22 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 5).bias(false) };
				//layers[i].cnv12 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 5).bias(false) };
				//layers[i].cnv32 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 5).bias(false) };
				//layers[i].cnv4 = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 3)};
				//if (i == 0)
				//layers[i].mha = torch::nn::MultiheadAttention{ torch::nn::MultiheadAttentionOptions(20, 4) };
				//layers[i].mha1 = torch::nn::MultiheadAttention{ torch::nn::MultiheadAttentionOptions(20, 4) };
				//layers[i].batchntom1d = torch::nn::BatchNorm1d{ torch::nn::BatchNorm1dOptions(20) };
				//layers[i].rnnresh = torch::zeros({ 2, 2 * 6, 20, 20 });
				//if (i == 0)
				//	layers[i].trans = torch::nn::Transformer{ torch::nn::TransformerOptions().d_model(20).nhead(4).dropout(0.0)};
				//layers[i].trans1 = torch::nn::Transformer{ torch::nn::TransformerOptions().d_model(20).nhead(4).dropout(0.0) };
				//layers[i].trans2 = torch::nn::Transformer{ torch::nn::TransformerOptions().d_model(20).nhead(4).dropout(0.0) };
				//layers[i].trans3 = torch::nn::Transformer{ torch::nn::TransformerOptions().d_model(20).nhead(4).dropout(0.0) };
				layers[i].trans4 = torch::nn::Transformer{ torch::nn::TransformerOptions().d_model(20).nhead(4).dropout(0.0) };
				//layers[i].transenc = torch::nn::TransformerEncoder{ torch::nn::TransformerEncoderOptions(torch::nn::TransformerEncoderLayerOptions(20, 4), 6) };
				//layers[i].transdec = torch::nn::TransformerDecoder{ torch::nn::TransformerDecoderOptions(torch::nn::TransformerDecoderLayerOptions(20, 4), 6) };
				//layers[i].norm1 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({2 * 6, 20, 20}) };
				//layers[i].norm2t = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 40}) };
				//layers[i].norm3 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 20, 1}) };
				//layers[i].norm4 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({1, 20, 1}) };
				//layers[i].norm5 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({1, 20, 20}) };

				//layers[i].norm1 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 18  }) };
				//layers[i].norm2 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({2 * 6, 20, 20}) };
				//layers[i].norm3 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 20  }) };

				//layers[i].norm11 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 12}) };
				//layers[i].norm21 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 10 }) };
				//layers[i].norm31 = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 8 }) };
				int di = 0;
				for (int y = 0; y < 1; ++y) {
					int chnls = 20;
					int tchnls = 4;
					/*for (int ii = 0; ii < 8; ++ii) {
						layers[i].cnvs1d1[y * 8 + ii].cnv = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 3).dilation(1 << di).padding((1 << di)) }; // .dilation(1 << di).padding((1 << di) - 1)
						register_module("cnvs1d1" + std::to_string(i) + std::to_string(y * 8 + ii), layers[i].cnvs1d1[y * 8 + ii].cnv);
						//if (ii < 8)
						//layers[i].tcnvs1d[y * 8 + ii].cnv = torch::nn::ConvTranspose1d{ torch::nn::ConvTranspose1dOptions(tchnls, tchnls + 2, 3 + (ii == 8) * 2) };
						//register_module("tcnvs1d" + std::to_string(i) + std::to_string(y * 8 + ii), layers[i].tcnvs1d[y * 8 + ii].cnv);
						//chnls -= 2;
						//tchnls += 2;
						//else
						//	chnls = 1;
						if (ii % 2 == 1)
							di += 1;
					}*/
					di = 0;
					for (int ii = 0; ii < 10; ++ii) {
						layers[i].cnvs1d[y * 8 + ii].cnv = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 3).dilation(1 << di).padding((1 << di) - 1) }; // .dilation(1 << di).padding((1 << di) - 1)
						register_module("cnvs1d" + std::to_string(i) + std::to_string(y * 8 + ii), layers[i].cnvs1d[y * 8 + ii].cnv);
						//if (ii < 8)
						//layers[i].tcnvs1d[y * 8 + ii].cnv = torch::nn::ConvTranspose1d{ torch::nn::ConvTranspose1dOptions(tchnls, tchnls + 2, 3 + (ii == 8) * 2) };
						//register_module("tcnvs1d" + std::to_string(i) + std::to_string(y * 8 + ii), layers[i].tcnvs1d[y * 8 + ii].cnv);
						//chnls -= 2;
						//tchnls += 2;
						//else
						//	chnls = 1;
						cnvpars.append_range(layers[i].cnvs1d[y * 8 + ii].cnv->parameters());
						if (ii % 2 == 1)
							di += 1;
					}

					//for (int ii = 0; ii < 11; ++ii, ++di) {
					//	layers[i].cnvs1d[y * 8 + ii].cnv = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 3) };
					//	register_module("cnvs1d" + std::to_string(i) + std::to_string(y * 8 + ii), layers[i].cnvs1d[y * 8 + ii].cnv);
					//}

					//if (y < 1)
#if 0
					chnls = 1;
					for (int ii = 0; ii < 9; ++ii) {
						tcnvs1df[y * 8 + ii].cnv = torch::nn::ConvTranspose1d{ torch::nn::ConvTranspose1dOptions(chnls, (ii == 8) + chnls + 2, (ii == 8) + 3) };
						register_module("tcnvs1df" + std::to_string(y * 8 + ii), tcnvs1df[y * 8 + ii].cnv);
						//cnvs1df[1].cnv = torch::nn::Conv1d{ torch::nn::Conv1dOptions(40, 1, 4).bias(false) };
						//register_module("cnvs1d1f" + std::to_string(1), cnvs1df[1].cnv);
						chnls += 2;
					}
#endif
				}

				//for (int ii = 0; ii < 18; ++ii) {
				//	layers[i].lnorm[ii].no = torch::nn::LayerNorm{ torch::nn::LayerNormOptions({20, 8 }) };
				//	register_module("lnorm" + std::to_string(i) + std::to_string(ii), layers[i].lnorm[ii].no);
				//}
			}

			//layers[i].pool1 = torch::nn::MaxPool1d{ torch::nn::MaxPool1dOptions(2) };
			//layers[i].pool2 = torch::nn::MaxPool1d{ torch::nn::MaxPool1dOptions(20) };

			register_module("rnn1" + std::to_string(i), layers[i].rnn1);
			//layers[i].rnn2 = register_module("rnn2" + std::to_string(i), torch::nn::RNN(torch::nn::RNNOptions(40, 1).nonlinearity(torch::kReLU)));
			//register_module("cnvtr1" + std::to_string(i), layers[i].cnvtr1);
			//register_module("cnvtr2" + std::to_string(i), layers[i].cnvtr2);
			//register_module("cnvtr3" + std::to_string(i), layers[i].cnvtr3);
			//layers[i].cnvtr2 = register_module("cnvtr2" + std::to_string(i), torch::nn::ConvTranspose1d{ torch::nn::ConvTranspose1dOptions(20, 1, 400) });
			//layers[i].cnvtr3 = register_module("cnvtr3" + std::to_string(i), torch::nn::ConvTranspose1d{ torch::nn::ConvTranspose1dOptions(20, 40, 100) });
			//register_module("lin1" + std::to_string(i), layers[i].lin1);
			//register_module("lin2" + std::to_string(i), layers[i].lin2);
			//register_module("lin3" + std::to_string(i), layers[i].lin3);

			//register_module("norm1" + std::to_string(i), layers[i].norm1);
			//register_module("norm2" + std::to_string(i), layers[i].norm2);
			//register_module("norm2t" + std::to_string(i), layers[i].norm2t);
			//register_module("norm3" + std::to_string(i), layers[i].norm3);
			//register_module("norm11" + std::to_string(i), layers[i].norm11);
			//register_module("norm21" + std::to_string(i), layers[i].norm21);
			//register_module("norm31" + std::to_string(i), layers[i].norm31);
			//register_module("norm4" + std::to_string(i), layers[i].norm4);
			//register_module("norm5" + std::to_string(i), layers[i].norm5);

			//register_module("pool1" + std::to_string(i), layers[i].pool1);
			//register_module("pool2" + std::to_string(i), layers[i].pool2);
			//layers[i].lin4 = register_module("lin4" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(40, 1) });
			//layers[i].lin4_1 = register_module("lin4_1" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(40, 40) });
			//layers[i].lin3_1 = register_module("lin3_1" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(40, 20) });
			//layers[i].lin6 = register_module("lin6" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(100, 40) });
			//layers[i].lin7 = register_module("lin7" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(40, 20) });
			//layers[i].cnv1 = register_module("cnv1" + std::to_string(i), layers[i].cnv1 );
			//layers[i].lin5 = register_module("lin5" + std::to_string(i), torch::nn::Linear{ torch::nn::LinearOptions(20, 20) });
			//register_module("cnv1" + std::to_string(i), layers[i].cnv1);

			//register_module("cnv2" + std::to_string(i), layers[i].cnv2);
			//register_module("cnv3" + std::to_string(i), layers[i].cnv3);
			//register_module("cnv4" + std::to_string(i), layers[i].cnv4);
			//register_module("cnv21" + std::to_string(i), layers[i].cnv21);
			//register_module("cnv31" + std::to_string(i), layers[i].cnv31);
			//register_module("cnv11" + std::to_string(i), layers[i].cnv11);

			//register_module("cnv22" + std::to_string(i), layers[i].cnv22);
			//register_module("cnv32" + std::to_string(i), layers[i].cnv32);
			//register_module("cnv12" + std::to_string(i), layers[i].cnv12);
			//if (i == 0)
			//register_module("mha" + std::to_string(i), layers[i].mha);
			//register_module("mha1" + std::to_string(i), layers[i].mha1);
			//register_module("bnm" + std::to_string(i), layers[i].batchntom1d);
			//register_parameter("rnnresh" + std::to_string(i), layers[i].rnnresh);
			//if (i == 0)
			//register_module("transenc" + std::to_string(i), layers[i].transenc);
			//register_module("transdec" + std::to_string(i), layers[i].transdec);
			//	register_module("trans" + std::to_string(i), layers[i].trans);
			//register_module("trans1" + std::to_string(i), layers[i].trans1);
			//register_module("trans2" + std::to_string(i), layers[i].trans2);
			//register_module("trans3" + std::to_string(i), layers[i].trans3);
			register_module("trans4" + std::to_string(i), layers[i].trans4);
		}
		to(device);
		calc_nels();
		auto opts = std::make_unique<torch::optim::NAdamOptions>(1e-6);
		opts->weight_decay(1e-6 * 100.);
		param_groups.push_back({ layers[0].trans4->parameters(), std::unique_ptr<torch::optim::OptimizerOptions>{dynamic_cast<torch::optim::OptimizerOptions*>(opts.release())} });
		opts = std::make_unique<torch::optim::NAdamOptions>(0.005);
		opts->weight_decay(0.005 * 100.);
		param_groups.push_back({ layers[0].rnn1->parameters(), std::unique_ptr<torch::optim::OptimizerOptions>{dynamic_cast<torch::optim::OptimizerOptions*>(opts.release())} });
		opts = std::make_unique<torch::optim::NAdamOptions>(0.05);
		opts->weight_decay(0.05 * 10.);
		param_groups.push_back({ cnvpars, std::unique_ptr<torch::optim::OptimizerOptions>{dynamic_cast<torch::optim::OptimizerOptions*>(opts.release())} });
		//layers[0].rnnresh = layers[0].rnnresh.requires_grad_(false);
	}




	//torch::Tensor forward(torch::Tensor input) {
	//	return forward(input, layers[0].rnnresh, true, 0).first;
	//}
	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input, torch::Tensor key, torch::Tensor val, torch::Tensor hlin, torch::Tensor* hlout, int i = 0) {
		torch::Tensor inputs[2];
		auto mdli = i;
		i = 0;
		//input = torch::relu(layers[i].cnv1(input));
#if 0
		if (input.ndimension() > 1) {
			input = input.reshape({ input.size(0), 20, 1 });
			if (btrain)
				input = layers[i].batchntom1d(input);
		}
		else
			input = input.reshape({ 1, 20, 1 });
#endif
		std::vector<torch::Tensor> out0, out1, outf;
		out0.resize(N_MODES);
#if 1
		out1.resize(N_MODES);
#endif
		//layers[i].rnn1->flatten_parameters();
		auto batchn = input.size(0);
		//auto inputl = torch::relu(layers[i].batchntom1d(input[i].reshape({ batchn, 20, 20 })));//.reshape({ 20, 20, 1 });//torch::relu(layers[i].cnv3(input.reshape({ 1, 20, 20 }))).reshape({ 1, 20, 1 });
		//auto rnnres = layers[i].rnn1(inputl, std::tuple{ hl[i][0].detach(), hl[i][1].detach() });
		//.select(-1, 0).select(-1, 0).reshape({ batchn, 1, 1})
		auto inputl = val;//torch::cat({ val, key }, 2).cuda();//torch::zeros({ 1, 20, 20 }).cuda();//input.select(-1, 0).select(-1, 0).unsqueeze(-1).unsqueeze(-1);//torch::relu(layers[i].cnvtr2(input.reshape({ batchn, 20, 20 }).select(-1, 0).select(-1, 0).reshape({ batchn, 1, 1 })));//[0][0]; //torch::relu(layers[i].batchntom1d(input[i].reshape({ batchn, 20, 20 })));//input;//.reshape({ batchn, 20, 20 });
		//torch::nn::init::xavier_uniform_(inputl);
		//inputl.zero_();
		torch::Tensor res = input, totranpsose;
		//inputl = torch::relu(layers[i].cnvtr2(inputl.toType(c10::ScalarType::Half)));
		//auto res = torch::relu(layers[i].cnv1(inputl));
		//if (betsitesrmade > 399) {

		//}
		//inputl = (std::get<0>(layers[i].mha(inputl, key, val)));
		int ii = 0;
		int y = 0;
#if 0
		//res = (layers[i].pool1((res)));
		//res = torch::relu(std::get<0>(layers[i].mha(res, input, input)));
		for (; y < 1; ++y) {
			//for (ii = 0; ii < 8; ++ii) {
				//if (inputl.size(-1) == 2)
				//	inputl = inputl.swapdims(-1, -2);
			//	res = torch::relu(layers[0].cnvs1d1[y * 8 + ii].cnv(res));
				//res = torch::relu(tcnvs1df[ii].cnv(res));
				//res = torch::relu(layers[0].cnvs1d[ii].cnv(res));

				//std::stringstream ss;
				//ss << " " << inputl.sizes() << std::endl;
				//orig_buf->sputn(ss.str().c_str(), ss.str().size());
			//}

			for (ii = 0; ii < 9; ++ii) {
				//if (inputl.size(-1) == 2)
				//	inputl = inputl.swapdims(-1, -2);
				//res = torch::relu(layers[0].cnvs1d1[ii].cnv(res));
				res = torch::relu(tcnvs1df[y * 8 + ii].cnv(res));
				//res = torch::relu(layers[0].cnvs1d[ii].cnv(res));

				//std::stringstream ss;
				//ss << " " << inputl.sizes() << std::endl;
				//orig_buf->sputn(ss.str().c_str(), ss.str().size());
			}

			//for (ii = 0; ii < 8; ++ii) {
				//if (inputl.size(-1) == 2)
				//	inputl = inputl.swapdims(-1, -2);
				//res = torch::relu(layers[0].cnvs1d1[ii].cnv(res));
				//res = torch::relu(tcnvs1df[ii].cnv(res));
			//	res = torch::relu(layers[0].cnvs1d[y * 8 + ii].cnv(res));

				//std::stringstream ss;
				//ss << " " << inputl.sizes() << std::endl;
				//orig_buf->sputn(ss.str().c_str(), ss.str().size());
			//}
		}

		//for (; ii < 6; ++ii) {
		//	res = torch::relu(tcnvs1df[ii].cnv(res));
		//}

		//res = torch::relu(layers[i].cnv2(inputl));
		//res = torch::relu(layers[i].trcnv2(res));

		inputl = res;
#endif

#if 0
		inputl = layers[i].transenc(inputl);
		auto enct = inputl;
		inputl = input;
		y = 0;
		for (ii = 0; ii < 8; ++ii) {
			//if (inputl.size(-1) == 2)
			//	inputl = inputl.swapdims(-1, -2);
			//inputl = torch::relu(layers[0].cnvs1d1[y * 8 + ii].cnv(inputl));
			//std::stringstream ss;
			//ss << ii << inputl.sizes() << std::endl;
			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
			inputl = (layers[i].cnvs1d1[y * 8 + ii].cnv(inputl));
			//if (ii == 8, 0)//;
			//	rnno = torch::sigmoid(rnno);
			//else
			inputl = torch::relu(inputl);


			//res = torch::relu(tcnvs1df[ii].cnv(res));
			//res = torch::relu(layers[0].cnvs1d[ii].cnv(res));

			//std::stringstream ss;
			//ss << " " << inputl.sizes() << std::endl;
			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
		}
#if 0
		auto rnnres = layers[i].rnn1(inputl, std::tuple{ hlin[i][0].toType(c10::ScalarType::Half), hlin[i][1].toType(c10::ScalarType::Half) });

		//std::cout << x20x40.sizes() << std::endl;
		//std::cout << std::get<0>(rnnres).sizes() << std::endl;
		//std::cout << std::get<1>(rnnres).sizes() << std::endl;

		auto rnno = (std::get<0>(rnnres));//std::get<0>(rnnres).chunk(2, -1)[0];//,
		//auto rnno1 = std::get<0>(rnnres).chunk(2, -1)[1];//,
		//rnno_h = torch::relu((layers[i].cnv1(((std::get<1>(rnnres)))))).swapaxes(0, 2);

	//std::cout << rnno.sizes() << std::endl;

	//std::cout << rnno_h.sizes() << std::endl;

		//std::cout << rnno.sizes() << std::endl;
		//std::cout << x20x100.sizes() << std::endl;
		//std::cout << x20x40.sizes() << std::endl;

		if (hlout) {
			//*hlout = hlout->detach().clone().cuda().contiguous();
			(*hlout)[i][0].copy_(std::get<0>(std::get<1>(rnnres)).detach().clone());//.detach().clone().cuda().contiguous();
			(*hlout)[i][1].copy_(std::get<1>(std::get<1>(rnnres)).detach().clone());//.detach().clone().cuda().contiguous();
		}

		//res = torch::relu(layers[i].cnv1(rnno));

		//res = torch::cat({ rnno.swapaxes(-1, -2), input }, -2);

		//rnno = torch::relu(layers[i].cnv1(rnno));

		//res = torch::cat({ rnno, input }, -2);

		inputl = layers[i].norm2t(rnno);
		inputl = torch::relu(layers[i].cnv1(inputl));
		inputl = (layers[i].trans(inputl, inputl));
#endif
		y = 0;
		for (ii = 0; ii < 8; ++ii) {
			inputl = (layers[i].tcnvs1d[y * 8 + ii].cnv(inputl));
			inputl = torch::relu(inputl);
		}
#endif
		//inputl = (layers[i].transdec(inputl, enct));

		/*y = 0;
		//inputl = rnno;
		for (ii = 0; ii < 8; ++ii) {
			//if (inputl.size(-1) == 2)
			//	inputl = inputl.swapdims(-1, -2);
			//inputl = torch::relu(layers[0].cnvs1d1[y * 8 + ii].cnv(inputl));
			//std::stringstream ss;
			//ss << ii << inputl.sizes() << std::endl;
			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
			inputl = (layers[i].cnvs1d1[y * 8 + ii].cnv(inputl));
			//if (ii == 8, 0)//;
			//	rnno = torch::sigmoid(rnno);
			//else
			inputl = torch::relu(inputl);


			//res = torch::relu(tcnvs1df[ii].cnv(res));
			//res = torch::relu(layers[0].cnvs1d[ii].cnv(res));

			//std::stringstream ss;
			//ss << " " << inputl.sizes() << std::endl;
			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
		}*/
#if 1
		/*auto rk = torch::tanh(layers[i].trans2(key, val));
		auto wk = torch::tanh(layers[i].trans3(key, val));
		auto er = torch::sigmoid(layers[i].trans(key, val));
		auto ad = torch::tanh(layers[i].trans1(key, val));

		auto simrr = torch::cosine_similarity(rk, mem);
		auto simrw = torch::cosine_similarity(wk, mem);

		auto wsr = torch::softmax(simrr, 1);
		auto rdata = torch::matmul(wsr, mem);

		auto wsw = torch::softmax(simrw, 1);
		auto erg = torch::matmul(wsw, er);
		auto adg = torch::matmul(wsw, ad);
		mem = mem * (1. - erg) + adg;


		auto mhares = std::get<0>(layers[i].mha1(rdata, mem, mem));*/
		//inputl = torch::relu(layers[i].cnvtr1(mhares));
		//std::stringstream ss;
		//ss << " mem" << mem << std::endl;
		//orig_buf->sputn(ss.str().c_str(), ss.str().size());
		auto rnnres = layers[i].rnn1(inputl, std::tuple{ hlin[i][0].toType(c10::ScalarType::Half), hlin[i][1].toType(c10::ScalarType::Half) });

		//std::cout << x20x40.sizes() << std::endl;
		//std::cout << std::get<0>(rnnres).sizes() << std::endl;
		//std::cout << std::get<1>(rnnres).sizes() << std::endl;

		auto rnno = (std::get<0>(rnnres));//std::get<0>(rnnres).chunk(2, -1)[0];//,
		//auto rnno1 = std::get<0>(rnnres).chunk(2, -1)[1];//,
		//rnno_h = torch::relu((layers[i].cnv1(((std::get<1>(rnnres)))))).swapaxes(0, 2);

	//std::cout << rnno.sizes() << std::endl;

	//std::cout << rnno_h.sizes() << std::endl;

		//std::cout << rnno.sizes() << std::endl;
		//std::cout << x20x100.sizes() << std::endl;
		//std::cout << x20x40.sizes() << std::endl;
		auto hl0 = std::get<0>(std::get<1>(rnnres));
		auto hl1 = std::get<1>(std::get<1>(rnnres));
		if (hlout) {
			//*hlout = hlout->detach().clone().cuda().contiguous();
			(*hlout)[i][0].copy_((hl0).detach().clone());//.detach().clone().cuda().contiguous();
			(*hlout)[i][1].copy_((hl1).detach().clone());//.detach().clone().cuda().contiguous();
		}

		//res = torch::relu(layers[i].cnv1(rnno));

		//res = torch::cat({ rnno.swapaxes(-1, -2), input }, -2);

		//rnno = torch::relu(layers[i].cnv1(rnno));

		//res = torch::cat({ rnno, input }, -2);

		//inputl = layers[i].norm2t(rnno);
		y = 0;
		inputl = rnno;
		for (ii = 0; ii < 10; ++ii) {
			//if (inputl.size(-1) == 2)
			//	inputl = inputl.swapdims(-1, -2);
			//inputl = torch::relu(layers[0].cnvs1d1[y * 8 + ii].cnv(inputl));
			//std::stringstream ss;
			//ss << ii << inputl.sizes() << std::endl;
			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
			inputl = (layers[i].cnvs1d[y * 8 + ii].cnv(inputl));
			//if (ii == 8, 0)//;
			//	rnno = torch::sigmoid(rnno);
			//else
			inputl = torch::relu(inputl);


			//res = torch::relu(tcnvs1df[ii].cnv(res));
			//res = torch::relu(layers[0].cnvs1d[ii].cnv(res));

			//std::stringstream ss;
			//ss << " " << inputl.sizes() << std::endl;
			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
		}
		//auto mhares = (layers[i].mha(oucnv, oucnv.reshape({ batchn, 10, 40 }), oucnv.reshape({ batchn, 10, 40 })));
		//inputl = std::get<0>(mhares).reshape_as(cnvres);

		//inputl = torch::relu(layers[i].cnvtr1(inputl));
		//inputl = (layers[i].cnvtr2(inputl));
		//inputl = (layers[i].norm3(inputl));
#endif
		//inputl = torch::relu(layers[i].cnv2(inputl));
		//inputl = torch::relu(layers[i].cnvtr2(inputl));
		//inputl = torch::relu(layers[i].trans(inputl, inputl));

		//auto mhares = (layers[i].mha(inputl, inputl, inputl));

		//inputl = (std::get<0>(mhares));
#if 0
		//totranpsose = res;

		//for (int ii = 0; ii < 2; ++ii) {
			//if (inputl.size(-1) == 2)
			//	inputl = inputl.swapdims(-1, -2);
		//	totranpsose = torch::relu(tcnvs1df[ii].cnv(totranpsose));
			//std::stringstream ss;
			//ss << " " << inputl.sizes() << std::endl;
			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
		//}

		//inputl = totranpsose;

		//for (int ii = 0; ii < 2; ++ii) {
			//if (inputl.size(-1) == 2)
			//	inputl = inputl.swapdims(-1, -2);
		//	(ii < 4 ? inputl : totranpsose) = torch::relu(layers[0].cnvs1d[ii].cnv(ii > 4 ? totranpsose : inputl));
			//std::stringstream ss;
			//ss << " " << inputl.sizes() << std::endl;
			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
		//}



		//inputl = inputl;




		//res = totranpsose;


		//rnno = torch::relu(layers[i].cnv1(rnno));

		//res = torch::cat({ rnno, input }, -2);

		//inputl = rnno;
		y = 0;
		for (; y < 1; ++y) {
			for (ii = 0; ii < 9; ++ii) {
				//if (inputl.size(-1) == 2)
				//	inputl = inputl.swapdims(-1, -2);
				//std::stringstream ss;
			//ss << ii << inputl.sizes() << std::endl;
			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
				inputl = torch::relu(layers[0].cnvs1d1[y * 8 + ii].cnv(inputl));
				//inputl = (tcnvs1df[y * 8 + ii].cnv(inputl));
				//if (ii == 18)//;
				//	inputl = torch::sigmoid(inputl);
				//else
				//	inputl = torch::relu(inputl);
				//res = torch::relu(tcnvs1df[ii].cnv(res));
				//res = torch::relu(layers[0].cnvs1d[ii].cnv(res));

				//std::stringstream ss;
				//ss << " " << inputl.sizes() << std::endl;
				//orig_buf->sputn(ss.str().c_str(), ss.str().size());
			}


		}
#endif
		//auto outfc = torch::cat(outf);
		//totranpsose = torch::relu(tcnv1df(totranpsose));
		//todrawres = inputl;
		//inputl = torch::stack({ inputl }, 1);

		//inputl = torch::relu(cnvf(inputl.reshape({ batchn, 20, 8 + 20 })));
		//inputl = torch::relu(std::get<0>(layers[i].mha(inputl.reshape({ batchn, 20, 20 }), featurescomb.reshape({ batchn, 20, 20 })
		//	, featurescomb.reshape({ batchn, 20, 20 }))));
		//if (i == 19) 

		//torch::Tensor outsrnn = inputl;
		//inputl.zero_();
		//for (int i = 0; i < 1; ++i) {

		for (int i = 0; i < 1; ++i) {
#if 0
			auto rnnres = layers[i].rnn1(inputl, std::tuple{ hlin[i][0].toType(c10::ScalarType::Half), hlin[i][1].toType(c10::ScalarType::Half) });

			//std::cout << x20x40.sizes() << std::endl;
			//std::cout << std::get<0>(rnnres).sizes() << std::endl;
			//std::cout << std::get<1>(rnnres).sizes() << std::endl;

			auto rnno = (std::get<0>(rnnres));//std::get<0>(rnnres).chunk(2, -1)[0];//,
			//auto rnno1 = std::get<0>(rnnres).chunk(2, -1)[1];//,
			//rnno_h = torch::relu((layers[i].cnv1(((std::get<1>(rnnres)))))).swapaxes(0, 2);

		//std::cout << rnno.sizes() << std::endl;

		//std::cout << rnno_h.sizes() << std::endl;

			//std::cout << rnno.sizes() << std::endl;
			//std::cout << x20x100.sizes() << std::endl;
			//std::cout << x20x40.sizes() << std::endl;
			auto wh0 = (std::get<0>(std::get<1>(rnnres)));
			auto wh1 = (std::get<1>(std::get<1>(rnnres)));
			//torch::Tensor hloutt = torch::vstack({ wh0, wh1 });
			if (hlout) {
				//*hlout = hlout->detach().clone().cuda().contiguous();
				(*hlout)[i][0].copy_(wh0.detach().clone());//.detach().clone().cuda().contiguous();
				(*hlout)[i][1].copy_(wh1.detach().clone());//.detach().clone().cuda().contiguous();
			}
			rnno = layers[i].norm2t(rnno);

			auto cnres = (layers[i].cnv1(rnno));
			//auto chks = rnno.chunk(2, -1);
			//auto fh = (layers[i].trans1(inputl, chks[0]));
			//auto bh = (layers[i].trans2(inputl, chks[1]));
			//auto res2 = (layers[i].trans2(inputl, chks[1]));
			//inputl = torch::relu(layers[i].cnv1(inputl));
			//inputl = torch::relu(layers[i].cnvtr1(inputl));
#endif


			//res = torch::relu(layers[i].cnv1(rnno));

			//res = torch::cat({ rnno.swapaxes(-1, -2), input }, -2);

			//rnno = torch::relu(layers[i].cnv1(rnno));

			//res = torch::cat({ rnno, input }, -2);
			//auto dirs = outsrnn[0].chunk(2, -1);
			//auto mhares = std::get<0>(layers[i].mha(input, dirs[0], dirs[1]));
			//mhares = torch::relu(cnv1df(mhares));
			//rnno = layers[i].norm2t(rnno);

			//if (ii == 8, 0)//;
			//	rnno = torch::sigmoid(rnno);
			//else

			//inputl = transres;
			//inputl = transres;//(layers[i].cnv2(rnno));

			//inputl = outsrnn;
		}
		//inputl = layers[i].norm2t(rnno);
		//inputl = torch::relu(layers[i].cnv1(inputl));
		inputl = layers[i].trans4(inputl, inputl);
		//inputl = (layers[i].trans(inputl, inputl));
		//auto lini = torch::relu(linf(input.flatten())).reshape_as(input);

		//inputl = layers[i].transdec(inputl, enct);
		todrawres = inputl;
		//inputl = (cnv1df(inputl));
		auto cnvout = inputl;
		//inputl = torch::relu(inputl);
//		inputl = input;
#if 0
		y = 0;
		for (ii = 0; ii < 9; ++ii) {
			//if (inputl.size(-1) == 2)
			//	inputl = inputl.swapdims(-1, -2);
			//inputl = torch::relu(layers[0].cnvs1d1[y * 8 + ii].cnv(inputl));
			//std::stringstream ss;
			//ss << ii << inputl.sizes() << std::endl;
			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
			inputl = (layers[i].cnvs1d1[y * 8 + ii].cnv(inputl));
			if (ii == 8, 0)//;
				inputl = torch::sigmoid(inputl);
			else
				inputl = torch::relu(inputl);
			//res = torch::relu(tcnvs1df[ii].cnv(res));
			//res = torch::relu(layers[0].cnvs1d[ii].cnv(res));

			//std::stringstream ss;
			//ss << " " << inputl.sizes() << std::endl;
			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
		}
#endif
		/*if (i == 0) {
			//auto dirs = outsrnn[0].chunk(2, -1);
			//auto mhares = std::get<0>(layers[i].mha(input, dirs[0], dirs[1]));
			//mhares = torch::relu(cnv1df(mhares));
			auto cnres = torch::relu(layers[i].cnv1(outsrnn[0]));
			inputl = (layers[i].trans(inputl, cnres));
		}
		else
			inputl = (cnvout);*/

		{
			//auto res1 = torch::sigmoid(layers[i].cnv4(inputl.reshape({ batchn, 20, 8 + 20 })));
			//auto res1 = inputl.clone().detach().cuda();

			//inputl = (linf((inputl.reshape({ batchn, 20, 38 }))));
			//inputl = (layers[i].norm1(inputl.reshape({ 20, 20, 1 })));

			//for (int ii = 0; ii < 4; ++ii) {
			//	inputl = torch::relu(cnvs1df[ii].cnv(inputl));
			//}

			//auto res0 = torch::sigmoid((inputl));//.squeeze(0);

			//res0 = torch::cat({ res0, rnno }, -1);

			//res0 = torch::sigmoid(res0);//linf(res0));

			//res0 = torch::sigmoid(cnvs1df[1].cnv(res0));

			//res0 = res0.reshape({ batchn, 1, 1, 1 });

			//res0 = (std::get<0>(layers[i].mha(input.reshape({ batchn, 20, 20 }), res0.reshape({ batchn, 20, 20 })
			//	, res0.reshape({ batchn, 20, 20 }))));

			//auto res1 = torch::sigmoid(layers[i].cnv4(inputl.reshape({ batchn, 20, 8 })));//torch::sigmoid(layers[i].cnv4(inputl.reshape({ batchn, 20, 8 + 20 })));

			//res0 = torch::sigmoid(res0);
			//auto tmp = torch::zeros({ batchn, 1, 2 }).cuda();
			//tmp.select_scatter(res1.select(-1, 0), -1, 0);
			//res0 = res0.reshape({ batchn * 20, 20, 20 });
			//res1 = res1.reshape({ batchn, 1, 1 });
#if 0
			if (input.ndimension() > 1)
				res0 = res0.reshape({ input.size(0), 1, i != 1 ? 20 : 4 });
			else
				res0 = res0.reshape({ 1, 1, i != 1 ? 20 : 4 });
#endif

#if 0
			//auto res1 = torch::relu(res);

			//std::stringstream ss;
			//ss << " " << std::hex << res << std::endl;
		//ss << "triggered at: " << res[2] << std::endl;
			//ss << torch::unsqueeze(mhres1, 0).sizes() << std::endl;
			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
			//res1 = torch::relu(layers[i].cnv1(torch::unsqueeze(mhres1, 0)));
			auto res1 = torch::relu(res.reshape({ batchn, 20, 20 }));
			res1 = torch::sigmoid(layers[i].cnv1(res1));

			//std::stringstream ss;
			//ss << " " << std::hex << res << std::endl;
		//ss << "triggered at: " << res[2] << std::endl;
		//ss << res1.sizes() << std::endl;
		//orig_buf->sputn(ss.str().c_str(), ss.str().size());

			res1 = res1.reshape({ batchn, 1, 4 });
#endif
			//if (i == 9) {

			out1[0] = inputl;
			out0[0] = inputl;//torch::sigmoid((transres));//torch::zeros({ 1, 20, 20 }).cuda();
			//}
			//else {
			//	inputl = torch::relu(res.reshape({ batchn, 20, 20 }));
			//}

			//return { res0, res1 };
		}
		//out0[0] = torch::zeros({ 1, 20, 20 }).cuda();
		return { torch::stack(out0).cuda(), torch::stack(out1).cuda() };
	}

};
TORCH_MODULE(Net2);

//static Net2 gwmodel, gwmodelmin;

static std::vector<torch::Tensor> lstarrnet;

static void clonemodelto(torch::nn::Cloneable<NetImpl>* inm, torch::nn::Cloneable<NetImpl>* outm) {

	torch::serialize::OutputArchive out{};
	//fwdhl = fwdhlb.clone().cuda();
	std::stringstream iot;
	//fwdhlb = torch::zeros({  5, 2, 2 * 6, 20, 20 }).cuda();
	//fwdhl = torch::zeros({  5, 2, 2 * 6, 20, 20 }).cuda();
	//testt->layers[0].rnnresh = torch::zeros({ 2, 2 * 6, 20, 20 }).cuda();

	inm->save(out);

	out.save_to(iot);

	//out.save_to("mdl_bsa.pt");
	torch::serialize::InputArchive in{};//::OutputArchive out{};

	iot.seekg(0);

	in.load_from(iot);

	outm->load(in);
}


static Net* testl;

BOOL WINAPI consoleHandler(DWORD signal) {

	if (signal == CTRL_C_EVENT) {
		if (testl) {
			savestuff(true, **testl, 0);
			//std::terminate();
			condt4 = !condt4;
		}
	}

	//printf("Ctrl-C handled\n"); // do cleanup

	return TRUE;
}


/*uint32_t unpackColor(float f) {
	std::array <uint16_t, 3> color;
	color[2] = floor(f / 256.0 / 256.0);
	color[1] = floor((f - color[2] * 256.0 * 256.0) / 256.0);
	color[0] = floor(f - color[2] * 256.0 * 256.0 - color[0] * 256.0);
	// now we have a vec3 with the 3 components in range [0..255]. Let's normalize it!
	return color[0] | (color[1] << 8) | (color[2] << 16);
}*/

void GetHardwareAdapter(IDXGIFactory4* pFactory, IDXGIAdapter1** ppAdapter)
{
	*ppAdapter = nullptr;
	for (UINT adapterIndex = 0; ; ++adapterIndex)
	{
		IDXGIAdapter1* pAdapter = nullptr;
		if (DXGI_ERROR_NOT_FOUND == pFactory->EnumAdapters1(adapterIndex, &pAdapter))
		{
			// No more adapters to enumerate.
			break;
		}

		// Check to see if the adapter supports Direct3D 12, but don't create the
		// actual device yet.
		if (SUCCEEDED(D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
		{
			*ppAdapter = pAdapter;
			if (adapterIndex > 0)
				return;
		}
		pAdapter->Release();
	}
}
std::atomic_int upds = 0;
std::atomic_int64_t bal = 0;
std::atomic_int64_t rbal = 0;
std::atomic_int64_t vbal = 0;
std::atomic_int64_t vbalt = 0;
std::atomic_int64_t vbalb = 0;
std::atomic_int64_t vbalactual = 0;
std::atomic_int64_t vbalmax = INT64_MIN;
std::atomic_int64_t vbalmin = INT64_MAX;
static bool wasahead = false;
std::atomic_int64_t vbalmaxlst = INT64_MIN;
std::atomic_int64_t vbalminlst = INT64_MAX;
float lstvbal2 = 0;
std::atomic_int64_t lstlstvbal2 = 0;
std::atomic_int64_t lstvbal = 0;
float vbal2 = 0;
std::atomic_int64_t vbal2max = INT64_MIN;
std::atomic_int64_t vbal2min = INT64_MAX;
std::atomic_int64_t vbal2minitesr = 0;
std::atomic_int64_t vbal2maxitesr = 0;
std::atomic_int64_t vbal2pa = 0;
int64_t vbal2pas[20] = { 0 };
std::atomic_int64_t vbal2palast = 0;
std::atomic_int64_t vbal3 = 0;
std::atomic_int64_t rbal2 = 0;
std::atomic_int64_t vbalm[3] = {};
std::atomic_int64_t rrbal = 0;
std::atomic_int64_t rrbalt = 0;
std::atomic_int64_t rrbalv = 0;
std::atomic_int64_t rrbalmaxv = INT64_MIN;
std::atomic_int64_t rrbalminv = INT64_MAX;
std::atomic_int64_t lstrrbal = 0;
std::atomic_int64_t iterslearn = 100;
std::atomic_bool endcond = false;
std::atomic_int64_t maxbal = 0;
std::atomic_int64_t minbal = 0;
std::atomic_int64_t rmaxbal = 0;
std::atomic_int64_t rminbal = 0;
#define DEF_BET_AMNTF 0.00006//0.0001//0.002//0.00002
std::atomic_int64_t betamnt = 100000000 * DEF_BET_AMNTF;
float betamntfd = DEF_BET_AMNTF;//0.0;
std::atomic_uint64_t wins = 0;
std::atomic_uint64_t winst = 0;
std::atomic_uint64_t winsconseq = 0;
std::atomic_uint64_t losses = 0;
std::atomic_uint64_t lossest = 0;
std::atomic_int64_t monbal = 0;
std::atomic_int64_t maxmonbal = 0;
std::atomic_int64_t minmonbal = 0;
std::atomic_int64_t thrcount = 0;
std::atomic<float> ylav = 0;
std::mutex ylavupd;
std::atomic_bool rlup = false;
std::atomic_uint64_t requests = 0;
std::mutex dobetm;
std::condition_variable cnd;
std::barrier bar(2);
static float betamntf = ((double)betamnt / 100000000);
static float betamntft = ((double)betamnt / 100000000);

float genranf() {
	unsigned long long rnd;
	while (!_rdseed64_step(&rnd));
	auto i0 = *(unsigned long*)&rnd;
	auto i1 = 1[(unsigned long*)&rnd];
	auto min = std::min(i0, i1);
	auto max = std::max(i0, i1);
	return (float)min / (float)max;//(double)rnd / UINT64_MAX;
}

void upd_bal(int resi) {
	if (resi >= 0) {
		if (!resi) {//!resi) {
			bal += betamnt;
			wins += 1;
			monbal += 50;
			winsconseq += 1;
			//float lr = optim.param_groups()[0].options().get_lr();
			//optim.~decltype(optim)();
			//new (&optim)torch::optim::Adam(test->parameters(), torch::optim::AdamOptions{ lr });
			//optim.param_groups()[0].set_options(std::unique_ptr<torch::optim::OptimizerOptions>(new torch::optim::AdamOptions{ optim.param_groups()[0].options().get_lr() * 10 }));
		}
		else {
			bal -= betamnt;
			monbal -= 50;
			winsconseq = 0;
			losses += 1;
			//betamnt = betamnt * 2;
			//optim.param_groups()[0].set_options(std::unique_ptr<torch::optim::OptimizerOptions>(new torch::optim::AdamOptions{ optim.param_groups()[0].options().get_lr() / 10 }));
		}
		{
			//std::unique_lock lck(ylavupd);
			float ylavtmp = ylav;
			ylavtmp += bal;
			if (upds > 0) {
				ylav = ylavtmp / 2;
			}
			else {
				ylav = ylavtmp;
			}
		}
		maxmonbal = std::max(maxmonbal, monbal).load();
		minmonbal = std::max(minmonbal, monbal).load();
		upds += 1;
		int64_t maxbaltmp = std::max(bal, maxbal);
		int64_t minbaltmp = std::min(bal.load(), minbal.load());

		if (maxbaltmp != maxbal) {
			maxbal += (maxbaltmp - maxbal);
		}

		if (minbaltmp != minbal) {
			minbal -= (minbal - minbaltmp);
		}

		/*if (rlup) {
			if (!resi) {//!resi) {
				rbal += 50;
				//float lr = optim.param_groups()[0].options().get_lr();
				//optim.~decltype(optim)();
				//new (&optim)torch::optim::Adam(test->parameters(), torch::optim::AdamOptions{ lr });
				//optim.param_groups()[0].set_options(std::unique_ptr<torch::optim::OptimizerOptions>(new torch::optim::AdamOptions{ optim.param_groups()[0].options().get_lr() * 10 }));
			}
			else {
				rbal -= 50;
				//optim.param_groups()[0].set_options(std::unique_ptr<torch::optim::OptimizerOptions>(new torch::optim::AdamOptions{ optim.param_groups()[0].options().get_lr() / 10 }));
			}
		}*/
	}
}

#define LOAD_SAVE 0
#define LOAD_SAVE_H 0
#define LOAD_SAVE_IO 0
#define LOAD_SAVE_PREDS 0

int dobet(bool above, float amnt, float ch, volatile int &resnum) {
	STARTUPINFO si;
	ZeroMemory(&si, sizeof(STARTUPINFO));
	si.cb = sizeof(si);
	SECURITY_ATTRIBUTES saAttr;
	saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
	saAttr.bInheritHandle = TRUE;
	saAttr.lpSecurityDescriptor = NULL;

	CHAR chBuf[260]{};

	HANDLE g_hChildStd_OUT_Rd = NULL;
	HANDLE g_hChildStd_OUT_Wr = NULL;

	CreatePipe(&g_hChildStd_OUT_Rd, &g_hChildStd_OUT_Wr, &saAttr, 0);
	si.dwFlags |= STARTF_USESTDHANDLES;
	si.hStdOutput = g_hChildStd_OUT_Wr;

	PROCESS_INFORMATION pi;
	ZeroMemory(&pi, sizeof(pi));

	std::string args = "C:\\Users\\Administrator\\.bun\\bin\\bun.exe \"C:\\Users\\Administrator\\Documents\\torchprj2\\dobet.mjs\" ";

	if (above) {
		args += "above";
	}
	else {
		args += "below";
	}

	auto chm = std::to_string(ch);
	chm.resize(5);

	//if (fifty) {
		args += " " + chm;
	//}
	//else {
	//	args += " false";
	//}

	args += " ";
	args += std::to_string(amnt) + "";
	int resi = -1;
	//{
		//std::unique_lock lck(dobetm);
		//if (requests = -1)
		//	return -1;
		//requests += 1;
	requests += 1;
	//bar.arrive_and_wait();
	//dobetm.lock();
	//if (!rlup)
	while (resi < 0 || resi > 1)
	{
		BOOL b = CreateProcessA("C:\\Users\\Administrator\\.bun\\bin\\bun.exe", args.data(), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
		DWORD dwretw = WaitForSingleObject(pi.hProcess, INFINITE);  // wait for process to end
		DWORD dwExitCode = -1;
		if (dwretw == WAIT_OBJECT_0)
			::GetExitCodeProcess(pi.hProcess, &dwExitCode);
		else
			TerminateProcess(pi.hProcess, -1);
		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);
		resi = (signed char)dwExitCode;
		std::cout << "exit code dw " << dwExitCode << std::endl;
	}
	ReadFile(g_hChildStd_OUT_Rd, chBuf, 260, NULL, NULL);
	std::stringstream sstream(std::string{ chBuf });
	sstream >> (int &)resnum;
	requests -= 1;
	//dobetm.unlock();
	/*if (resi >= 0) {
			rmaxbal = std::max(rbal, rmaxbal).load();
			rminbal = std::min(rbal, rminbal).load();
			rbal += !resi ? amnt *100000000 : -(amnt * 100000000);

			if (rbal > 0) {
				rlup = true;
			}
		}*/
		//}
		//dobetm.unlock();
	CloseHandle(g_hChildStd_OUT_Rd);
	CloseHandle(g_hChildStd_OUT_Wr);
	return resi;
}

int64_t rttseed(std::string sd) {
	STARTUPINFO si;
	ZeroMemory(&si, sizeof(STARTUPINFO));
	si.cb = sizeof(si);
	SECURITY_ATTRIBUTES saAttr;
	saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
	saAttr.bInheritHandle = TRUE;
	saAttr.lpSecurityDescriptor = NULL;

	CHAR chBuf[260]{};

	HANDLE g_hChildStd_OUT_Rd = NULL;
	HANDLE g_hChildStd_OUT_Wr = NULL;

	CreatePipe(&g_hChildStd_OUT_Rd, &g_hChildStd_OUT_Wr, &saAttr, 0);
	si.dwFlags |= STARTF_USESTDHANDLES;
	si.hStdOutput = g_hChildStd_OUT_Wr;

	PROCESS_INFORMATION pi;
	ZeroMemory(&pi, sizeof(pi));
	//HANDLE oldh = GetStdHandle(STD_OUTPUT_HANDLE);

	//SetStdHandle(STD_OUTPUT_HANDLE, hMapFile);

	std::string args = "C:\\Users\\Administrator\\.bun\\bin\\bun.exe \"C:\\Users\\Administrator\\source\\repos\\TorchProject2\\randomiseseed.mjs\" ";

	args += sd + "\0";

	int64_t resi = -4;
	DWORD dwExitCode = -1;

	//while (dwExitCode != 0 || resi < 0) {
	BOOL b = CreateProcessA("C:\\Users\\Administrator\\.bun\\bin\\bun.exe", args.data(), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
	DWORD dwretw = WaitForSingleObject(pi.hProcess, INFINITE);  // wait for process to end

	if (dwretw == WAIT_OBJECT_0)
		::GetExitCodeProcess(pi.hProcess, &dwExitCode);
	else
		TerminateProcess(pi.hProcess, -1);
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
	//SetStdHandle(STD_OUTPUT_HANDLE, oldh);
	ReadFile(g_hChildStd_OUT_Rd, chBuf, 260, NULL, NULL);
	std::stringstream sstream(std::string{ chBuf });
	if (dwExitCode == 0) {

		sstream >> resi;
	}

	orig_buf->sputn(chBuf, strnlen_s(chBuf, 260));
	//}
	//std::cout << "resirtsed " << resi << std::endl;
	CloseHandle(g_hChildStd_OUT_Rd);
	CloseHandle(g_hChildStd_OUT_Wr);
	return resi;
}

uint64_t getbetamnt() {
	STARTUPINFO si;
	ZeroMemory(&si, sizeof(STARTUPINFO));
	si.cb = sizeof(si);
	SECURITY_ATTRIBUTES saAttr;
	saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
	saAttr.bInheritHandle = TRUE;
	saAttr.lpSecurityDescriptor = NULL;

	CHAR chBuf[260]{};

	HANDLE g_hChildStd_OUT_Rd = NULL;
	HANDLE g_hChildStd_OUT_Wr = NULL;

	CreatePipe(&g_hChildStd_OUT_Rd, &g_hChildStd_OUT_Wr, &saAttr, 0);
	si.dwFlags |= STARTF_USESTDHANDLES;
	si.hStdOutput = g_hChildStd_OUT_Wr;

	PROCESS_INFORMATION pi;
	ZeroMemory(&pi, sizeof(pi));
	//HANDLE oldh = GetStdHandle(STD_OUTPUT_HANDLE);

	//SetStdHandle(STD_OUTPUT_HANDLE, hMapFile);

	std::string args = "C:\\Users\\Administrator\\.bun\\bin\\bun.exe \"C:\\Users\\Administrator\\Documents\\torchprj2\\getballtc.mjs\" ";


	uint64_t resi = 0;
	DWORD dwExitCode = -1;

	while (dwExitCode != 0) {
		BOOL b = CreateProcessA("C:\\Users\\Administrator\\.bun\\bin\\bun.exe", args.data(), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
		DWORD dwretw = WaitForSingleObject(pi.hProcess, INFINITE);  // wait for process to end

		if (dwretw == WAIT_OBJECT_0)
			::GetExitCodeProcess(pi.hProcess, &dwExitCode);
		else
			TerminateProcess(pi.hProcess, -1);
		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);
		//SetStdHandle(STD_OUTPUT_HANDLE, oldh);
		if (dwExitCode == 0) {
			ReadFile(g_hChildStd_OUT_Rd, chBuf, 260, NULL, NULL);
			std::stringstream sstream(std::string{ chBuf });
			sstream >> resi;
		}
	}
	std::cout << "betamnt " << resi << std::endl;
	CloseHandle(g_hChildStd_OUT_Rd);
	CloseHandle(g_hChildStd_OUT_Wr);
	return resi;
}




//#define _rdseed64_step terryrnd64

#define _rdseed32_step(a) terryrnd64(a, 32)

int terryrnd16(unsigned short* u16) {
	*u16 = getrngterry();
	return 1;
}

std::string hexStr(const uint8_t* data, int len)
{
	std::stringstream ss;
	ss << std::hex;

	for (int i(0); i < len; ++i)
		ss << std::setw(2) << std::setfill('0') << (int)data[i];

	return ss.str();
}

void make_hash(std::string in, BYTE** hash, long* len, ALG_ID hash_ty)
{
	HCRYPTPROV hProv = 0;
	HCRYPTHASH hHash = 0;
	BYTE* pbHash = NULL;
	DWORD dwHashLen;

	DWORD dwCount;
	DWORD i;;

	if (!CryptAcquireContext(&hProv, "torch_project_test_set", MS_ENH_RSA_AES_PROV, PROV_RSA_AES, CRYPT_NEWKEYSET)) {
		if (!CryptAcquireContext(&hProv, "torch_project_test_set", MS_ENH_RSA_AES_PROV, PROV_RSA_AES, 0))
			return;
	}
	if (!CryptCreateHash(hProv, hash_ty, 0, 0, &hHash)) {
		return;
	}

	if (!CryptHashData(hHash, (const BYTE*)in.c_str(), in.size(), 0)) {
		return;
	}

	dwCount = sizeof(DWORD);
	if (!CryptGetHashParam(hHash, HP_HASHSIZE, (BYTE*)&dwHashLen, &dwCount, 0)) {
		return;
	}
	if ((pbHash = (unsigned char*)malloc(dwHashLen)) == NULL) {
		return;
	}


	memset(pbHash, 0, dwHashLen);

	if (!CryptGetHashParam(hHash, HP_HASHVAL, pbHash, &dwHashLen, 0)) {
		return;
	}

	*hash = pbHash;
	*len = dwHashLen;

	if (hHash) CryptDestroyHash(hHash);
	if (hProv) CryptReleaseContext(hProv, 0);
}

std::string hash256(std::string in, ALG_ID hash_ty = CALG_SHA_256) {
	BYTE* pb = NULL;
	long len = 0;
	make_hash(in, &pb, &len, hash_ty);
	std::string out = hexStr(pb, len);
	free(pb);
	return out;
}

// max 8 nonce

int getRoll(std::string serverSeed, std::string clientSeed, int nonce) {
	const std::string hash = hash256(serverSeed + clientSeed + std::to_string(nonce), CALG_SHA_512);
	int index = 0;
	int lucky;
	do {
		lucky = std::stol(hash.substr(index, 5), nullptr, 16);
		index += 5;
	} while (lucky >= 1000000);

	return lucky % 10000;
}

/*LONG WINAPI
VectoredHandler1(
	struct _EXCEPTION_POINTERS* ExceptionInfo
)
{
	//UNREFERENCED_PARAMETER(ExceptionInfo);

	std::cout << std::hex << ((UCHAR*)ExceptionInfo->ContextRecord->Rip - (UCHAR*)GetModuleHandleA(NULL)) << std::endl;
	system("PAUSE");

	//Actual[0] = Sequence++;
	return EXCEPTION_CONTINUE_SEARCH;
}*/


//#include "torch/torch.h"


int main(int, char**) {
	//DeleteFile(".\\test2.pt");
	//DeleteFile(".\\fwdhlblout.pt");
	cublasContext* cublas_handle;
	cublasStatus_t stat = cublasCreate(&cublas_handle);
	torch::nn::init::xavier_uniform_(fwdhlbl2);
	torch::nn::init::xavier_uniform_(fwdhlbl2w);
	torch::nn::init::xavier_uniform_(fwdhlbl2l);
	fwdhlbl2o = fwdhlbl2.clone();
	static Net testsmpl;
	static std::array<size_t, 1000> ics = { 0 };
	static auto dogeneratesmpls = [&](int n, torch::Tensor* totrainop, torch::Tensor* totraintop) {
#if 1
		concurrency::parallel_for(int(0), n, [&](int x)
			{
				torch::Tensor inp = torch::zeros({ 20 }).cuda(), lstinp = torch::zeros({ 20 }).cuda(), inp2 = torch::zeros({ 20 }).cuda();
				lstinp = lstinp.toType(c10::ScalarType::Float);
				for (int i = 0; i < 20; ++i) {
					unsigned long long rndval = {};
					//if (i > 10)
					//if (i >= 10, !pseudoswitch)
					//	if (i % 2 == !!switchswitch)
					//		getrngt(rndval);
					///	else
					getrng(rndval);
					//else
					//	getrngp(rndval);
					lstinp[i] = rndval;
					//std::cout << inp[i] << std::endl;
				}

				lstinp /= UINT64_MAX;
				lstinp = lstinp.cuda().toType(c10::ScalarType::Half);
				lstinp /= 10.;

				inp2 = inp2.toType(c10::ScalarType::Float);
				for (int i = 0; i < 20; ++i) {
					unsigned long long rndval = {};
					//if (i < 10)
					//if (i >= 10, false)
					//	getrngp(rndval);
					//else
					//	if (i % 2 == !!switchswitch)
					//		getrngt(rndval);
					//	else
					//		getrng(rndval);
					getrng(rndval);
					inp2[i] = rndval;
					//std::cout << inp[i] << std::endl;
				}

				inp2 /= UINT64_MAX;
				//inp2 = inp2.cuda().toType(c10::ScalarType::Half);
				inp2 /= 10.;

				inp = inp.toType(c10::ScalarType::Float);
				for (int i = 0; i < 20; ++i) {
					unsigned long long rndval = {};
					//if (i >= 10, false)
					//	getrngp(rndval);
					//else
					//	if (i % 2 == !switchswitch)
					//		getrngt(rndval);
					//	else
					// 
					//		getrng(rndval);
					getrng(rndval);
					inp[i] = rndval;
					//std::cout << inp[i] << std::endl;
				}

				inp /= UINT64_MAX;
				inp = torch::pow(inp, inp2);
				inp = inp.cuda().toType(c10::ScalarType::Half);
				inp *= inp2;
				inp /= 10.;
				//inp /= inp2;

				//lstinp = torch::sqrt(lstinp);
				//lstinp /= inp2;
#if 0
				{
					unsigned long long rndval = {};
					getrngt(rndval);
					lstinp = torch::zeros({ 20 }).cuda();//torch::flip(inp, { 0 }).cuda();
					lstinp[((double)rndval / ((double)UINT64_MAX + 1)) * 20] = 1.0;//;
				}
#endif
				//inp = torch::pow(inp, inp2) / inp2;
				//inp = inp.cuda().toType(c10::ScalarType::Half);
				//std::stringstream ss;
				//ss << "inp " << inp << std::endl;
				//orig_buf->sputn(ss.str().c_str(), ss.str().size());
				/*if (fliptrain, true) {
					for (int i = 0; i < 20; ++i) {
						inp[i] = 1.0 - inp[i].item().toFloat();
					}
				}*/
				//lstinpri = inp.clone().cuda();
				{
					std::unique_lock lk{ trainreslk };
					//if (totrain.ndimension() == 1)
					//	trainhl = fwdhl.clone().cuda();
					//if (totrain2.ndimension() == 1)
					//	trainhl2 = fwdhl;
					/*if (totrain.size(0) > 20, false) {
						auto inp = totrain.slice(0, 0, 1);
						test->forward(inp.reshape({ 1, 20, 1 }), trainhl);
						trainhl = test->layers[0].rnnresh;
						totrain = totrain.slice(0, 1, totrain.size(0));
						totrainto = totrainto.slice(0, 1, totrainto.size(0));
					}*/
					//if (totrain.ndimension() == 1)
					//	trainhl = ntrainhl;
					//trainhl = fwdhl;
					if (false)
					{
						totrain = torch::slice(totrain.clone().cuda(), 0, 1, 20).cuda();
						//totrain1 = torch::slice(totrain1.clone().cuda(), 0, 1, 20).cuda();
						//totrain2 = torch::slice(totrain2.clone().cuda(), 0, 1, 20).cuda();
					}
					if (true) {
						std::stringstream ss;
						//ss << "totrain.size(0) " << totrain.size(0) << std::endl;
						//orig_buf->sputn(ss.str().c_str(), ss.str().size());
						if (totrainop) {
							auto& totrain = *totrainop;
							totrain = totrain.ndimension() == 1 ? torch::reshape(inp.clone().cuda(), { 1, 1, 20 }) : torch::vstack({ totrain, torch::reshape(inp.clone().cuda(), {1, 1, 20}) });
						}
						//totrain1 = totrain1.ndimension() == 1 ? torch::reshape(inp2.clone().cuda(), { 1, 1, 20 }) : torch::vstack({ totrain1, torch::reshape(inp2.clone().cuda(), {1, 1, 20}) });
						//totrain2 = totrain2.ndimension() == 1 ? torch::reshape(refinp.clone().cuda(), { 1, 1, 20 }) : torch::vstack({ totrain2, torch::reshape(refinp.clone().cuda(), {1, 1, 20}) });


						//trainreslk.unlock();
						//condvartrain.notify_one();
					}
					if (false)
					{
						totrainto = torch::slice(totrainto, 0, 1, 20).cuda();
						//totrainto2 = torch::slice(totrainto2, 0, 1, 20).cuda();
					}
					if (true) {
						//std::stringstream ss;
						//ss << "totrainto.size(0) " << totrainto.size(0) << std::endl;
						//ss << res1 << std::endl;
						//orig_buf->sputn(ss.str().c_str(), ss.str().size());
						if (totraintop) {
							torch::Tensor& totrainto = *totraintop;
							totrainto = totrainto.ndimension() == 1 ? torch::reshape(lstinp.clone().cuda(), { 1, 1, 20 }) : torch::vstack({ totrainto, torch::reshape(lstinp.clone().cuda(), {1, 1, 20}) });

						}//totrainto2 = totrainto2.ndimension() == 1 ? torch::reshape(lstinp2.clone().cuda(), { 1, 1, 20 }) : torch::vstack({ totrainto2, torch::reshape(lstinp2.clone().cuda(), {1, 1, 20}) });
					}
					//if (coefbal > 0.7 && !resi) {
					//	totrain2 = totrain2.ndimension() == 1 ? torch::reshape(inp, { 1, 1, 20 }) : torch::vstack({ totrain2, torch::reshape(inp, {1, 1, 20}) });
					//	totrainto2 = totrainto2.ndimension() == 1 ? torch::reshape(lstinp, { 1, 1, 20 }) : torch::vstack({ totrainto2, torch::reshape(lstinp, {1, 1, 20}) });
					//}
				}

			});
#else
		torch::data::datasets::MNIST mn{ "D:\\Users\\sasho\\Downloads\\MNIST" };

		static std::vector<torch::data::Example<>> smpls;
		if (ics[0] == 0) {
			std::iota(ics.begin(), ics.end(), 0);

		}
		smpls = mn.get_batch(ics);
		auto& totrain = *totrainop;
		auto& totrainto = *totraintop;
		for (auto& smpl : smpls) {
			//std::stringstream ss;

			//ss << smpl.data << std::endl;
			auto lstinp = torch::one_hot(smpl.target, 10).toType(c10::ScalarType::Float);
			totrain = totrain.ndimension() == 1 ? torch::reshape(smpl.data.toType(torch::get_default_dtype_as_scalartype()).clone().cuda(), { 1, 28, 28 }) : torch::vstack({ totrain, torch::reshape(smpl.data.toType(torch::get_default_dtype_as_scalartype()).clone().cuda(), {1, 28, 28}) });
			totrainto = totrainto.ndimension() == 1 ? torch::reshape(lstinp.clone().cuda(), { 1, 1, 10 }) : torch::vstack({ totrainto, torch::reshape(lstinp.clone().cuda(), {1, 1, 10}) });

			//orig_buf->sputn(ss.str().c_str(), ss.str().size());
			//std::exit(0);
		}
#endif
		};
	//AddVectoredExceptionHandler(1, VectoredHandler1);
	{
		unsigned int num;
		while (!_rdseed32_step(&num));
		in32srvmseed = num;
		mstsrvrng.seed(in32srvmseed);
		while (!_rdseed32_step(&num));
		in32clmseed = num;
		mstclrng.seed(in32clmseed);
		while (!_rdseed32_step(&num));
		in32mseed = num;
		mstrng.seed(in32mseed);
	}
	orig_buf = std::cout.rdbuf();
	mainthreadid = GetCurrentThreadId();
	static std::string serverSeed;
	static std::string serverSeedlspec;
	static int noncem;
	static std::string serverSeedlspecm;
	static std::string clientSeed;
	static std::string clientSeedm;
	static std::atomic_int nonce;
	unsigned int num;
	while (!_rdseed32_step(&num));
	std::mt19937 rngm(num);
	std::uniform_int_distribution<std::mt19937::result_type> distm(0, 1);
	int64_t orbal;
	int64_t ormaxbal;
	int64_t orminbal;
	double orminbalahead;
	//int64_t iterslearn = 100;
	float coefbal = 0.;
	static float coefbalt = 0.;
	double coefbalsum = 0.0;
	double coefbalavlst = 0.5;
	double coefbalavlstbar = 0.1;
	double coefbalav = 0.0;
	float lstcoefbal = coefbal;
	float lstcoefbalav = coefbalav;
	double coefbalavdist = 0.0;
	double coefbalavdistlst = 0.1;
	double lssavsum = 0.0;
	double lssav = 0.0;
	double lssavdist = 0.0;
	double lssavdistmin = DBL_MAX;
	double lssavdistmax = DBL_MIN;
	double lstlssav = 0.0;
	double minlssav = DBL_MAX;
	double maxlssav = DBL_MIN;
	float dist;
	bool condtll4m = false;
	//static bool haswon = true;
	int resitlm = 0;

	auto start_t = std::chrono::steady_clock::now();
	/*0std::cout << "pos1" << std::endl;
	system("PAUSE");
	GetCursorPos(&p1);
	std::cout << "pos2" << std::endl;
	system("PAUSE");
	GetCursorPos(&p2);
	std::cout << "pos3/reset bet button" << std::endl;
	system("PAUSE");
	GetCursorPos(&p3);*/
	std::cout.rdbuf(NULL);
	//betamnt = getbetamnt() / 10;
	//bool betcnd = false;
	//char nw;
	//uint64_t tt;
	//std::cout << hash256("test") << std::endl;
	torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(c10::ScalarType::Half));
#if 0
	{
		std::stringstream ss;

		ss << torch::globalContext().hasCuDNN() << std::endl;
		ss << torch::globalContext().userEnabledCuDNN() << std::endl;
		ss << torch::globalContext().userEnabledCuDNNSDP() << std::endl;
		ss << torch::globalContext().versionCuDNN() << std::endl;
		ss << torch::globalContext().allowTF32CuBLAS() << std::endl;
		ss << torch::globalContext().allowTF32CuDNN() << std::endl;
		//ss << torch::globalContext().() << std::endl;

		orig_buf->sputn(ss.str().c_str(), ss.str().size());
		exit(0);
	}
#endif
	torch::globalContext().setAllowTF32CuDNN(true);
	{

		{

			std::ifstream ifb("bal.txt");
			double tmp;
			int con;
			ifb >> tmp;
			orbal = tmp * 100000000.;
			ifb >> tmp;
			ormaxbal = tmp * 100000000.;
			ifb >> tmp;
			orminbal = tmp * 100000000.;
			ifb >> orminbalahead;
			int ltm;
			ifb >> ltm;
			nonce = ltm;
			ifb >> serverSeed;
			ifb >> clientSeed;
			ifb >> serverSeedlspec;
			ifb >> con;
			ifb >> rolllosti;
			//ifb >> minlss;
			//ifb >> std::hex >> in32srvmseed;
			//mstsrvrng.seed(in32srvmseed);
			//std::ifstream ifbd("dist.bin");
			//mstsrvdist.
			//condt5 = con;
			globaliters = con;
			condt5 = false;
			condt4 = false;
			rolllosti = 0;
		}
		//#ifdef REAL_BAL
		if (REAL_BAL) {
			orbal = getbetamnt();
			//	bal_gen = true;
		}
		betamnt = orbal / 50;
		//betamntfd = (double)(orbal / 1000) / 100000000;

		//while (1) terryrnd64(&tt);
		betamntf = ((double)betamnt / 100000000);

		auto betamntt = orbal / 200;

		betamntflss = betamntfd;

		betamntft = betamntt;//((double)(orbal / 50) / 100000000);
#if 1
		HWND hwndcraeted{};
		hwndcraeted = GetConsoleWindow();//hwndtmp;
		if (HWND hwndtmp = CreateWindowEx(0, MAKEINTATOM(32770), "Game", WS_BORDER | WS_CAPTION | WS_POPUP, 0, 0, 640, 480, hwndcraeted, NULL, NULL, NULL)) {
			ShowWindow(hwndtmp, SW_SHOW);//MINNOACTIVE);

			hwndcraeted = hwndtmp;
		}

		//uint32_t *bmapdata = new uint32_t[100 * 100];
		//BITMAPINFO bmpinf = { {sizeof(BITMAPINFOHEADER), 100, 100, 1, 32, BI_RGB} };
		//Sleep(-1);
		using Microsoft::WRL::ComPtr;
#if defined(_DEBUG)
		// Enable the D3D12 debug layer.
		{

			ComPtr<ID3D12Debug> debugController;
			if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
			{
				debugController->EnableDebugLayer();
			}
		}
#endif

		ComPtr<IDXGIFactory4> factory;
		CreateDXGIFactory1(IID_PPV_ARGS(&factory));

		ComPtr<IDXGIAdapter1> hardwareAdapter;
		GetHardwareAdapter(factory.Get(), &hardwareAdapter);

		ComPtr<ID3D12Device> m_device;

		D3D12CreateDevice(
			hardwareAdapter.Get(),
			D3D_FEATURE_LEVEL_11_0,
			IID_PPV_ARGS(&m_device)
		);

		// Describe and create the command queue.
		D3D12_COMMAND_QUEUE_DESC queueDesc = {};
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

		ComPtr<ID3D12CommandQueue> m_commandQueue;

		m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue));

		// Describe and create the swap chain.
		DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
		swapChainDesc.BufferCount = 2;
		swapChainDesc.BufferDesc.Width = 640;
		swapChainDesc.BufferDesc.Height = 480;
		swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
		swapChainDesc.OutputWindow = hwndcraeted;
		swapChainDesc.SampleDesc.Count = 1;
		swapChainDesc.Windowed = TRUE;

		ComPtr<IDXGISwapChain> swapChain;
		factory->CreateSwapChain(
			m_commandQueue.Get(),        // Swap chain needs the queue so that it can force a flush on it.
			&swapChainDesc,
			&swapChain
		);

		ComPtr<IDXGISwapChain3> m_swapChain;

		swapChain.As(&m_swapChain);

		// This sample does not support fullscreen transitions.
		factory->MakeWindowAssociation(hwndcraeted, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);

		UINT m_frameIndex;

		//m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

		ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
		UINT m_rtvDescriptorSize;

		// Create descriptor heaps.
		{
			// Describe and create a render target view (RTV) descriptor heap.
			D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
			rtvHeapDesc.NumDescriptors = 2;
			rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
			rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
			m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap));

			m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		}


		ComPtr<ID3D12Resource> m_renderTargets[2];
		// Create frame resources.
		{
			D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();

			// Create a RTV for each frame.
			for (UINT n = 0; n < 2; n++)
			{
				m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n]));
				m_device->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr, rtvHandle);
				rtvHandle.ptr += m_rtvDescriptorSize;
			}
		}

		ComPtr<ID3D12CommandAllocator> m_commandAllocator;

		m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator));

		ComPtr<ID3D12RootSignature> m_rootSignature;

		// Create an empty root signature.
		{
			D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc{};
			rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

			ComPtr<ID3DBlob> signature;
			ComPtr<ID3DBlob> error;
			D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
			m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature));
		}

		ComPtr<ID3D12PipelineState> m_pipelineState;

		// Create the pipeline state, which includes compiling and loading shaders.
		{
			ComPtr<ID3DBlob> vertexShader;
			ComPtr<ID3DBlob> pixelShader;
			ComPtr<ID3DBlob> geometryShader;
			//ComPtr<ID3DBlob> computeShader;

#if defined(_DEBUG)
			// Enable better shader debugging with the graphics debugging tools.
			UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
			UINT compileFlags = 0;
#endif

			D3DCompileFromFile(L"shaders.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", compileFlags, 0, &vertexShader, nullptr);
			D3DCompileFromFile(L"shaders.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", compileFlags, 0, &pixelShader, nullptr);
			D3DCompileFromFile(L"shaders.hlsl", nullptr, nullptr, "GSMain", "gs_5_0", compileFlags, 0, &geometryShader, nullptr);

			// Define the vertex input layout.
			D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
			};

			// Describe and create the graphics pipeline state object (PSO).
			D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
			psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
			psoDesc.pRootSignature = m_rootSignature.Get();
			psoDesc.VS = { reinterpret_cast<UINT8*>(vertexShader->GetBufferPointer()), vertexShader->GetBufferSize() };
			psoDesc.PS = { reinterpret_cast<UINT8*>(pixelShader->GetBufferPointer()), pixelShader->GetBufferSize() };
			//psoDesc.GS = { reinterpret_cast<UINT8*>(geometryShader->GetBufferPointer()), geometryShader->GetBufferSize() };
			psoDesc.RasterizerState = D3D12_RASTERIZER_DESC{ .FillMode = D3D12_FILL_MODE_SOLID, .CullMode = D3D12_CULL_MODE_NONE, .DepthClipEnable = FALSE, .ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF };
			psoDesc.BlendState = D3D12_BLEND_DESC{ .RenderTarget = { {
				.SrcBlend = D3D12_BLEND_ONE,
				.DestBlend = D3D12_BLEND_ZERO,
				.BlendOp = D3D12_BLEND_OP_ADD,
				.SrcBlendAlpha = D3D12_BLEND_ONE,
				.DestBlendAlpha = D3D12_BLEND_ZERO,
				.BlendOpAlpha = D3D12_BLEND_OP_ADD,
				.LogicOp = D3D12_LOGIC_OP_NOOP,
				.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL
				}
				}
			};
			psoDesc.DepthStencilState.DepthEnable = FALSE;
			psoDesc.DepthStencilState.StencilEnable = FALSE;
			psoDesc.SampleMask = UINT_MAX;
			psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			psoDesc.NumRenderTargets = 1;
			psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
			psoDesc.SampleDesc.Count = 1;
			m_device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_pipelineState));
		}

		ComPtr<ID3D12GraphicsCommandList> m_commandList;

		// Create the command list.
		m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(), m_pipelineState.Get(), IID_PPV_ARGS(&m_commandList));

		// Command lists are created in the recording state, but there is nothing
		// to record yet. The main loop expects it to be closed, so close it now.
		m_commandList->Close();



		const UINT vertexBufferSize = sizeof(Vertex[50]);
		const UINT indexBufferSize = sizeof(uint16_t[50]);

		D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView{};
		D3D12_INDEX_BUFFER_VIEW m_indexBufferView{};
		D3D12_RESOURCE_DESC desc = { .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
				.Alignment = 0, .Width = vertexBufferSize,
				.Height = 1, .DepthOrArraySize = 1, .MipLevels = 1,
				.Format = DXGI_FORMAT_UNKNOWN, .SampleDesc = {.Count = 1, .Quality = 0},
				.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_NONE };// = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);

		D3D12_RESOURCE_DESC descind = { .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
				.Alignment = 0, .Width = indexBufferSize,
				.Height = 1, .DepthOrArraySize = 1, .MipLevels = 1,
				.Format = DXGI_FORMAT_R16_UINT, .SampleDesc = {.Count = 1, .Quality = 0},
				.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_NONE };

		// Create the vertex buffer.
		{
			// Define the geometry for a triangle.
			/*Vertex triangleVertices[] =
			{
				{ { 0.0f, 0.25f , 0.0f, 1.0f, 0.0f, 0.0f, 1.0f } },
				{ { 0.25f, -0.25f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f } },
				{ { -0.25f, -0.25f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f } }
			};*/

			// Note: using upload heaps to transfer static data like vert buffers is not 
			// recommended. Every time the GPU needs it, the upload heap will be marshalled 
			// over. Please read up on Default Heap usage. An upload heap is used here for 
			// code simplicity and because there are very few verts to actually transfer.
			D3D12_HEAP_PROPERTIES heapProps{ .Type = D3D12_HEAP_TYPE_DEFAULT,
					.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
					.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
					.CreationNodeMask = 1,
					.VisibleNodeMask = 1 };

			/*m_device->CreateCommittedResource(
				&heapProps,
				D3D12_HEAP_FLAG_SHARED,
				&desc,
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&m_vertexBuffer));* /

			// Copy the triangle data to the vertex buffer.
			/*UINT8* pVertexDataBegin;
			D3D12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
			m_vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin));
			memcpy(pVertexDataBegin, triangleVertices, vertexBufferSize);
			m_vertexBuffer->Unmap(0, nullptr);*/

			// Initialize the vertex buffer view.
			//m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
			//m_vertexBufferView.StrideInBytes = sizeof(Vertex);
			//m_vertexBufferView.SizeInBytes = vertexBufferSize;
		}

		ComPtr<ID3D12Fence> m_fence;
		UINT64 m_fenceValue;
		HANDLE m_fenceEvent;

		// Create synchronization objects and wait until assets have been uploaded to the GPU.
		{
			m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
			m_fenceValue = 1;

			// Create an event handle to use for frame synchronization.
			m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);


			// Wait for the command list to execute; we are reusing the same command 
			// list in our main loop but for now, we just want to wait for setup to 
			// complete before continuing.
			/*const UINT64 fence = m_fenceValue;
			m_commandQueue->Signal(m_fence.Get(), fence);
			m_fenceValue++;

			// Wait until the previous frame is finished.
			if (m_fence->GetCompletedValue() < fence)
			{
				m_fence->SetEventOnCompletion(fence, m_fenceEvent);
				WaitForSingleObject(m_fenceEvent, INFINITE);
			}

			m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();*/
		}//*/
#endif
		/*torch::Tensor tensor = torch::rand({2, 3});
		if (torch::cuda::is_available()) {
			std::cout << "CUDA is available! Training on GPU" << std::endl;
			auto tensor_cuda = tensor.cuda();
			std::cout << tensor_cuda << std::endl;
		}
		else
		{
			std::cout << "CUDA is not available! Training on CPU" << std::endl;
			std::cout << tensor << std::endl;
		}*/

		//torch::Device device(torch::kCUDA);

		///torch::Device devicecpu(torch::kCPU);

		//, test2;
		//std::shared_ptr<torch::nn::Module> testcp;

		//Net test2;

		//test2.to(device);	

		static Net test, testtb, testt, testrn, testrn1, test3, test4, test5, test6;
		static Net2 test2, smpl;




		testt->ema_update(0, *test, !dirw ? 0.01 : -0.01, [](torch::Tensor& w, const torch::Tensor& w1, double decay) {
			//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
			//w.set_data(w.data().fmod(w1.data()));
			//w.set_data(w.data().fmod(w1.data()));
			w.set_data(w1.data().clone().detach());
			torch::Tensor wd = w1.data().unsqueeze(0);
			std::vector<long long> szsz(wd.ndimension(), 1);
			szsz[0] = 30;
			wd = wd.repeat(szsz);
			//wd[0] = w.data().clone().detach();
			lstarrnet.push_back(wd);
			//if (!dirwlal)
			//	dirwlal = torch::any(w.data() > 2.).item().toBool() || torch::any(w.data() < -2.).item().toBool();
			////if (!dirwlb)
			//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
			//if (shrink)
			//w.set_data(w.data().uniform_(-1, 1));
			//w.clip
			//std::stringstream ss;
//ss << "rrbal: " << rrbal << std::endl;
//ss << "triggered at: " << res1[16] << std::endl;
//ss << torch::mean(w) << std::endl;
			//gr3 = (torch::mean(w) >= 3.0).item().toBool();
			//w.set_data(-w.data());

//orig_buf->sputn(ss.str().c_str(), ss.str().size());
			}, [](torch::Tensor& w, const torch::Tensor& w1, double decay) {
				//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
				//w.set_data(w.data().fmod(w1.data()));
				//w.set_data(w.data().fmod(w1.data()));
				w.set_data(w1.data().clone().detach());
				torch::Tensor wd = w1.data().unsqueeze(0);
				std::vector<long long> szsz(wd.ndimension(), 1);
				szsz[0] = 30;
				wd = wd.repeat(szsz);
				//wd[0] = w.data().clone().detach();
				lstarrnet.push_back(wd);
				//if (!dirwlal)
				//	dirwlal = torch::any(w.data() > 2.).item().toBool() || torch::any(w.data() < -2.).item().toBool();
				//if (!dirwlb)
				//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
				//w.set_data(-w.data());
				//if (shrink)
				//w.set_data(w.data() - (0.001 * (1) * (1)) * w.data());
				});
#if 0
			auto lstarriter = lstarrnet.begin();
			static std::vector<torch::Tensor> lstarrnetsum;
			//auto lstarritersum = lstarrnet.begin();
			testt->ema_update(0, *test, 50, [&lstarriter](torch::Tensor& w, const torch::Tensor& w1, double decay) {
				//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
				//w.set_data(w1.data());
				//w.set_data(w1.data());
				//*lstarriter = torch::vstack({ *lstarriter, w.data().clone().detach().cuda().unsqueeze(0) });
				//(*lstarriter)[0] = w1.data().clone().detach().cuda();
				//(*lstarriter) = torch::roll(*lstarriter, 1, 0).cuda();
				w.set_data(torch::sum(*lstarriter, 0).cuda());
				//w.set_data(w.data() * (2. / w.data().norm()));

				lstarrnetsum.push_back(torch::sum(*lstarriter, 0).cuda());
				lstarriter += 1;
				//w.set_data(w1.data());
				//w.set_data(w.data() / torch::mean(w.data()));
				//w.set_data(torch::pow(w.data() * (2. / w.data().norm()), (itesr * 2)));
				//torch::nn::LayerNorm()()
				//w.set_data(w.data() * ((10 ) / w.data().norm()));
				//w.set_data(w.data().fmod(w1.data()));

				//if (!dirwlb)
				//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
				//if (shrink)
				//w.set_data(w.data().uniform_(-1, 1));
				//w.clip
				//std::stringstream ss;
	//ss << "rrbal: " << rrbal << std::endl;
	//ss << "triggered at: " << res1[16] << std::endl;
	//ss << torch::mean(w) << std::endl;
				//gr3 = (torch::mean(w) >= 3.0).item().toBool();
				//w.set_data(-w.data());

	//orig_buf->sputn(ss.str().c_str(), ss.str().size());
				}, [&lstarriter](torch::Tensor& w, const torch::Tensor& w1, double decay) {
					//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
					//w.set_data(w1.data());
					//(*lstarriter)[0] = w.data().clone().detach().cuda();
					//(*lstarriter) = torch::roll((*lstarriter), 1);
					//*lstarriter = torch::vstack({ *lstarriter, w.data().clone().detach().cuda().unsqueeze(0) });
					//(*lstarriter)[0] = w1.data().clone().detach().cuda();
					//(*lstarriter) = torch::roll(*lstarriter, 1, 0).cuda();
					w.set_data(torch::sum(*lstarriter, 0).cuda());
					//w.set_data(w.data() * (2. / w.data().norm()));

					lstarrnetsum.push_back(torch::sum(*lstarriter, 0).cuda());
					lstarriter += 1;
					//w.set_data(w1.data());
					//w.set_data(torch::sum(*lstarriter++, 0).cuda());
					//w.set_data(torch::pow(w.data() * (2. / w.data().norm()), 2.));
					//w.set_data((w.data() / (w.data().mean() * w.data().norm()) / itesr));
					//w.set_data(w.data().fmod(w1.data()));
					//if (!dirwlb)
					//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
					//w.set_data(-w.data());
					//if (shrink)
					//w.set_data(w.data() - (0.001 * (1) * (1)) * w.data());
					});
#endif
				if (std::filesystem::exists("mdl_bsa.pt") && LOAD_SAVE) {

					//torch::serialize::InputArchive in{};//::OutputArchive out{};

					//in.load_from("mdl_bsa.pt");

					//test->load(in);//.save(out);
					//test2->load(in);

					//test->layers[1] = testt->layers[1];
					//testt->load(in);//.save(out);
					//test2->load(in);
					torch::load(test2, "mdl_bsa.pt");
					//testt->layers[1] = test->layers[1];


				}
				else {
					//test->to(device);
					//testt->to(device);

				}
				//for (auto& ref : lstarrnetsum) {
				//	allw = torch::cat({ torch::flatten(allw), torch::flatten(ref) }).cuda();
				//}
				//allw = allw.reshape({ 1, 20, 30, 121 });
			//auto t = gwmodel->forward(lstarrnet);
			/*auto lstarriter = lstarrnet.begin();
			testt->ema_update(0, *testt, 50, [&lstarriter](torch::Tensor& w, const torch::Tensor& w1, double decay) {
				//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
				//w.set_data(w1.data());
				//w.set_data(w1.data());
				//auto wdator = w.data().unsqueeze(0).clone().cuda();
				//*lstarriter = torch::vstack({ *lstarriter, w.data().clone().detach().cuda().unsqueeze(0) });
				w.set_data(torch::sum(*lstarriter++, 0).clone().cuda());//w1.data().clone().detach().cuda());
				//auto wdator = w.data().unsqueeze(0).clone().cuda();
				//if (vbal2 > lstvbal2)
				//if (lstarriter->size(0) >= 20)
				//*lstarriter = wdator;
				//lstarriter += 1;
				//*lstarriter = lstarriter->resize_({ 1, -1 });
				//w.set_data(w.data() / torch::mean(w.data()));
				//w.set_data(torch::pow(w.data() * (2. / w.data().norm()), (itesr * 2)));
				//torch::nn::LayerNorm()()
				//w.set_data(w.data() * ((10 ) / w.data().norm()));
				//w.set_data(w.data().fmod(w1.data()));

				//if (!dirwlb)
				//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
				//if (shrink)
				//w.set_data(w.data().uniform_(-1, 1));
				//w.clip
				//std::stringstream ss;
	//ss << "rrbal: " << rrbal << std::endl;
	//ss << "triggered at: " << res1[16] << std::endl;
	//ss << torch::mean(w) << std::endl;
				//gr3 = (torch::mean(w) >= 3.0).item().toBool();
				//w.set_data(-w.data());

	//orig_buf->sputn(ss.str().c_str(), ss.str().size());
				}, [&lstarriter](torch::Tensor& w, const torch::Tensor& w1, double decay) {
					//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
					//w.set_data(w1.data());
					//auto wdator = w.data().unsqueeze(0).clone().cuda();
					//*lstarriter = torch::vstack({ *lstarriter, w.data().clone().detach().cuda().unsqueeze(0) });
					w.set_data(torch::sum(*lstarriter++, 0).clone().cuda());
					//auto wdator = w.data().unsqueeze(0).clone().cuda();
					//if (vbal2 > lstvbal2)
					//if (lstarriter->size(0) >= 20)
					//*lstarriter = wdator;
					//lstarriter += 1;
					//*lstarriter = lstarriter->resize_({ 1, -1 });
					//w.set_data(torch::pow(w.data() * (2. / w.data().norm()), 2.));
					//w.set_data((w.data() / (w.data().mean() * w.data().norm()) / itesr));
					//w.set_data(w.data().fmod(w1.data()));
					//if (!dirwlb)
					//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
					//w.set_data(-w.data());
					//if (shrink)
					//w.set_data(w.data() - (0.001 * (1) * (1)) * w.data());
					});*/
					//lstarrnet.push_back(fwdhlbl.reshape({ 1, 2, 2 * 6, 20, 20 }).clone().detach().cuda());
				test3->ema_update(0, *test, !dirw ? 0.01 : -0.01, [](torch::Tensor& w, const torch::Tensor& w1, double decay) {
					//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
					//w.set_data(w.data().fmod(w1.data()));
					//w.set_data(w.data().fmod(w1.data()));
					w.set_data(w1.data());

					//if (!dirwlal)
					//	dirwlal = torch::any(w.data() > 2.).item().toBool() || torch::any(w.data() < -2.).item().toBool();
					////if (!dirwlb)
					//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
					//if (shrink)
					//w.set_data(w.data().uniform_(-1, 1));
					//w.clip
					//std::stringstream ss;
		//ss << "rrbal: " << rrbal << std::endl;
		//ss << "triggered at: " << res1[16] << std::endl;
		//ss << torch::mean(w) << std::endl;
					//gr3 = (torch::mean(w) >= 3.0).item().toBool();
					//w.set_data(-w.data());

		//orig_buf->sputn(ss.str().c_str(), ss.str().size());
					}, [](torch::Tensor& w, const torch::Tensor& w1, double decay) {
						//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
						//w.set_data(w.data().fmod(w1.data()));
						//w.set_data(w.data().fmod(w1.data()));
						w.set_data(w1.data());
						//if (!dirwlal)
						//	dirwlal = torch::any(w.data() > 2.).item().toBool() || torch::any(w.data() < -2.).item().toBool();
						//if (!dirwlb)
						//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
						//w.set_data(-w.data());
						//if (shrink)
						//w.set_data(w.data() - (0.001 * (1) * (1)) * w.data());
						});

					torch::Tensor fwdhlb = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();

					if (std::filesystem::exists("lstrnnresh.pt") && LOAD_SAVE_H) {
						torch::load(fwdhlbl, "lstrnnresh.pt");
						//lstrnnresh = lstrnnresh.cuda();
						//fwdhlb[0] = lstrnnresh.clone().cuda();
					}

					toinfer = torch::ones({ 20, 20 }).cuda();
					toinferm = torch::ones({ N_MODES, 20, 20 }).cuda();
					toinferm2 = torch::ones({ N_MODES, 20, 20 }, c10::ScalarType::Long).cuda();
					toinfermi = torch::ones({ N_MODES, 20, 20 }, c10::ScalarType::Long).cuda();
					toinfero = torch::ones({ N_MODES, 20, 20 }).cuda();
					toinferin = torch::zeros({ 20, 20 }).cuda();
					tolrn = torch::ones({ 20, 20 }).cuda();
					totrain = torch::ones({ 20, 20 }).cuda();
					totrainto = torch::ones({ 20, 20 }).cuda();
					static torch::Tensor totrainldummy = torch::ones({ 20, 20 }).cuda();
					totrain /= 10;
					totrainto /= 10;
					totrainldummy /= 10;
					tolrn /= 10;
					toinfer /= 10;
					tolrn3sz /= 2.;

					for (int x = 0; x < 20; ++x)
						for (int y = 0; y < 20; ++y) {
							toinfer[x][y] = y < 10 ? (y % 10) / 10. : (9 - (y % 10)) / 10.;//0.0;//0.1;//(y + x * 20) / 400.;//y / 20.;//(y + x * 20) / 400.;//(y + x * 20) / 400.;// 0.1;
							for (int i = 0; i < N_MODES; ++i)
								toinferm[i][x][y] = y < 10 ? (y % 10) / 10. : (9 - (y % 10)) / 10.,//;//0.0;//0.1;//(y + x * 20) / 400.;//y / 20.;//(y + x * 20) / 400.;//(y + x * 20) / 400.;// 0.1;
								toinfero[i][x][y] = y < 10 ? (y % 10) / 10. : (9 - (y % 10)) / 10.,
								toinfermi[i][x][y] = y + x * 20 + 1;//(y + x * 20) < 200 ? ((y + x * 20) % 200) : (199 - ((y + x * 20) % 200));//y + x * 20;

							//toinfero[x][y] = 0.1;//(y < 10 ? (y % 10) / 10. : (9 - (y % 10)) / 10.);//(y + x * 20) / 400.;
							tolrn[x][y] = 0.1;//y < 10 ? 0.1 : 0.0;//0.1;//y < 10 ? (y % 10) / 10. : (9 - (y % 10)) / 10.;//y < 10 ? 0.1 : 0.0;//y < 10 ? 0.1 : 0.0;//(y % 10) / 10. : (9 - (y % 10)) / 10.;//(y + x * 20) / 400.;//(y % 10) / 10.;
							tolrn0[x][y] = 0.1;//y < 10 ? 0.1 : 0.0;
							tolrn1[x][y] = (y + x * 20) / 400.;
							tolrn3[x][y] = (y % 10) / 10.;
						}
					for (int i = 0; i < 20; ++i) {
						toinfermii[i][0] = i;
						toinfermii[i][1] = 10;
					}
					torch::nn::init::xavier_uniform_(toinferm);
					toinferm.zero_();
					//if (std::filesystem::exists("toinfermi.pt")) {
					//	torch::load(toinfermi, "toinfermi.pt");
						//lstrnnresh = lstrnnresh.cuda();
						//fwdhlb[0] = lstrnnresh.clone().cuda();
					//}
					//toinfermi.zero_();
					toinfermio2 = toinfermi.clone().detach();
					toinfermi2 = toinfermi.clone().detach();
					//torch::nn::init::xavier_uniform_(toinfero);
					//toinferm.uniform_();
					toinfero = toinferm.clone().cuda();
					totrainl = toinferm.clone().cuda();
					totrainl = torch::ones(toinferm.sizes()).cuda();
					totrainll = torch::ones(toinferm.sizes()).cuda();
					totrainlw = torch::ones(toinferm.sizes()).cuda();
					if (std::filesystem::exists("totrainll.pt") && LOAD_SAVE) {
						torch::load(totrainll, "totrainll.pt");
					}

					if (std::filesystem::exists("totrainlw.pt") && LOAD_SAVE) {
						torch::load(totrainlw, "totrainlw.pt");
					}

					if (std::filesystem::exists("totrainlm.pt") && LOAD_SAVE) {
						torch::load(totrainlm, "totrainlm.pt");
					}

					if (std::filesystem::exists("tolrnl52m.pt") && LOAD_SAVE) {
						torch::load(tolrnl52m, "tolrnl52m.pt");
					}

					totrainl = (totrainlw - totrainll).abs().clone().detach();
					//toinferm2 = toinferm.clone().cuda();
					//toinferm2.zero_();
					toinferm3 = torch::zeros({ N_MODES, 20, 20 }).cuda();//toinferm.clone().cuda();
					toinferm3mask = torch::zeros({ N_MODES, 20, 20 }).cuda();
					toinferm3r = torch::zeros({ N_MODES, 20, 20 }).cuda();
					toinferm4 = torch::zeros({ N_MODES, 20, 20 }).cuda();
					//toinferm3.zero_();
					tolrnl = toinfer.clone().cuda();//torch::ones({ 20, 20 }).cuda();//tolrn.clone().cuda();
					tolrnl2 = torch::zeros({ 1, 20, 20 }).cuda();//toinferm.clone().cuda();
					tolrnl2m = torch::tensor({}).cuda();
					toinferlost = toinfer.clone().cuda();
					toinferwon = toinfer.clone().cuda();
					toinferlst = toinfero.clone().cuda();
					//toinfer = torch::tensor({ 0 }).cuda();;
					//tolrnl = torch::tensor({ 0 }).cuda();;
					//dogeneratesmpls(20, nullptr, &tolrnl);
					//tolrn[0][0] = 0.1;
#if 0
					{
						unsigned long long rnd;
						torch::Generator gen = at::detail::createCPUGenerator();
						//gen.get();
						getrng(rnd);
						gen.set_current_seed(rnd);
						for (auto i = 0; i < 20; ++i) {
							auto rndperm = torch::randperm(20, gen).cuda();
							toinfer[i] = toinfer[i].index_select(0, rndperm);
							tolrn[i] = tolrn[i].index_select(0, rndperm);
						}
					}
#endif

					if (std::filesystem::exists("toinfer.pt") && LOAD_SAVE_IO) {
						torch::load(toinferm, "toinfer.pt");
						torch::load(toinfero, "toinfero.pt");
						//totrain = totrain.cuda();
						//torch::load(totrainto, "totrain.pt");
						//totrainto = totrain.clone().cuda();
						//toinfer = totrain.clone().cuda();
						//fwdhlb[0] = lstrnnresh.clone();
					}

					if (std::filesystem::exists("resss.bin") && LOAD_SAVE_PREDS) {
						std::ifstream resss{ "resss.bin", std::ios::binary | std::ios::in };
						for (char& it : preds)
							resss.read(&it, 1);
						//totrain = totrain.cuda();
						//torch::load(totrainto, "totrain.pt");
						//totrainto = totrain.clone().cuda();
						//toinfer = totrain.clone().cuda();
						//fwdhlb[0] = lstrnnresh.clone();
					}

					//torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(c10::ScalarType::Half));

					testl = &testt;
#if 0
					at::autocast::set_autocast_enabled(torch::kCUDA, true);
					{
						torch::Tensor hlt = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
						(void)testt->forward(torch::rand({ 20 }).cuda(), hlt, false, 0);
					}
#endif
					//test2.to(device);

					//out.save_to("mdl_bs.mdl");

					//std::cout << test->forward(torch::rand({ 20 }).cuda()) << std::endl;

					//torch::optim::SGD optim(test->parameters(), torch::optim::SGDOptions{ 0.0001 });

					torch::Tensor lstinp = torch::zeros({ 24 }).cuda();
					torch::Tensor lstinp2 = torch::ones({ 20 }).cuda();
					torch::Tensor lstinpr = torch::ones({ 20 }).cuda();
					torch::Tensor lstinpri = torch::ones({ 20 }).cuda();
					torch::Tensor lstinpalt = torch::ones({ 20 }).cuda();
					torch::Tensor lstinptotrain = torch::ones({ 20 }).cuda();

					ComPtr<ID3D12Resource> m_vertexBuffer;
					ComPtr<ID3D12Resource> m_indexBuffer;

					static int ind = 0, ind1 = 0;

					ID3D12CommandList* ppCommandLists[1];

					int trtbal = 0;
					//bool lstres = false;

					bool end = false;
					int itersend = 0;

					int testi = 0;
					std::atomic_int64_t iter = 1;
					std::atomic_int64_t curiter = 1;
					bool inited = false;
					int lstresi = -1;

					std::atomic_int iiters = 0;

					std::atomic<float> lossf;

					/*std::thread{[&] {
						torch::Tensor lstinp = torch::ones({ 10 }).cuda();
						torch::Tensor inpg, res, loss, resin;
						unsigned long long rnd, rndtest;
						std::atomic_bool donernd = false;
						auto lmbdseed = [&] {
							//while (!_rdrand64_step(&rnd));
							inpg = torch::empty({ 20 }).cuda();
							for (int i = 0; i < 20; ++i) {
								unsigned long long rndval;
								while (!_rdseed64_step(&rndval));
								inpg[i] = (double)rndval / UINT64_MAX;
							}
							donernd = true;
						};
						(std::thread{ lmbdseed }).detach();
						while (!donernd) Sleep(0);
						while (true) {
							//Sleep(4000);
							//if (!end) {
							torch::Tensor inp = inpg;
							donernd = false;
							(std::thread{ lmbdseed }).detach();
							while(!donernd) {
								optim.zero_grad();
								res = test->forward(inp);
								//resin = res.detach().clone();
								loss = torch::smooth_l1_loss(res, lstinp, at::Reduction::Sum);
								lossf = loss.item().toFloat();

								//std::cout << "iter: " << iter << std::endl;
								//std::cout << "testi: " << testi << std::endl;
								//std::cout << res << std::endl;
								//std::cout << lstinp << std::endl;
								//auto max = torch::argmax(lstinp).item().toInt();
								//lstinp = torch::zeros({ 10 });
								//lstinp[max] = 1.0;
								//lstinp = lstinp.cuda();
								//lstres = max > 4;
								std::cout << loss << std::endl;
								//if (true) {//maxpred != 9) {//(iter < 100) && predtrue) {

								//if (!(maxpred > 4)) {
										/*test->~decltype(test)();
										new (&test) Net();
										test->to(device);
										optim.~decltype(optim)();
										new (&optim)torch::optim::Adam(test->parameters(), torch::optim::AdamOptions{ (loss.item().toFloat() * 0.000001) }.weight_decay(0.0).amsgrad(true));
										* /
										//if (resi) {
								loss.backward();
								optim.step();
								iiters += 1;
							}
							lstinp = inp.detach().slice(0, 0, 10);
						}
						} }.detach();*/
					float mul = 1.0;
					static double lr = 0.0000002;//1. / 500000000; /// 500000000.;
					static double wd = 0.0002;//0.0000002 //0.0002
					lr = wd;
					lrf = wd;
					torch::Tensor loss = torch::empty({ 1 }).cuda();
					bool donetrain = false;
					torch::autograd::variable_list grads;
					//torch::autograd::variable_list losses;

					std::unique_ptr<torch::optim::Optimizer> optim[4]{};
					std::function<torch::optim::Optimizer* ()> factories[] = {
						[&] {
							//auto optim = new torch::optim::LBFGS1(testtb->parameters(), torch::optim::LBFGS1Options{ 0.0000002}.max_iter(100));
							return nullptr;
							},
							[&] {
							return new torch::optim::NAdam(test2->param_groups);// new torch::optim::SGD(testtb->parameters(), torch::optim::SGDOptions{wd});//torch::optim::SGD(testtb->parameters(), torch::optim::SGDOptions{wd});//torch::optim::Adagrad(testtb->parameters(), torch::optim::AdagradOptions{wd});//NAdam(test->parameters(), torch::optim::NAdamOptions{1.}.weight_decay(1.));
							},
							[&] {
							return new tgt_optim_t(test3->parameters(), tgt_optim_opts_t(0.));//nullptr;//new torch::optim::NAdam(testtb->parameters(), torch::optim::NAdamOptions{lr}.weight_decay(wd));
							},
							[&] {
							return new tgt_optim_t(test3->parameters(), tgt_optim_opts_t(0.1));
							//auto optim = new torch::optim::LBFGS(testtb->parameters(), torch::optim::LBFGSOptions{wd}.max_iter(200));
							//return optim;
							},
					};

					optim[0] = std::unique_ptr<torch::optim::Optimizer>(factories[0]());
					optim[1] = std::unique_ptr<torch::optim::Optimizer>(factories[1]());
					optim[2] = std::unique_ptr<torch::optim::Optimizer>(factories[2]());
					optim[3] = std::unique_ptr<torch::optim::Optimizer>(factories[3]());

					//dynamic_cast<torch::optim::AdamWOptions&>(optim[2]->param_groups()[0].options()).weight_decay(0.).lr(0.001);

					//torch::optim::ReduceLROnPlateauScheduler shed = torch::optim::ReduceLROnPlateauScheduler(*optim[1]);



					//optim[1]->add_param_group(test->parameters());
					//optim[1]->
					//optim[2] = std::unique_ptr<torch::optim::Optimizer>(factories[2]());

					/*std::unique_ptr<torch::optim::Optimizer> optim2[2]{};
					std::function<torch::optim::Optimizer* ()> factories2[] = {
						[&] {
						return new torch::optim::SGD(test2.parameters(), torch::optim::SGDOptions{ lr }.weight_decay(wd));
						},
						[&] {
						return new torch::optim::NAdam(test2.parameters(), torch::optim::NAdamOptions{lr}.weight_decay(wd));
						}
					};
					optim2[0] = std::unique_ptr<torch::optim::Optimizer>(factories2[0]());
					optim2[1] = std::unique_ptr<torch::optim::Optimizer>(factories2[1]());*/

					int optimi = 1;

					//if (std::filesystem::exists("opt_bs.pt") && LOAD_SAVE) {
						//torch::serialize::InputArchive in{};//::OutputArchive out{};

						//in.load_from("opt_bs.pt");
						//optim[0]->load(in);
				//		torch::load(optim[0]->parameters(), "opt_bs.pt");
						//optim[0]->add_parameters(testtb->parameters());
				//	}
					//optim[0]->add_parameters(testtb->parameters());

					//torch::optim::RMSprop optim2(test2.parameters(), torch::optim::RMSpropOptions{ lr }.eps(0.000000000000000000000001).weight_decay(wd));
					float yieldav = 0.;
					//std::vector<torch::
					bool dir = true;
					bool first = true;
					int resicur = -1;
					float lstlss = FLT_MAX;
					float curlss = 0.;
					float intlss = 0.;
					float lssdiff = 0.0;
					bool lstres = false;
					std::atomic_bool condt = false, condt1 = false, condt2 = false, condt2done = false, condtl = false;
					std::condition_variable condtlv;
					std::mutex condtlm;

					bool pseudoswitchl = false;
					bool switchswitchl = false;

					bool condswap = false, resmcondswap = false;
					float resm = 0.0;
					double condtlst = NAN;
					int itersb = 0;
					uint64_t itersy = 1;
					uint64_t itersx = 0;
					std::atomic_int64_t itersz = 0;
					uint64_t iterszlst = 0;
					uint64_t distmbl = 0;
					uint64_t lstseed = 1;
					bool lstabove = false;

					static torch::Tensor totrainback = torch::tensor({ 0 }).cuda();
					static torch::Tensor restotrain = torch::tensor({ 0 }).cuda();
					static torch::Tensor totrain1 = torch::tensor({ 0 }).cuda();
					static torch::Tensor totrain2 = torch::tensor({ 0 }).cuda();


					static torch::Tensor totraintoalt = torch::tensor({ 0 }).cuda();
					static torch::Tensor totrainalt = torch::tensor({ 0 }).cuda();
					//torch::Tensor totrain2 = torch::tensor({ 0 }).cuda();
					//torch::Tensor totrainto2 = torch::tensor({ 0 }).cuda();
					bool train10 = false;
					uint64_t ntrain = 0;
					static int iters = 0;
					static std::atomic_bool lost_mode = false;
					static torch::Tensor trainhl = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
					static torch::Tensor trainhln = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
					static torch::Tensor trainhl2 = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
					torch::Tensor ntrainhl = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
					torch::Tensor fwdhl = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
					torch::Tensor fwdhlrn = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
					torch::Tensor lstgood = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
					torch::Tensor hl = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
					torch::Tensor hlb = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
					torch::Tensor hlrn = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
					const torch::Tensor hlzero = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();

					lstrnntrainh = torch::zeros({ 2, 2 * 6, 20, 20 }).cuda();
					torch::Tensor trainhlb = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();
					//if (std::filesystem::exists("rnnresh.pt") && LOAD_SAVE) {
					//	torch::load(testtb->layers[0].rnnresh, "rnnresh.pt");
					//}
					//std::stringstream ss;
					//ss << " " << std::hex << testtb->layers[0].rnnresh << std::endl;
				//ss << "triggered at: " << res[2] << std::endl;
				//ss << "pred is: " << (res[0] > res[1]) << std::endl;

					//orig_buf->sputn(ss.str().c_str(), ss.str().size());
					savestuff = [&](bool sv, const torch::nn::Module& net, const char* name) {//itersx > 4) {
						//while (requests) Sleep(0);
						//dobet(lstinpsum1res > lstinpsum2res, 0.00016384);
						//if (ntrain == 0)
						//	goto deleting_end;
						std::ofstream ifb("bal.txt");
						ifb << (orbal + rrbal) / 100000000. << std::endl;
						ifb << std::max(ormaxbal, rmaxbal.load()) / 100000000. << std::endl;
						ifb << std::min(orminbal, rminbal.load()) / 100000000. << std::endl;
						ifb << std::max(orminbalahead, (std::abs(rminbal.load()) / 100000000.) / (std::max(orbal, orbal + rrbal) / 100000000.)) << std::endl;
						ifb << +nonce << std::endl;
						ifb << serverSeed << std::endl;
						ifb << clientSeed << std::endl;
						ifb << serverSeedlspec << std::endl;
						ifb << globaliters << std::endl;
						ifb << rolllosti << std::endl;
						//ifb << std::hex << in32srvmseed << std::endl;
						//optimlk.lock();
#if 0
						mdllk.lock();
						if (sv) {
							torch::serialize::OutputArchive out{}, o1{};

							//test->to(devicecpu);

							net.save(out);
							std::string mdlname = !name ? "mdl_bsa" : name;
							mdlname += ".pt";//std::to_string(i) + ".pt";
							out.save_to(mdlname);
							torch::save(fwdhlbl, "lstrnnresh.pt");
							torch::save(toinferm, "toinfer.pt");
							torch::save(toinfero, "toinfero.pt");
							//torch::save(totrainto, "totrainto.pt");
							//optim[0]->zero_grad();
							//torch::save(optim[0]->parameters(), "opt_bs.pt");
							//optim[0]->save(o1);

							//o1.save_to("opt_bs.pt");
							//torch::save(testtb->layers[0].rnnresh, "rnnresh.pt");
						}
						mdllk.unlock();
						std::ofstream resss{ "resss.bin", std::ios::trunc | std::ios::binary | std::ios::out };
						for (char it : preds)
							resss.write(&it, 1);
						//optimlk.unlock();
						//std::terminate();
#endif
						return;
						//itersx = 0;
						//condt1 = false;
						};

					unsigned long cnt = 0;
					bool alts = false;
					//if (!SetConsoleCtrlHandler(consoleHandler, TRUE)) {
					//	printf("\nERROR: Could not set control handler");
					//	return 1;
					//}
					torch::Tensor inp, res = torch::ones({ 20 }).cuda(), res1 = torch::ones({ 4 }).cuda(), resin, inptotrain, refinp, resref, inp2;
					//savestuff(true);
					inp = torch::zeros({ 20 }).cuda();
					inp2 = torch::zeros({ 20 }).cuda();
					//inp = inp.toType(c10::ScalarType::Double);
					while (true) try {
						//testt->reset(0);
						//for (int x = 0; x < 40; ++x)

						//condt2 = false;//((maxbal - bal) == 0 && bal > 0);// && itersz > 500;// && initialloss < 0.5;
#if 1
						const UINT64 fence = m_fenceValue;
						m_commandQueue->Signal(m_fence.Get(), fence);

						m_fenceValue++;

						// Wait until the previous frame is finished.
						if (m_fence->GetCompletedValue() < fence)
						{
							m_fence->SetEventOnCompletion(fence, m_fenceEvent);
							WaitForSingleObject(m_fenceEvent, INFINITE);
						}
#endif

						auto gensrsstr = [] {
							unsigned long long rnd1, rndtest1, rnd, rndtest;
							getrng(rnd);
							getrng(rndtest);
							getrng(rnd1);
							getrng(rndtest1);
							std::stringstream ss;
							ss << std::setfill('0') << std::setw(64) << std::hex << rnd << rndtest <<
								rnd1 << rndtest1;
							auto str = ss.str();
							str.resize(64);
							return str;
							};

						auto gensrclstr = [] {
							unsigned long long rnd, rndtest;
							getrng(rnd);
							getrng(rndtest);
							std::stringstream ss;
							ss << std::setfill('0') << std::setw(30) << std::hex << rnd << rndtest;
							auto str = ss.str();
							str.resize(30);
							return str;
							};



						auto dobetl2 = [&]() {
							static torch::Tensor loss2 = torch::empty({});
							static int itesrtrain = 0, itesrtraing = 0;
							static float lss2min;
							auto wcopy = [](torch::Tensor& w, const torch::Tensor& w1, double decay) {
								//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
								//w.set_data(w1.data());
								//w.set_data(w1.data());
								//auto wdator = w.data().unsqueeze(0).clone().cuda();
								//*lstarriter = torch::vstack({ *lstarriter, w.data().clone().detach().cuda().unsqueeze(0) });
								w.set_data(w1.data().clone().detach().cuda());
								//auto wdator = w.data().unsqueeze(0).clone().cuda();
								//if (vbal2 > lstvbal2)
								//if (lstarriter->size(0) >= 20)
								//*lstarriter = wdator;
								//lstarriter += 1;
								//*lstarriter = lstarriter->resize_({ 1, -1 });
								//w.set_data(w.data() / torch::mean(w.data()));
								//w.set_data(torch::pow(w.data() * (2. / w.data().norm()), (itesr * 2)));
								//torch::nn::LayerNorm()()
								//w.set_data(w.data() * ((10 ) / w.data().norm()));
								//w.set_data(w.data().fmod(w1.data()));

								//if (!dirwlb)
								//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
								//if (shrink)
								//w.set_data(w.data().uniform_(-1, 1));
								//w.clip
								//std::stringstream ss;
					//ss << "rrbal: " << rrbal << std::endl;
					//ss << "triggered at: " << res1[16] << std::endl;
					//ss << torch::mean(w) << std::endl;
								//gr3 = (torch::mean(w) >= 3.0).item().toBool();
								//w.set_data(-w.data());

					//orig_buf->sputn(ss.str().c_str(), ss.str().size());
								};

							dobetr = 1;
							//static Net test2;
							//}
							//if (std::filesystem::exists("totrainll.pt"))
							//	torch::load(totrainll, "totrainll.pt");
							//if (std::filesystem::exists("totrainlw.pt"))
							//	torch::load(totrainlw, "totrainlw.pt");
							//if (std::filesystem::exists("test2.pt")) {
							//	torch::load(test2, "test2.pt");
							if (std::filesystem::exists("test2.pt"))
								torch::load(test2, "test2.pt");
							if (std::filesystem::exists("fwdhlbl2.pt"))
								torch::load(fwdhlbl2, "fwdhlbl2.pt");
							if (std::filesystem::exists("totrainlm.pt") && std::filesystem::exists("tolrnl52m.pt")) {
								torch::load(totrainlm, "totrainlm.pt");
								torch::load(tolrnl52m, "tolrnl52m.pt");
								dobetr = false;
								//btrain = true;
							}
							//}
							//if (std::filesystem::exists("fwdhlblout.pt")) {
							//	torch::load(fwdhlblout, "fwdhlblout.pt");
							///}
							//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
							//fwdhlbl.copy_(fwdhlbl2.contiguous().detach());
							//totrainl = totrainlw.clone().detach();
							totrainl.zero_();
							totrainl[0][0][0] = 1.;
							//totrainlm[0][0][0] = 1.;
							static int rightiw = 0;
							static int leftiw = -1;
							static int rightil = 0;
							static int leftil = -1;
							static int xorab = 0;
							//if (std::filesystem::exists("totrainlm.pt")) {
							//	if (std::filesystem::exists("tolrnl52m.pt") && std::filesystem::exists("itesr.bin")) {
							//		torch::load(totrainlm, "totrainlm.pt");
							//		torch::load(tolrnl52m, "tolrnl52m.pt");

							//		std::ifstream resss{ "itesr.bin", std::ios::binary | std::ios::in };
							//		resss.read((char*)&righti, 4);
							//		resss.read((char*)&lefti, 4);
							//
							//		dobetr = 0;
							//	}
							//}
							torch::nn::init::xavier_uniform_(fwdhlbl2);
							//totrainllst[0][0][0] = 1.;
							if (itesr == 99, vbal2pa > 0, ((vbal2s[-2][0] > vbal2s[-3][0]).item().toBool()), 1) {
								for (int i = 0; i < 1; ++i) {
									float vbaldist = std::abs(vbalmin) + std::abs(vbalmax);
									float vbalceofcur = vbal + std::abs(vbalmin);
									//, (vbal2 > lstvbal2) ? 0. : 1.
									vbalceofcur /= vbaldist;
									//if (rrbalmaxv != INT64_MIN)
									//vbalceofcur = i == 0 ? 1. : 0.;
									test2->train();
									//fwdhlbl2 = fwdhlbl.clone().detach();//.zero_();
									//torch::nn::init::xavier_uniform_(fwdhlbl);
									float lssdif = 0.;
									float lstlssdif = 0.;
									float lssdiftrgt = 1e-5;
									bool zrgr = true;
									bool btrain = false;
									bool needregen = true;
									torch::Tensor rfgrid = torch::zeros({ 1, 20, 20 }).cuda(), rfgridlst = torch::zeros({ 1, 20, 20 }).cuda();
									bool optsw = false;
									betsitesrmade = 0;

									//exit(0);
									//float f;//torch::Tensor fkgr = torch::empty({ test2->nels });
									//torch::nn::init::xavier_uniform_(fkgr);
									//Net smpl;// = Net(*dynamic_cast<NetImpl*>(test2->clone().get()));
									//clonemodelto(test2.get(), smpl.get());
									//test2 = Net(*dynamic_cast<NetImpl*>(smpl->clone().get()));
									//test2->layers[0].rnn1->flatten_parameters();
									//test2->calc_nels();
									runlr = 1e-6;//0.05;//0.00000166666 * loss2.item().toFloat();
									runlr2 = 0.0005;
									runlr3 = 0.0005;
									//totrainl[0][0][0] = genranf();
									smpl->train();
									//torch::nn::init::xavier_uniform_(totrainllst);
									auto sctpt = torch::ScalarType::Float;
									torch::Tensor x = torch::zeros({ test2->nels }).to(device).to(dtype(sctpt));//test2->gather_flat_grad().to(device);//torch::zeros({ N_elements }).to(optionst);
									torch::Tensor g = torch::zeros({ test2->nels }).to(device).to(dtype(sctpt));//test2->gather_flat_grad().to(device);//torch::zeros({ test2->nels }).to(device);
									torch::Tensor xl = torch::zeros({ test2->nels }).to(device).to(dtype(sctpt));
									torch::Tensor xu = torch::zeros({ test2->nels }).to(device).to(dtype(sctpt));
									torch::Tensor nbd = torch::zeros({ test2->nels }).to(dtype(torch::ScalarType::Int)).to(device);
									do {
										static std::mutex trainm;
										std::unique_lock lk(trainm);
										static bool sttteed = 0;
										static int sttteediters = 0;
										static bool sttteedorig = false;
										static bool lststalled = false;
										static bool minned = false;
										static bool xavsetted = false;
										static int nonminned = false;
										static bool stalled = false;
										bool lststtteed = sttteed;
										bool lststc = false;
										static bool fsw = true;
										static bool wasminned = false;
										static int waschanged = 0;
										static bool wassw = false;
										static int currmdl = 0;
										static bool orand = false;
										//bool btrain = false;
										int ierstr = 0;
										//torch::nn::init::xavier_uniform_(fwdhlbl2);
										//runlr = 0.05;
										//if (zrgr)
										//if (!sttteed) {
											//test2->zero_grad();
										if (itesrtrain == 0) {
											sttteed = false;
											//runlr = 1e-6;
											//runlr2 = 1e-6;
											optsw = false;
											wasminned = false;
											waschanged = 0;
											//runlr = 0.01;
											//runlr2 = 1.;
											//torch::nn::init::xavier_uniform_(fwdhlbl2);
											lss2min = FLT_MAX;
											losslstg = FLT_MAX;
											losslstgr = FLT_MAX;
											losslstgor = FLT_MAX;
											lststalled = false;
											minned = false;
											stalled = false;
											xavsetted = false;
											nonminned = 0;
											sttteediters = 0;
											lssdiftrgt = 1e-5;
											fsw = true;
											optim[1]->param_groups()[0].options().set_lr(runlr);
											dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[0].options()).weight_decay(runlr * 100.);
											optim[1]->param_groups()[1].options().set_lr(runlr2);
											dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[1].options()).weight_decay(runlr2 * 100.);
											optim[1]->param_groups()[2].options().set_lr(runlr3);
											dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[2].options()).weight_decay(runlr3 * 100.);
											//shed.~ReduceLROnPlateauScheduler();
											//new (&shed) torch::optim::ReduceLROnPlateauScheduler(*optim[1]);
										}


										//}
										if (dobetr) {
											if (betsitesrmade == 0) {
												//if (betsitesrmade400g % 20 == 0) {
												torch::save(test2, "test2.pt");
												//std::exit(0);
												//abvsgrids.zero_();
												//abvsgridsvals.zero_();
												//torch::nn::init::xavier_uniform_(abvsgrids);
												//torch::nn::init::xavier_uniform_(abvsgridsvals);
												//abvsgrids = (abvsgrids > 0.).toType(c10::ScalarType::Float);
												//abvsgridsvals = abvsgridsvals.fmod(1.).abs();
												//torch::nn::init::xavier_uniform_(fwdhlbl2);
											//}
											}
											//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
											//torch::save(test2, "test2.pt");
											//torch::save(fwdhlblout, "fwdhlblout.pt");
											//fwdhlbl.copy_(fwdhlbl2.contiguous().detach());
											//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
#if 0
											test2->ema_update(0, *test2, !dirw ? 0.01 : -0.01, [](torch::Tensor& w, const torch::Tensor& w1, double decay) {
												//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
												//w.set_data(w.data().fmod(w1.data()));
												//w.set_data(w.data().fmod(w1.data()));
												w.set_data(w1.data() * (2. / w1.data().norm()));
												//if (!dirwlal)
												//	dirwlal = torch::any(w.data() > 2.).item().toBool() || torch::any(w.data() < -2.).item().toBool();
												//if (!dirwlal)
												//	dirwlal = torch::any(w.data() > 2.).item().toBool() || torch::any(w.data() < -2.).item().toBool();
												////if (!dirwlb)
												//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
												//if (shrink)
												//w.set_data(w.data().uniform_(-1, 1));
												//w.clip
												//std::stringstream ss;
									//ss << "rrbal: " << rrbal << std::endl;
									//ss << "triggered at: " << res1[16] << std::endl;
									//ss << torch::mean(w) << std::endl;
												//gr3 = (torch::mean(w) >= 3.0).item().toBool();
												//w.set_data(-w.data());

									//orig_buf->sputn(ss.str().c_str(), ss.str().size());
												}, [](torch::Tensor& w, const torch::Tensor& w1, double decay) {
													//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
													//w.set_data(w.data().fmod(w1.data()));
													//w.set_data(w.data().fmod(w1.data()));
													w.set_data(w1.data() * (2. / w1.data().norm()));
													//if (!dirwlal)
													//	dirwlal = torch::any(w.data() > 2.).item().toBool() || torch::any(w.data() < -2.).item().toBool();
													//if (!dirwlal)
													//	dirwlal = torch::any(w.data() > 2.).item().toBool() || torch::any(w.data() < -2.).item().toBool();
													//if (!dirwlb)
													//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
													//w.set_data(-w.data());
													//if (shrink)
													//w.set
													// _data(w.data() - (0.001 * (1) * (1)) * w.data());
													});
#endif
												//if (!sttteed) {

												//}
												sttteed = false;
												//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
												//if (totrainlm.size(0) > 10) {
													//std::exit(0);
													//totrainlm = totrainlm[-1].unsqueeze(0);
													//tolrnl52m = tolrnl52m[-1].unsqueeze(0);
													//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
													//fwdhlbl.copy_(fwdhlblout.contiguous().detach());
												//}


												auto indn = (totrainl.flatten().argmax().item().toInt() + 0) % 400;
												volatile bool wasab = reswillwino.defined() ? (torch::sigmoid(reswillwino)[0][indn] > 0.5).item().toBool() : 0;

												bool actualpred = wasab;
												//vbal2 = 0;
												float coef = reswillwino.defined() ? (torch::sigmoid(reswillwino)[0][indn]).item().toFloat() : 0.5;
												int numtrgt = std::max(200, std::min((int)(10000 * coef), 9800));
												int numtrgtprob = (wasab ? (10000 - numtrgt) : numtrgt);

												float mul = ((float)10000 / numtrgtprob) * (99. / 100.);
												volatile float ch = ((float)numtrgtprob / 10000) * 100.;
												trainedb = 0;// reswillwino.defined() ? (torch::sigmoid(reswillwino)[0][indn].abs() >= 0.9).item().toBool() : 0;// && betsitesrmade == 399; //&& (ierstr > 0);//betsitesrmade > 20;// || wasnotab;//(torch::min(reswillwino) < 0.5).item().toBool();
												prabs[modes] = (float)actualpred;
												//if (trainedb) {
												//	fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
												//}
												int noncel;
												std::string serverSeedl, clientSeedl;

												{
													std::unique_lock lkn{ incnonce };
													noncel = nonce;
													nonce += 1;
													serverSeedl = serverSeed;
													clientSeedl = clientSeed;

												}

												volatile int numres, resir, fresir;

												
												volatile float betamntfl = trainedb ? (1. / 100.) * ((double)(orbal + rrbal) / 100000000.) : DEF_BET_AMNTF;
												//if (!trainedb) {
												//	actualpred = false;//reswillwino1.defined() ? (reswillwino1[0][indn] > 0.5).item().toBool() : 0;
												//}
												if (REAL_BAL) {
													fresir = dobet(wasab, betamntfl, ch, numres);
													resir = !((numres > 4999) == !!actualpred);
												}
												else {
													numres = getRoll(serverSeedl, clientSeedl, noncel);
													resir = !((numres > 4999) == !!actualpred);
													fresir = wasab ? !((numres > numtrgt)) : !((numres < numtrgt));
												}


												
												//betsitesrmade400g % 2 == 1 ? resir : !resir;
												if (!fresir && reswillwino.defined()) {
													itesrwin += 1;
													acccoef += reswillwino[0][indn].item().toFloat();
												}
												if (trainedb) {
													rrbalv += ((actualdir4 ? !fresir : fresir) ? 1 : -1);
													rrbalmaxv = std::max(rrbalmaxv, rrbalv).load();
													rrbalminv = std::min(rrbalminv, rrbalv).load();

													//runlr = 0.005;
												}
												//vbal2 += (!resir ? 1 : -1);
												
												//if (reswillwino.defined())
												//	betamntfl *= reswillwino[0][indn].item().toFloat();
												//else
												//	betamntfl = DEF_BET_AMNTF;
												float avret = !fresir ? mul - 1. : -1.;
												rrbal += betamntfl * 100000000 * avret;
												//rbal += !resi ? betamntf * 100000000 : -(betamntf * 100000000);
												rmaxbal = std::max(rrbal, rmaxbal).load();
												rminbal = std::min(rrbal, rminbal).load();

												vbal2 += avret;

												vbal += (!fresir ? 1 : -1);


												savestuff(false, *testt, 0);

												//if (rrbalv > 0) {
												//	TerminateProcess(GetCurrentProcess(), 0);
												//}

												bool predright = !resir ? actualpred : !actualpred;

												if (predright) {
													tolrnl52[0] = 1.;//98;
													//tolrnl52[1] = 0.02;
												}
												else {
													tolrnl52[0] = 0.0;
													//tolrnl52[1] = 0.98;
												}

												bool needchg = false;



												betsitesrmade += 1;

												/*bool haswon = 1.;
												auto& totrainlref = (haswon ? totrainlw : totrainll);
												//totrainlref[0][0][0] = (float)predright;
												//totrainl = totrainlref.clone().detach();//(totrainlw - totrainll).abs().clone().detach();//totrainlref.clone().detach();
												static int &righti = (haswon ? rightiw : rightil);
												static int &lefti = (haswon ? leftiw : leftil);
												if (predright) {
													totrainlref[0].flatten()[righti] = (totrainlref[0].flatten()[righti] - 1.).abs();
													righti += 1;
													righti %= 400;
												}
												else {
													totrainlref[0].flatten()[lefti] = (totrainlref[0].flatten()[lefti] - 1.).abs();
													lefti %= 400;
													lefti -= 1;
												}*/
												//totrainl = (totrainlw - totrainll).abs().clone().detach();
												//totrainl[0][0][0] = float(predright);
												//totrainl = torch::roll(totrainl, 1);
												//totrainl = totrainlref.clone().detach();
												//for (int x = 0; x < 20; ++x)
												//	for (int y = 0; y < 20; ++y)
												//		totrainl[0][x][y] = genranf();
												//totrainlref = torch::roll(totrainlref, -1);
												//totrainllst = totrainl.clone().detach();
												//totrainl = torch::roll(totrainl, 1);
												//tolrnll2 = abvsgrid.clone().detach();
												for (int y = 0; y < 1; ++y) {
													//if (needregen) {
													//	auto rf = genranf();
													//	rfgrid[0].flatten()[indn] = rf;
													//}
													///if ((abvsgridslst[0].flatten()[indn] > 0.5).item().toBool() == predright) {
													///	tolrnll2[y] += totrainl[0];// * rf;
														//abvsgrids[y].fmod_(2.);
														//if ((abvsgrids[y].flatten()[indn] >= 2.).item().toBool()) {
														//	abvsgrids[y].flatten()[indn] = 0.;
														//}
													//}
													//if (resir != (rf > 0.5))
													//	tolrnll2 += totrainl;
												}
												rfgrid[0].flatten()[indn] = float(predright);
												//if (!resir)
												//	tolrnll2[0].flatten()[indn] *= 2.;
												//else
												//	tolrnll2[0].flatten()[indn] /= 2.;
												//	tolrnll2 = tolrnll2.toType(c10::ScalarType::Bool).bitwise_xor(totrainl.clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);
												//(reswillwino.defined() ? totrainl * reswillwino.detach().reshape_as(tolrnll2) : totrainl);
												//else
												//	tolrnll2 += (reswillwino.defined() ? -totrainl * reswillwino.detach().reshape_as(tolrnll2) : -totrainl);
												//tolrnll2 = (abvsgrid.clone().detach() > 0.).toType(c10::ScalarType::Float);
												totrainl = torch::roll(totrainl, 1);
												//totrainllst = totrainl.clone().detach();
												//tolrnll2 = abvsgrid.clone().detach();
												bool aboveres = wasab;
												//torch::save(totrainll, "totrainll.pt");
												//torch::save(totrainlw, "totrainlw.pt");
												//torch::save(totrainlm[-1].unsqueeze(0), "totrainlm.pt");
												//torch::save(tolrnl52m[-1].unsqueeze(0), "tolrnl52m.pt");
												//std::ofstream resss{ "itesr.bin", std::ios::trunc | std::ios::binary | std::ios::out };
												//resss.write((char*)&righti, 4);
												//resss.write((char*)&lefti, 4);
												//if (dobetr) {
												//tolrnll2[0].flatten()[indn] = xorab & predright;//abvsgrids.toType(c10::ScalarType::Bool).bitwise_and(rfgrid.clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);
												/*auto div = ((vbal2 + 400.) / 800.);
												auto div1 = ((lstvbal2 + 400.) / 800.);
												if (div > div1) {
													std::swap(div, div1);
													auto ar = 1. - (div1 != 0. ? div / div1 : 1.);
													tolrnll2 = totrainllst.clone().detach() * ar;
												}
												else {
													auto ar = (div1 != 0. ? div / div1 : 1.);
													tolrnll2 = totrainllst.clone().detach() * ar;
												}*/
												//xorab ^= predright;
												//abvsgrids[0].flatten()[indn] = xorab;
												if (betsitesrmade == 400) {
													btrain = totrainlm.defined();//(betsitesrmade == 400);//vbal2 < 0;
													dobetr = !btrain;//!btrain;
													needregen = vbal2 < 0;
													tolrnll2 = abvsgrids.toType(c10::ScalarType::Bool).bitwise_and(rfgrid.clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);
													if (vbal2 < lstvbal2)
														fwdhlbl2.copy_(fwdhlblout.contiguous().clone());
													if (tolrnl52m.size(0) < 2, 1) {
														if (betsitesrmade > 0, tolrnl52m.defined(), 0)
															tolrnl52m = torch::vstack({ tolrnl52m, tolrnll2 }).cuda();
														else {
															//needchg = !(tolrnl52m.size(0) <= 100);
															tolrnl52m = tolrnll2.clone().detach();

														}
													}
													else {
														tolrnl52m = torch::roll(tolrnl52m, -1, 0);
														tolrnl52m[-1] = tolrnll2[0];
													}
													lstvbal2 = +vbal2;
													vbal2 = 0;
													betsitesrmade400g += 1;
												}


												//loss2 = //optimi == 0 ? torch::smooth_l1_loss(res, lstinp, mean ? at::Reduction::Mean : at::Reduction::Sum) :
													//(torch::binary_cross_entropy(res.slice(0, 2, 4), lstinp.slice(0, 2, 4)) * 0.8 
												//		+ 0.2 * torch::smooth_l1_loss(res.slice(0, 0, 2), lstinp.slice(0, 0, 2), mean ? at::Reduction::Mean : at::Reduction::Sum));
													//mean ? torch::soft_margin_loss(res, lstinp, mean ? at::Reduction::Mean : at::Reduction::Sum) :
													//(dobetr ? torch::binary_cross_entropy(resallo, ((tolrnl2).toType(c10::ScalarType::Float)).clone().detach().cuda(), {}, at::Reduction::Mean) :
														//torch::binary_cross_entropy(resallo, ((tolrnl4).toType(c10::ScalarType::Float)).clone().detach().cuda(), {}, at::Reduction::Mean));//.reshape({ 1 });
												//	torch::binary_cross_entropy(reswillwinpr.squeeze(0).flatten().slice(-1, 0, indn + 1), ((tolrnll2.flatten().slice(-1, 0, indn + 1).detach().toType(c10::ScalarType::Half))));
												//lss2min = loss2.item().toFloat();
												//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
												//torch::save(test2, "test2.pt");
												//torch::nn::init::xavier_uniform_(fwdhlbl2);
												//torch::save(fwdhlblout, "fwdhlblout.pt");
												//fwdhlbl.copy_(fwdhlbl2.contiguous().detach());

												//}

										}
										if (btrain) {
											//auto& runlr = 1 ? ::runlr2 : ::runlr;
											auto& trgtcrl = stalled;
											test2->train();

											//{
											//torch::nn::init::xavier_uniform_(fwdhlbl);
											//opt.weight_decay(1.);
											//optim[2]->zero_grad();
											//opt.lr(0.001);
											//opt.max_iter(30);

											//else
											//	vbalceofcur = 0.2;
											//vbalceofcur = vbal2 > 0 ? 0.1 : 0.9;//(float)(lstvbal2 + 20) / (float)(vbal2 + 20) ;//std::max(std::min(vbalceofcur / vbaldist, 0.7f), 0.3f);
											//std::powf(0.7, vbalceofcur * 20));
											//opt.tolerance_grad(1e-15);
											//opt.tolerance_change(1e-15);
											//optim[2]->zero_grad();

											//loss2.backward();
											int localiters = 0;

											auto cb = [&](torch::Tensor fwdhlblin, torch::Tensor* fwdhlblout, Net2 mdl) {

												//if (zrgr)
												//if (!sttteed)
												test2->zero_grad();

												//tolrnl52r = tolrnl52.clone().detach();
												//else if (true) {
												//	tolrnl2m = torch::roll(tolrnl2m, -1, 0);
												//	tolrnl2m[-1] = tolrnl2.clone().detach();
												//}
												//if ((totrainlr != totrain).item().toBool())
												auto indn = 400;//(totrainllst.flatten().argmax().item().toInt() + 1);
											beg:
												auto [resall, reswillwin] = mdl->forward(totrainlm, abvsgridslst, rfgridlst, fwdhlblin, fwdhlblout, currmdl);
												reswillwinotr = reswillwin.squeeze(0);//.flatten(1);
												//totrainllst = reswillwinotr.clone().detach();
												//resall = resall.squeeze();
												//reswillwino = reswillwin.squeeze(0).flatten(1);

												if (!torch::all(reswillwinotr.isfinite()).item().toBool()) {
													//return torch::ones({1}).neg();
													//torch::load(test2, "test2.pt");
													std::exit(0);
													//clonemodelto(smpl.get(), test2.get());
													//test2->ema_update(0, *smpl, 0., wcopy, wcopy);
													//test2->layers[0].rnn1->flatten_parameters();
													test2->train();
													goto beg;
												}
												//std::stringstream ss;
												//ss << "fwdhlblin " << fwdhlblin << std::endl;
//
																							//orig_buf->sputn(ss.str().c_str(), ss.str().size());
																						//	}
												loss2 = //optimi == 0 ? torch::smooth_l1_loss(res, lstinp, mean ? at::Reduction::Mean : at::Reduction::Sum) :
													//(torch::binary_cross_entropy(res.slice(0, 2, 4), lstinp.slice(0, 2, 4)) * 0.8 
												//		+ 0.2 * torch::smooth_l1_loss(res.slice(0, 0, 2), lstinp.slice(0, 0, 2), mean ? at::Reduction::Mean : at::Reduction::Sum));
													//mean ? torch::soft_margin_loss(res, lstinp, mean ? at::Reduction::Mean : at::Reduction::Sum) :
													//(dobetr ? torch::binary_cross_entropy(resallo, ((tolrnl2).toType(c10::ScalarType::Float)).clone().detach().cuda(), {}, at::Reduction::Mean) :
														//torch::binary_cross_entropy(resallo, ((tolrnl4).toType(c10::ScalarType::Float)).clone().detach().cuda(), {}, at::Reduction::Mean));//.reshape({ 1 });
													//torch::binary_cross_entropy(reswillwinotr,
													//	((tolrnl52m.detach().toType(c10::ScalarType::Half)))
												//		);
													hybrid_loss(reswillwinotr, tolrnl52m.detach().toType(c10::ScalarType::Half), rfgrid);
												//+ vbalceofcur * torch::binary_cross_entropy(resall, ((tolrnl2.squeeze()).toType(c10::ScalarType::Float)).clone().detach().cuda(), {}, at::Reduction::Mean);
											//fwdhlblin = fwdhlblin.detach().contiguous();

												float loss = loss2.item().toFloat();
												lssdif = (loss - losslstg);
												if (lssdif == 0.) {
													fsw = false;
												}
												//if (std::abs(lssdif) < 1e-5 && itesrtraing > 0)
												//	std::exit(0);

												//if (lssdif > 1e-5) {
												//	torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
												//}
												//torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
												//if (lssdif != 0)
#if 1
												if (std::min(loss, lss2min) != lss2min) {
													minned = true;
													wasminned = true;
													//test2->zero_grad();
													//smpl->ema_update(0, *test2, 0., wcopy, wcopy);
													//smpl = Net(*dynamic_cast<NetImpl*>(test2->clone().get()));
													//smpl->layers[0].rnn1->flatten_parameters();
													//torch::save(test2, "test2.pt");
													//torch::save(fwdhlblout, "fwdhlblout.pt");
													//test2->zero_grad();
													//torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
													nonminned = 0;
													//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
													//runlr += runlr / 100;
													//xavsetted = false;
													//runlr = 0.01;

												}
												else {
													//smpl = Net(*dynamic_cast<NetImpl*>(test2->clone().get()));
													//if (!xavsetted) {
													//if (std::filesystem::exists("test2.pt")) {
													//		torch::load(test2, "test2.pt");
															//
													//	}
													//}
													/*if (summary.info == -5, 0) {
														//runlr += runlr / 100;
													//	runlr = 10.;
														//xt.zero_();
														//gt.zero_();
														//xavsetted = !xavsetted;
														//if (!xavsetted)
														//test2->zero_grad();
														//runlr = 0.01;
														torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
														//xavsetted = true;


														//fwdhlbl2.copy_(fwdhlbl2o.contiguous().detach());
														//xavsetted = false;
														//if (std::filesystem::exists("test2.pt")) {
														//		torch::load(test2, "test2.pt");
														//			//
														//}
														//test2 = Net(*dynamic_cast<NetImpl*>(smpl->clone().get()));
														//clonemodelto(smpl, test2);
														test2->ema_update(0, *smpl, 0., wcopy, wcopy);
														test2->layers[0].rnn1->flatten_parameters();
														test2->train();
														//test2->to(device);
															//}

													}
													#
													//else {
													//	xavsetted = false;
													//	fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
													//}
									///				if (std::filesystem::exists("fwdhlblout.pt")) {
										//				torch::load(fwdhlblout, "fwdhlblout.pt");
										//			}
										*/
										//	nonminned += 1;
													minned = false;
													//runlr = 1e-6;//runlr / 100;
												}
#endif
												lss2min = std::min(loss, lss2min);
												losslstg = loss;

												//lssdiftrgt = (loss2.item().toFloat() / 600000.);
												//runlr = 0.1 * loss2.item().toFloat();
												//lssdiv = 16666.6666667 * loss2.item().toFloat();//16666.6666667 * lossorigin;
												auto& runlrr = (!optsw, true ? runlr : runlr2);
												//if (lssdif < 0.)
												//	test2->zero_grad();
												if (sttteed && stalled, lssdif != 0., 1) {
													if ((std::abs(lssdif) < runlr) != (lssdif < 0.)) {
														//optim[2]->zero_grad();
														//lossdelta = 0.0;
														//itesr = -1;
														//dirwla = true;
														//auto &opt = dynamic_cast<tgt_optim_opts_t&>(optim[2]->param_groups()[0].options());
														//opt.weight_decay(opt.weight_decay() - (opt.weight_decay() / 100));
														//optsw = !optsw;
														//auto rls = runlrr / std::abs(runlrr);
														//runlrr = std::abs(runlrr);
														runlr += runlr / runlradv;
														runlr2 += runlr2 / runlradv;
														runlr3 += runlr3 / runlradv;
														//runlrr *= -rls;
														///lssdiftrgt -= lssdiftrgt / 100;
														zrgr = false;
														//opt.lr((runlr / lssdiffl / lssdiv));
														//if (lsdifftd > 1)
														//if (lssdif != 0.)
														//	lsdifft /= 10.;
														//lsdifftd -= 1.;
														//dynamic_cast<torch::optim::AdamWOptions&>(optim[2]->param_groups()[0].options()).weight_decay(0.).lr(0.001);
													}
													else {//if ((lssdif < opt.lr())) {
														//auto& opt = dynamic_cast<tgt_optim_opts_t&>(optim[2]->param_groups()[0].options());
														//opt.weight_decay(opt.weight_decay() + (opt.weight_decay() / 100));
														//runlrr = -runlrr;
														//runlrr = std::abs(runlrr);
														//opt.lr(opt.lr() - (opt.lr() / 100.));
														runlr -= runlr / runlradv;
														runlr2 -= runlr2 / runlradv;
														runlr3 -= runlr3 / runlradv;
														//optsw = !optsw;
														///lssdiftrgt += lssdiftrgt / 100;
														zrgr = true;
														//opt.lr((runlr / lssdiffl / lssdiv));
														//if (lsdifftd > 1)
														//if (lssdif != 0.)
														//	lsdifft *= 10.;
														//lsdifftd += 1.;
													}
													optim[1]->param_groups()[0].options().set_lr(runlr);
													dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[0].options()).weight_decay(runlr * 100.);
													optim[1]->param_groups()[1].options().set_lr(runlr2);
													dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[1].options()).weight_decay(runlr2 * 100.);
													optim[1]->param_groups()[2].options().set_lr(runlr3);
													dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[2].options()).weight_decay(runlr3 * 100.);
													//runlr = loss / lssdif ;
													//runlr = std::min(runlr, 1.f);
													//opt.lr(runlr);
													//runlradv = std::abs(lssdif) / 1e-6;
												}

												loss2.backward();
												//if (needchg) {
												//torch::nn::init::xavier_uniform_(fwdhlblin);
												//if (std::abs(lssdif) < 0.1)
												test2->mem.detach_();
												//	fwdhlblin.copy_((*fwdhlblout).contiguous().clone());

												//if (torch::all(fwdhlblout->isfinite()).item().toBool())
												//	fwdhlbloutst.copy_(fwdhlblout->contiguous().clone());
												//}
												//if (trainedb)
												//if (!torch::all(reswillwinotr.isfinite()).item().toBool())
												//fwdhlblin.copy_(fwdhlblout->contiguous().clone());


												//optim[2]->zero_grad();

												//optim[2]->step();
												localiters += 1;
												itesrtraing += 1;
												return loss2;

												};
											//fwdhlblout.zero_();


											//loss2 = cb(fwdhlbl2, &fwdhlblout);
											//if (!sttteed)
											//x.copy_(test->gather_flat_grad_uniform().to(device));
											//if (summarycu.num_iteration == 0) {
											//losslstgorig = loss2.item().toFloat();
											//}
											//	trainedb = loss2.item().toFloat() > 0.5;
											//loss2.backward();
											//optim[2]->step();
											//fwdhlbl2.zero_();
											typedef float mift;
											LBFGSB_CUDA_OPTION<mift> lbfgsb_options;
											lbfgsbcuda::lbfgsbdefaultoption<mift>(lbfgsb_options);
											lbfgsb_options.mode = LCM_CUDA;
											//lbfgsb_options.hessian_approximate_dimension = 4;
#if 1
											lbfgsb_options.eps_f = FLT_MIN;//1.0;
											lbfgsb_options.eps_g = FLT_MIN;//options.tolerance_grad();
											lbfgsb_options.eps_x = FLT_MIN;//options.tolerance_change();
											//lbfgsb_options.machine_epsilon = 0;
#endif
											lbfgsb_options.max_iteration = 100;//options.max_iter();
											lbfgsb_options.step_scaling = runlr;



											LBFGSB_CUDA_STATE<mift> statecu{};
											LBFGSB_CUDA_SUMMARY<mift> summarycu{};
											statecu.m_cublas_handle = cublas_handle;

											//auto opt_cond = (val(flat_grad.abs().max()) <= tolerance_grad);

											// optimal condition
											//if (opt_cond) {
											//	return orig_loss;
											//}
											//auto grm = test2->gather_flat_grad().abs().max();
											//{
											//	std::stringstream ss;
											///	ss << "grm " << grm << std::endl;
											//	orig_buf->sputn(ss.str().c_str(), ss.str().size());
											//}
											//if ((loss2.item().toFloat() < 0.1)) { //|| (grm <= 1e-7).item().toBool()) {
											//	break;
											//}
											auto clos = std::bind(cb, fwdhlbl2, &fwdhlblout);

											statecu.m_funcgrad_callback = [&](
												mift* x, mift& f, mift* g,
												const cudaStream_t& stream,
												const LBFGSB_CUDA_SUMMARY<mift>& summary) {
													std::unique_lock lk(trainm);
													//lbfgsb_options.eps_f = 1e-15;
													auto xt = torch::from_blob(x, { test2->nels }, device);
													auto gt = torch::from_blob(g, { test2->nels }, device);
													if (lssdif < 1e-5, 0)
													{
														// test2->zero_grad();
														//std::unique_lock lk(trainm);
														torch::AutoGradMode enable_grad(true);
														cb(fwdhlbl2, &fwdhlblout, test2).item<float>();
														optim[1]->step();
														//std::stringstream ss;
														//ss << "loss " << loss2 << std::endl;
														//ss << "runlr2 " << runlr2 << std::endl;
														//ss << "lssdif " << lssdif << std::endl;
														//ss << "t " << t << std::endl;
														//orig_buf->sputn(ss.str().c_str(), ss.str().size());
														//loss = cb(fwdhlbl2, &fwdhlblout, test2).item<float>();
														//if (summarycu.num_iteration == 0) {
															//if (itesrtrain == 0)
														//		losslstgorig = loss2.item().toFloat();
														//}
														//if (loss < 0.1) {
														//	lbfgsb_options.eps_f = 1.;
														//	return 0;
														//}
													}
													//if ((std::abs(lssdif) < 1e-5) == minned) 
													//	runlr -= runlr / 100.;
													//else
													//	runlr += runlr / 100.;
													//optim[1]->param_groups()[0].options().set_lr(runlr);
													//auto &opt = dynamic_cast<tgt_optim_opts_t&>(optim[2]->param_groups()[0].options());
															//opt.weight_decay(opt.weight_decay() - (opt.weight_decay() / 100));
													//auto err = cudaMemcpy(xt.data_ptr<float>(), x, N_elements * sizeof(float), ::cudaMemcpyDefault);
													//err = cudaMemcpy(gt.data_ptr<float>(), g, N_elements * sizeof(float), ::cudaMemcpyDefault);
													//auto gth = gt.to(optionscu);
													float loss;
													torch::Tensor flat_grad;
													std::stringstream ss;
													lststalled = stalled;
													stalled = !torch::any(xt != 0.).item().toBool();
													if (stalled) {
														//runlr += runlr / 100;
														//test2->zero_grad();
														//torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
													}
													else {
														//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
														//test2->zero_grad();
														//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
														//runlr += runlr / 100;
													}
													/*
													if (n_iter >= options.max_iter())
														return 0;
														*/


														//ss << " " << std::hex << res << std::endl;
													//ss << "triggered at: " << res[2] << std::endl;
													//ss << "pred is: " << (res[0] > res[1]) << std::endl;
													  // ss << gt << std::endl;
													   //orig_buf->sputn(ss.str().c_str(), ss.str().size());
													   // ss << gt << std::endl;
													   //             _add_grad(t, d);
												// NOLINTNEXTLINE(cppcoreguidelines-init-variables)
														//_set_param1(gt);
														//auto lstflat_grad = _gather_flat_grad();
													auto t = runlr2;//(stalled ? 1. : 0.1);
													lbfgsb_options.step_scaling = t;
													///if (torch::any(xt != 0.).item().toBool())
													//{
													//	std::stringstream ss;
													//	ss << "xt " << xt << std::endl;
													//	orig_buf->sputn(ss.str().c_str(), ss.str().size());
													//}
													//Net smpl = Net(*dynamic_cast<NetImpl*>(test2->clone().get()));
													//smpl->add_grad(t, xt.to(device));
													//smpl->layers[0].rnn1->flatten_parameters();
													if (!sttteed, 0)
														test2->add_grad(t, xt.to(device));
													else {
														test2->set_grad(1., xt.to(device).toType(c10::ScalarType::Half));
														optim[1]->step();
													}
													//optsw = !optsw;
													//_wd(0.1);
													//if (!sttteed) {
													//	sttteed = stalled;
													//}
													//else {
													//	sttteed = !stalled;
													//}
													if ((losslstgr - loss2.item().toFloat()) < 0., 0) {

														//test2->zero_grad();
														//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
														//torch::nn::init::xavier_uniform_(fwdhlbl2, runlr);
														runlr = 0.01;// 1e-6;
														runlr2 = 1.;
													}
													if (lssdif == 0., stalled) {
														//runlr -= runlr / 100;
														//test2->zero_grad();
														losslstgr = loss2.item().toFloat();
														//sttteed = false;


													}
													else {
														//if (lststalled)
														//	runlr = 1.;
														//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
													}
													//runlr2 -= runlr2 / 100;
													//runlr -= runlr / 100;
													if (summarycu.num_iteration > 0 && stalled) {

													}
													//if (stalled)
													{
														//std::unique_lock lk(trainm);
														torch::AutoGradMode enable_grad(true);
														loss = cb(fwdhlbl2, &fwdhlblout, test2).item<float>();
														//optim[1]->step();
														//loss = cb(fwdhlbl2, &fwdhlblout, test2).item<float>();
														//if (summarycu.num_iteration == 0) {
															//if (itesrtrain == 0)
														//		losslstgorig = loss2.item().toFloat();
														//}
														//if (loss < 0.1) {
														//	lbfgsb_options.eps_f = 1.;
														//	return 0;
														//}
													}

													flat_grad = test2->gather_flat_grad();
													//_wd(t * f, &flat_grad);
												  //  _set_param1(xt);
												//}
												//else {
												//	flat_grad = test2->gather_flat_grad();
												//	loss = loss2.item().toFloat();
													//_wd(t * f, &flat_grad);
													//_set_param1(xt);
												//}
												//_add_grad(t, xt);
												//return std::make_tuple(loss, flat_grad);
												//auto [loss, grad] = _directional_evaluate1(closure, xt, t, gt);
												//double* tg;
												//cudaMalloc(&tg, N_elements * sizeof(tg[0]));
												//gt.copy_(grad);
												//xt.copy_(grad);
#if 0
				   // auto lm = [g, grad, N_elements, options, pf=&f, loss] {
					 //   torch::from_blob(g, { N_elements }, options).copy_(grad.cuda().toType(c10::ScalarType::Double));
					   // *pf = loss;
						//};
					//cudaStreamSynchronize(stream);
				   // auto t = torch::from_blob(g, { N_elements }, options).copy_(grad.cuda().toType(c10::ScalarType::Double));
													cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
														//decltype(lm)* plm = (decltype(lm)*)userData;
														//const std::string str{ "copying..." };
													   // orig_buf->sputn(str.c_str(), str.size());
														//(*plm)();
														//delete plm;
														cudaFree(userData);
														}, x, 0);
#endif
													//torch::from_blob(g, { N_elements }, optionscu).copy_(grad.cuda().toType(c10::ScalarType::Double));
													//if (!sttteed)
													//sttteed = lststalled && stalled;
													///if (!stalled) {
													//	sttteed = false;
													//}
													//if (stalled) {
													if (sttteed, 0)
														gt.copy_(test->gather_flat_grad_uniform().to(device) * t);
													else
														gt.copy_(flat_grad.to(device).toType(sctpt));
													//}
													//lssdif = (loss - losslstg);



													//if (lssdif > 1e-5) {
													//	torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
													//}
													//torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
													//if (lssdif != 0)
#if 0
													if (std::min(loss, lss2min) != lss2min, 0) {
														minned = true;
														smpl->ema_update(0, *test2, 0., wcopy, wcopy);
														//smpl = Net(*dynamic_cast<NetImpl*>(test2->clone().get()));
														//smpl->layers[0].rnn1->flatten_parameters();
														//torch::save(test2, "test2.pt");
														//torch::save(fwdhlblout, "fwdhlblout.pt");
														//test2->zero_grad();
														//torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
														nonminned = 0;
														//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
														//runlr += runlr / 100;
														//xavsetted = false;
														//runlr = 0.01;

													}
													else {
														//smpl = Net(*dynamic_cast<NetImpl*>(test2->clone().get()));
														//if (!xavsetted) {
														//if (std::filesystem::exists("test2.pt")) {
														//		torch::load(test2, "test2.pt");
																//
														//	}
														//}
														if (summary.info == -5, 0) {
															//runlr += runlr / 100;
														//	runlr = 10.;
															//xt.zero_();
															//gt.zero_();
															//xavsetted = !xavsetted;
															//if (!xavsetted)
															//test2->zero_grad();
															//runlr = 0.01;
															torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
															//xavsetted = true;


															//fwdhlbl2.copy_(fwdhlbl2o.contiguous().detach());
															//xavsetted = false;
															//if (std::filesystem::exists("test2.pt")) {
															//		torch::load(test2, "test2.pt");
															//			//
															//}
															//test2 = Net(*dynamic_cast<NetImpl*>(smpl->clone().get()));
															//clonemodelto(smpl, test2);
															test2->ema_update(0, *smpl, 0., wcopy, wcopy);
															test2->layers[0].rnn1->flatten_parameters();
															test2->train();
															//test2->to(device);
																//}

														}
														//else {
														//	xavsetted = false;
														//	fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
														//}
										///				if (std::filesystem::exists("fwdhlblout.pt")) {
											//				torch::load(fwdhlblout, "fwdhlblout.pt");
											//			}

													//	nonminned += 1;
														minned = false;
														//runlr = 1e-6;//runlr / 100;
													}
													lss2min = std::min(loss, lss2min);

													losslstg = loss;

													//lssdiftrgt = (loss2.item().toFloat() / 600000.);
													//runlr = 0.1 * loss2.item().toFloat();
													//lssdiv = 16666.6666667 * loss2.item().toFloat();//16666.6666667 * lossorigin;

													if (sttteed && stalled, 0) {
														if ((std::abs(lssdif) > runlr) == minned, 0) {
															//optim[2]->zero_grad();
															//lossdelta = 0.0;
															//itesr = -1;
															//dirwla = true;
															//auto &opt = dynamic_cast<tgt_optim_opts_t&>(optim[2]->param_groups()[0].options());
															//opt.weight_decay(opt.weight_decay() - (opt.weight_decay() / 100));

															runlr += runlr / 100;
															///lssdiftrgt -= lssdiftrgt / 100;
															zrgr = false;
															//opt.lr((runlr / lssdiffl / lssdiv));
															//if (lsdifftd > 1)
															//if (lssdif != 0.)
															//	lsdifft /= 10.;
															//lsdifftd -= 1.;
															//dynamic_cast<torch::optim::AdamWOptions&>(optim[2]->param_groups()[0].options()).weight_decay(0.).lr(0.001);
														}
														else {//if ((lssdif < opt.lr())) {
															//auto& opt = dynamic_cast<tgt_optim_opts_t&>(optim[2]->param_groups()[0].options());
															//opt.weight_decay(opt.weight_decay() + (opt.weight_decay() / 100));

															//opt.lr(opt.lr() - (opt.lr() / 100.));
															runlr -= runlr / 100;
															///lssdiftrgt += lssdiftrgt / 100;
															zrgr = true;
															//opt.lr((runlr / lssdiffl / lssdiv));
															//if (lsdifftd > 1)
															//if (lssdif != 0.)
															//	lsdifft *= 10.;
															//lsdifftd += 1.;
														}
														//runlr = loss / lssdif ;
														//runlr = std::min(runlr, 1.f);
														//opt.lr(runlr);
													}
													lststtteed = sttteed;
#endif
													//if (lssdif == 0) {
													//	xt.zero_();
														//test2->zero_grad();
														//fwdhlbl2.copy_(fwdhlbl.contiguous().detach());
													//}
													//else {
													//	x.zero_();// = torch::zeros({ test2->nels }).to(device);//test2->gather_flat_grad().to(device);//torch::zeros({ N_elements }).to(optionst);
													//	g.zero_();
													//}
													//xt.copy_(flat_grad.neg().to(device));
													//auto err = cudaMemcpy(g, flat_grad.data<float>(), test2->nels * sizeof(float), ::cudaMemcpyDefault);
													//err = cudaMemcpy(x, grad.cuda().toType(c10::ScalarType::Double).data<double>(), N_elements * sizeof(double), ::cudaMemcpyDefault);
													if (summarycu.num_iteration > 0, 1) {
														ss << loss << std::endl;
														//ss << _gather_flat_grad() - flat_grad << std::endl;
														//ss << !t.isnan().any().item<bool>();
														//ss << !t.isfinite().any().item<bool>();
													  // d = grad.neg();
														ss << "lr " << optim[1]->param_groups()[0].options().get_lr() << std::endl;
														ss << "lr1 " << optim[1]->param_groups()[1].options().get_lr() << std::endl;
														ss << "lr2 " << optim[1]->param_groups()[2].options().get_lr() << std::endl;
														ss << "lssdif " << lssdif << std::endl;
														//ss << "runlr " << runlr << std::endl;
														//ss << "runlr2 " << runlr2 << std::endl;
														//ss << "runlr3 " << runlr3 << std::endl;
														orig_buf->sputn(ss.str().c_str(), ss.str().size());
													}
#if 0
													else {
														ss << "n_iter " << n_iter << std::endl;
														// ss << xt << std::endl;
														 //ss << !t.isnan().any().item<bool>();
														 //ss << !t.isfinite().any().item<bool>();
													   // d = grad.neg();
														orig_buf->sputn(ss.str().c_str(), ss.str().size());
													}
#endif
													//shed.step(loss);
													f = loss;
													//n_iter += 1;
													//d = grad.neg();
											  //      orig_loss.fill_(loss);
													//state.d(xt);
													//state.t(f);
													//_add_grad(t, d);
													return (+(lssdif == 0.));
												};
											// cusrch();


												//if (lssdif == 0.) {
												//	x = torch::zeros({ test2->nels }).to(device);
													//torch::nn::init::xavier_uniform_(x);
												//}
												//do {
												//	statecu.m_funcgrad_callback(x.data<float>(), f,
												///		g.data<float>(), 0, summarycu);
												//} while (1);
												//runlr = 1e-10;
											test2->zero_grad();
											//{0.005}.weight_decay(0.5)
											//optim[1]->param_groups()[0].options().set_lr(0.005);
											//dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[0].options()).weight_decay(0.5);
											lk.unlock();
											optsw = true;
											if (optsw, 1) {
												//test2->zero_grad();
												lbfgsbcuda::lbfgsbminimize(test2->nels, statecu, lbfgsb_options, x.data<mift>(), nbd.data<int>(),
													xl.data<mift>(), xu.data<mift>(), summarycu);
											}
											//optsw = true;
											//lbfgsbcuda::lbfgsbminimize(test2->nels, statecu, lbfgsb_options, x.data<mift>(), nbd.data<int>(),
											//	xl.data<mift>(), xu.data<mift>(), summarycu);
											else {
												//do {
												cb(fwdhlbl2, &fwdhlblout, test2).item<float>();
												//optim[1]->param_groups()[0].options().set_lr(std::abs(runlr));
												//dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[0].options()).weight_decay(std::abs(runlr) * 100.);
												optim[1]->step();
												//	std::stringstream ss;
												//	ss << "loss " << loss2 << std::endl;
													//ss << "runlr2 " << runlr2 << std::endl;
												//	ss << "lssdif " << lssdif << std::endl;
													//ss << "t " << t << std::endl;
												//	orig_buf->sputn(ss.str().c_str(), ss.str().size());
											}
											//	while (std::abs(lssdif) > 1e-5);
											//}

											lk.lock();
											if ((losslstgor - loss2.item().toFloat()) != 0.) {

											}
											//torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
											/*static bool started = false;
											if (!started) std::thread{ [=, &optim] {
											auto trgt = loss2.item().toFloat();
											do
											 {

												{
													// test2->zero_grad();
													std::unique_lock lk(trainm);
													torch::AutoGradMode enable_grad(true);
													cb(fwdhlbl2, &fwdhlblout, test2).item<float>();
													optim[1]->step();
													std::stringstream ss;
													ss << "loss " << loss2 << std::endl;
													//ss << "runlr2 " << runlr2 << std::endl;
													//ss << "lssdif " << lssdif << std::endl;
													//ss << "t " << t << std::endl;
													orig_buf->sputn(ss.str().c_str(), ss.str().size());
													//loss = cb(fwdhlbl2, &fwdhlblout, test2).item<float>();
													//if (summarycu.num_iteration == 0) {
														//if (itesrtrain == 0)
													//		losslstgorig = loss2.item().toFloat();
													//}
													//if (loss < 0.1) {
													//	lbfgsb_options.eps_f = 1.;
													//	return 0;
													//}
												}
											} while (loss2.item().toFloat() > 0.1);
											} }.detach();*/
											//started = true;
											//torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
											losslstgor = loss2.item().toFloat();
											//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
											//fwdhlbl.zero_();
											//runlr = 0.01;
											static bool lsttrgtc = false;
											bool trgtc = (losslstgorig - loss2.item().toFloat() == 0.);
											//if (!(losslstgorig - loss2.item().toFloat() < 1e-5)) {
											//	runlr = 1e-5;
											//}
											//if (minned != trgtc) {
											//	runlr += runlr / 100;
											//}
											//else {
											//if (std::abs(losslstgorig - loss2.item().toFloat()) < 1e-5)
											//	optsw = !optsw;
											bool optc = false;
											static int minsz = 1;
											static bool minszd = false;
											bool wasminnedl = wasminned;
											//waschanged = fsw;
											if ((std::abs(losslstgorig - loss2.item().toFloat()) < 1e-7)) {
												/*if (std::filesystem::exists("test2.pt")) {
													torch::load(test2, "test2.pt");

												}
												if (std::filesystem::exists("fwdhlblout.pt")) {
													torch::load(fwdhlblout, "fwdhlblout.pt");
												}*/
												//test2->zero_grad();
												//lk.unlock();
												//if (runlr > 0.1)
												//optc = runlr > 0.1 || runlr < 1e-10;
												//if (runlr > 0.1 || runlr < 1e-10)
												//	runlr = 1e-4;
												//torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
												//(!optsw ? runlr : runlr2) += (!optsw ? runlr : runlr2) / 10;
												//optsw = !optsw;
												if (wasminned) {
													//runlr = 1e-6;
													//runlr2 = 1e-6;

												}
												//runlr += runlr / 100;
												//cb(fwdhlbl2, &fwdhlblout, test2).item<float>();
												//optim[1]->param_groups()[0].options().set_lr(std::abs(runlr));
												//dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[0].options()).weight_decay(std::abs(runlr) * 100.);
												wasminned = false;
												/*runlr = runlr2b;
												runlr2 = runlr2b;
												if (fsw) {
													runlr2b -= runlr2b / 100.;
												}
												else {
													runlr2b = 1e-4;
												}*/
												//std::swap(runlr, runlr2);
												//test2->ema_update(0, *smpl, 0., wcopy, wcopy);
												waschanged += 1;

												//fsw = true;
												//lbfgsbcuda::lbfgsbminimize(test2->nels, statecu, lbfgsb_options, x.data<mift>(), nbd.data<int>(),
												//	xl.data<mift>(), xu.data<mift>(), summarycu);
												//lk.lock();
											}
											else {
												//runlr -= runlr / 100;
												waschanged = 0;
												//waschanged += 1;
												//if (!fsw)
												//	waschanged += 1;
												//else

												//fsw = false;
											}
											//cb(fwdhlbl2, &fwdhlblout, test2).item<float>();
											//optim[1]->param_groups()[0].options().set_lr(std::abs(runlr));
											//dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[0].options()).weight_decay(std::abs(runlr) * 100.);
											//if (itesrtrain % 2 == 0) {
											//	runlr = 0.005;
											//}
											//else {
											//	runlr = 1e-6;
											//}
											//	runlr -= runlr / 100;
											//}
											//while (cb(fwdhlbl2, &fwdhlblout, test2).item<float>() > 0.1) {
												//optim[1]->step();
											//}
											//test2->zero_grad();
											//lk.unlock();
											//lbfgsbcuda::lbfgsbminimize(test2->nels, statecu, lbfgsb_options, x.data<mift>(), nbd.data<int>(),
											//	xl.data<mift>(), xu.data<mift>(), summarycu);
											//lk.lock();
											minned = 0;
											//else {
											//	runlr = 0.01;
											//}
											//if (trgtc && !sttteed) {
												//	//runlr = 0.0000005;
													//sttteediters = 0;
											//	sttteediters += 1;
											//	sttteed = !sttteed;
												//losslstgorig1 = loss2.item().toFloat();
											//	//test2->zero_grad();
												//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
												//torch::nn::init::xavier_uniform_(fwdhlbl2);
											//}
										//	if (sttteed) {
										//		sttteed = trgtc;
												//if (sttteed && runlr < 1e-9)
												//	sttteed = !sttteed;
												//if (!sttteed)
												//	runlr = 0.0000005;
												//if (!sttteed) {
												//	sttteediters += 1;
												//}
										//	}
											//if (sttteed == lststtteed) {
											//	if (lststc != trgtc)
											//		runlr = 0.0000005;
											//	
											//}
											lststc = trgtc;
											//if (sttteedorig != sttteed) {
											//	sttteedorig = sttteed;
											//}
											//if ((itesrtrain > 1) && sttteedorig != sttteed)
											losslstgorig = loss2.item().toFloat();
											//if (nonminned > 19) {
											//	torch::nn::init::xavier_uniform_(fwdhlbl2, 1.);
											//	nonminned = 0;
											//}
											//if (loss2.item().toFloat() > 10.) {
											//	std::exit(0);
											//}
											//runlr = 0.00000166666 * loss2.item().toFloat();
											//runlr2 = 0.00000166666 * loss2.item().toFloat();
											//optim[1]->param_groups()[0].options().set_lr(runlr);
											//dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[0].options()).weight_decay(std::abs(runlr) * 100.);
											if (itesrtrain > 1000, loss2.item().toFloat() < 0.1, !fsw, 1) {//, itesrtrain > 3, 1) { //|| !wasminnedl && !fsw) {// || (loss2.item().toFloat() / lss2min) > 2. || sttteed,1) { //|| sttteediters == 2) {//(itesrtrain > 1) && sttteedorig != sttteed) {//loss2.item().toFloat() < 0.1)
												//shed.step(loss2.item().toFloat());
												//test2->ema_update(0, *smpl, 0., wcopy, wcopy);
												//trainedb = loss2.item().toFloat() < 0.68;
												//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
												//break;
												/*fsw = fsw && itesrtrain < 100;
												if (fsw) {
													minsz -= 1;


													wasminned = true;
												}
												else {
													minsz += 1;
													wasminned = false;
												}
												if (minsz < 0)

													minsz = 0;
												if (minsz > totrainlm.size(0))
													minsz = totrainlm.size(0) - 1;
												//if (minsz <= 1 || minsz >= 10)
												//	minszd = !minszd;
												//if (minszd)
												//	minsz += 1;
												//else
												//	minsz -= 1;
												if (totrainlm.size(0) > 1) {
													//test2 = Net();
													//smpl = Net();
													//fwdhlbl.copy_(fwdhlblout.contiguous().detach());
													fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
													betsitesrmade = 0;
													//auto lstc = totrainlm[-1].clone().detach();
													//auto lstcl = tolrnl52m[-1].clone().detach();
													totrainlm = torch::flip(totrainlm, 0);
													std::vector sz = totrainlm.sizes().vec();
													sz[0] -= minsz;
													totrainlm.resize_(sz);
													totrainlm = torch::flip(totrainlm, 0);

													tolrnl52m = torch::flip(tolrnl52m, 0);
													sz = tolrnl52m.sizes().vec();
													sz[0] -= minsz;
													tolrnl52m.resize_(sz);
													tolrnl52m = torch::flip(tolrnl52m, 0);

												}*/
												lsttrgtc = false;
												if (currmdl == 1, 1) {

													sttteedorig = sttteed;
													itesrtrain = 0;
													itesrtraing = 0;
													btrain = false;
													ierstr += 1;
													dobetr = 1;
													currmdl = 0;
												}
												else {
													itesrtrain = 0;
													itesrtraing = 0;
													currmdl += 1;
													dobetr = 0;
												}


											}
											else
												dobetr = 0;
											lsttrgtc = trgtc;
										}
										if (dobetr) {
											//torch::nn::init::xavier_uniform_(fwdhlbl2);
											//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
											if (betsitesrmade == 400) {
												runlr = 1e-6;//0.05;//0.00000166666 * loss2.item().toFloat();
												runlr2 = 0.0005;
												runlr3 = 0.0005;//0.00000166666 * loss2.item().toFloat();
												runlradv = 100.;
												rfgridlst = abvsgrids.toType(c10::ScalarType::Bool).bitwise_and(rfgrid.clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);//rfgrid.clone().detach();
												//totrainllst = abvsgrids[0][betsitesrmade400g % 20].clone().detach().unsqueeze(0).unsqueeze(0);//reswillwino.reshape_as(rfgrid).mean(1).unsqueeze(0);

												//abvsgridsvals.fmod_(1.);
												//abvsgridsvals += tolrnll2.toType(c10::ScalarType::Int);
												//abvsgridsvals = torch::roll(abvsgridsvals, 1);
												//if (abvsgridsvals.min().item().toInt() > 0) {
												//	abvsgridsvals -= abvsgridsvals.min();
												//}
												//totrainllst = abvsgrids.clone().detach();//abvsgridsvals.toType(c10::ScalarType::Float) / abvsgridsvals.max().item().toFloat();
												//if (reswillwino.defined())
												//tolrnll2 *= reswillwino.reshape_as(tolrnll2);
												//tolrnll2 = abvsgrids.toType(c10::ScalarType::Bool).logical_not().toType(c10::ScalarType::Float);
												//else
												//totrainllst = abvsgrids.clone().detach();
												//DeleteFile(".\\test2.pt");
												//DeleteFile(".\\fwdhlblout.pt");
												//exit(0);
												//torch::save(test2, "test2.pt");
												//torch::save(fwdhlbl2, "fwdhlbl2.pt");
												//torch::save(totrainlm, "totrainlm.pt");
												//torch::save(tolrnl52m, "tolrnl52m.pt");
												//cb(fwdhlbl2, &fwdhlblout, test2).item<float>();
												//optim[1]->param_groups()[0].options().set_lr(std::abs(runlr));
												//dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[0].options()).weight_decay(std::abs(runlr) * 100.);
												//runlr = runlr / 10.;
												//abvsgridslst = abvsgrids.clone().detach();
												//orand = vbal2 < lstvbal2;
												//if (!orand)
												//totrainllst = tolrnll2.clone().detach();
												//totrainllst = torch::roll(totrainllst, 1);
												///if (reswillwino.defined()) {
												///	totrainllst = (reswillwino > 0.5).toType(c10::ScalarType::Float).clone().detach().reshape_as(totrainllst);
												//	totrainllst = torch::roll(totrainllst, betsitesrmade400g);
												//}

													//abvsgrids = abvsgrids.toType(c10::ScalarType::Bool).bitwise_xor(rfgrid.clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);
													//abvsgridsvals = torch::mean(torch::vstack({ abvsgridsvals, abvsgrids }), 0).unsqueeze(0);
													//abvsgridsvals.fmod_(1.);
													//abvsgridsvals += tolrnll2.toType(c10::ScalarType::Int);
													//abvsgridsvals = torch::roll(abvsgridsvals, 1);
													//if (abvsgridsvals.min().item().toInt() > 0) {
													//	abvsgridsvals -= abvsgridsvals.min();
													//}
													//totrainllst = abvsgrids.clone().detach();//abvsgridsvals.toType(c10::ScalarType::Float) / abvsgridsvals.max().item().toFloat();
													//if (reswillwino.defined())
													//tolrnll2 *= reswillwino.reshape_as(tolrnll2);
													//tolrnll2 = abvsgrids.toType(c10::ScalarType::Bool).logical_not().toType(c10::ScalarType::Float);
													//else
												totrainllst = test2->mem.detach().clone();//abvsgrids.clone().detach()[0][betsitesrmade400g % 20].unsqueeze(0).unsqueeze(0);
												//totrainllst = torch::mean(torch::vstack({ totrainllst, abvsgrids.clone().detach()[0][betsitesrmade400g % 20].unsqueeze(0).unsqueeze(0) }), 0).unsqueeze(0);
												//	abvsgrids = abvsgrids.toType(c10::ScalarType::Bool).bitwise_or(rfgrid.clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);
												auto totrainllstlst = totrainllst.clone().detach();
												//tolrnll2 = (rfgrid.clone().detach() > 0.).toType(c10::ScalarType::Float)[0].unsqueeze(0);
												//tolrnll2 = tolrnll2 / tolrnll2.max();
												//tolrnll2 = tolrnll2.clone().detach();
												//if (orand)
												//	totrainllst = totrainllst.toType(c10::ScalarType::Bool).bitwise_and(rfgrid.clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);
												//else
												//	totrainllst = totrainllst.toType(c10::ScalarType::Bool).bitwise_xor(abvsgrids.clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);
												//if (reswillwino.defined()) {
												//	totrainllst = (reswillwino.reshape_as(abvsgrids) * abvsgrids)[0][betsitesrmade400g % 20].clone().detach().unsqueeze(0).unsqueeze(0);//(torch::sigmoid(reswillwino) > 0.5).toType(c10::ScalarType::Float).clone().detach().reshape_as(rfgrid).mean(1).unsqueeze(0);
													//totrainllst[0][0] = abvsgrids[0][betsitesrmade400g % 20].toType(c10::ScalarType::Bool).bitwise_and(totrainllst[0][0].clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);
												//}

												abvsgridslst = abvsgrids.clone().detach();
												//abvsgrids[0][betsitesrmade400g % 20] = abvsgrids[0][betsitesrmade400g % 20].toType(c10::ScalarType::Bool).bitwise_xor(rfgrid[0][betsitesrmade400g % 20].clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);
												abvsgrids = abvsgrids.flatten().toType(c10::ScalarType::Bool).bitwise_and(rfgridlst.flatten().flip(0).clone().detach().toType(c10::ScalarType::Bool)).bitwise_not().toType(c10::ScalarType::Float).flatten().flip(0).reshape_as(abvsgrids);
												//totrainllst = torch::vstack({ totrainllst, rfgrid.clone().detach() });
												//auto toinf = totrainllst.toType(c10::ScalarType::Bool).bitwise_xor(rfgrid.clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);
#if 1
												test2->eval();
												//test2->mem.copy_(test2->mem.toType(c10::ScalarType::Bool).bitwise_xor(totrainllst.clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float));
												//auto toinfer = reswillwino.defined() ? reswillwino.clone().detach().reshape_as(totrainllst) : torch::zeros({1, 20, 20}).cuda();
												auto [resallpr, reswillwinpr] = test2->forward(totrainllst, abvsgridslst, rfgridlst, fwdhlbl2, nullptr, 0);
												//fwdhlbl.copy_(fwdhlbloutst.contiguous().clone());
												//auto [resallpr1, reswillwinpr1] = test2->forward(totrainllst, fwdhlbl2, nullptr, 1);
												reswillwino = resallpr.squeeze(0).flatten(1);

												//totrainllst = toinfer.clone().detach();
												//totrainllst = abvsgridsvals.clone().detach();
												//tolrnll2 = ((resallpr.squeeze(0).clone().detach()) > 0.5).toType(c10::ScalarType::Float);
												//reswillwino1 = reswillwinpr1.squeeze(0).flatten(1);
												if (!torch::all(reswillwino.isfinite()).item().toBool()) {
													std::exit(0);
												}
												//torch::zeros({ test2->nels }).to(device);
												//if (!sttteed)
												//fwdhlbl2.copy_(fwdhlblout.contiguous().clone());
												test2->zero_grad();
#endif
												if (todrawres.defined() && todrawres.size(0) > 0) for (auto in = 0; in < 2; ++in) {
													auto res1 = todrawres[0].swapaxes(-1, -2)[in].flatten();
#if 0
													if (itesrg == 0) {
														auto meani = (torch::arange(20, res1.options()) * res1).sum() / res1.sum();
														actualdir4 = meani.item().toFloat() > 10.;
														dirmean = meani.item().toFloat();
														//dobetr = 1;
													}
#endif
													//for (int in =0; in < 1;++in)
													if (!inited, true) {
														//res1 *= 4;

														int i = ind1;
														{
															if (i % 3 == 0) {
																//triangleVertices[i].vals[6] = 1.0;
																//triangleVertices[i].vals[5] = 1.0;
																//triangleIndices[i] = i;
															}
															else if (i % 2 == 0) {
																//triangleVertices[i].vals[6] = 1.0;
																//triangleIndices[i] = i;
															}
															else {
																//triangleIndices[i] = 49 - i;
																//triangleVertices[i].vals[6] = 1.0;
																//triangleVertices[i].vals[4] = 1.0;
															}
															triangleVertices[i].vals[0] = (res1[0].item().toFloat() * 2. - 1.);
															triangleVertices[i].vals[1] = (res1[1].item().toFloat() * 2. - 1.);
															triangleVertices[i].vals[2] = res1[2].item().toFloat();
															triangleVertices[i].vals[3] = 1.0;//res[3].item().toFloat();// + 0.15;
															triangleVertices[i].vals[4] = torch::tanh(res1[3]).item().toFloat();
															triangleVertices[i].vals[5] = torch::tanh(res1[4]).item().toFloat();
															triangleVertices[i].vals[6] = torch::tanh(res1[5]).item().toFloat();
															triangleVertices[i].vals[7] = torch::tanh(res1[7]).item().toFloat() * 360.;//genranf();
															triangleIndices[i] = i % 2 == 0 ? i : 49 - i;
														}
														ind1 += 1;
														inited = ind1 >= 50;

														if (inited) {
															//memset(triangleVertices, 0, sizeof triangleVertices);
															//memset(triangleIndices, 0, sizeof triangleIndices);
															//std::cout << "inited!" << std::endl;
															ind1 = 0;
#if 0
															maxx = FLT_MIN;
															minx = FLT_MAX;
															maxy = FLT_MIN;
															miny = FLT_MAX;
															for (int y = 0; y < 50; ++y) {
																maxx = std::max(triangleVertices[y].vals[0], maxx);
																minx = std::min(triangleVertices[y].vals[0], minx);

																maxy = std::max(triangleVertices[y].vals[1], maxy);
																miny = std::min(triangleVertices[y].vals[1], miny);
															}
#endif
														}

													}
												}
#if 0
												auto prevlss = torch::mse_loss(abvsgrid, totrainllst).item().toFloat();
												std::stringstream ss;
												ss << "prevlss " << prevlss << std::endl;
												//
												orig_buf->sputn(ss.str().c_str(), ss.str().size());
												totrainllst = abvsgrid.clone().detach();
#endif


												auto totrainll = totrainllst;//totrainl.select(-1, 0).select(-1, 0).unsqueeze(-1).unsqueeze(-1);
#if 1
												//if (!(vbal2 > lstvbal2) || !totrainlm.defined(), 1) {

												if (totrainlm.size(0) < 2, 1) {
													if (betsitesrmade > 0, totrainlm.defined(), 0)
														totrainlm = torch::vstack({ totrainlm, totrainllst }).cuda();
													else
														totrainlm = totrainllst.clone().detach();
												}
												else {
													totrainlm = torch::roll(totrainlm, -1, 0);
													totrainlm[-1] = totrainllst[0];
												}


												//}
#endif
																									//totrainllst = reswillwino.clone().detach().reshape_as(rfgrid).mean(1).unsqueeze(0);
																									//abvsgridsvals = torch::mean(torch::vstack({ abvsgridsvals, abvsgrids }), 0).unsqueeze(0);
																									//std::exit(0);
																									//totrainllst = abvsgrid.clone().detach();
																									//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
												betsitesrmade = 0;
												//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
												//abvsgrids.zero_();
												//if (reswillwino.defined())
												//abvsgrids = abvsgrids.toType(c10::ScalarType::Bool).logical_not().clone().detach().toType(c10::ScalarType::Float);
												//abvsgrids = rfgrid.clone().detach();
												//if (torch::all(abvsgrids == 0.).item().toBool()) {
												//	abvsgrids += 1.;
												//}
												orand = !orand;
												itesrwin = 0;
												acccoef = 0.;
												//Net smpl;
												//clonemodelto(smpl, test2);
												//tolrnll2.zero_();
												//tolrnll2 = tolrnll2.toType(c10::ScalarType::Bool).bitwise_xor(rfgrid.clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);
												//fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
												//totrainlm.zero_();// = totrainlm[-1].unsqueeze(0);
												//tolrnl52m// = tolrnl52m[-1].unsqueeze(0);
												//torch::nn::init::xavier_uniform_(totrainllst);
												if (loss2.item().toFloat() < 0.1, 0) {
													fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
													tolrnl52m.~decltype(tolrnl52m)();
													totrainlm.~decltype(totrainlm)();
													new(&tolrnl52m)decltype(tolrnl52m)();
													new(&totrainlm)decltype(totrainlm)();
												}
												//fwdhlbl.copy_(fwdhlblout.contiguous().detach());
											}


											//totrainlm2 = torch::vstack({ totrainlm2, totrainll }).cuda();

											//loss2.backward();
											//if (needchg) {
												//fwdhlblin.copy_(fwdhlblout->contiguous().clone());


										}

										//auto& opt = dynamic_cast<tgt_optim_opts_t&>(optim[2]->param_groups()[0].options());


										//runlr /= localiters;
										//if (lssdif == 0) {
										//	runlr = 0.001;
										//}
										//opt.weight_decay(0.);
										//optim[2]->zero_grad();
										//if (lssdif == 0.) {
											//torch::Tensor fkgr = torch::empty({ test2->nels });
											//torch::nn::init::xavier_uniform_(fkgr);
											//test2->add_grad(runlr, fkgr);
										//	if (!didon) {
										//		fwdhlbl2.zero_();
										//	}
										//	else {
										//		torch::nn::init::xavier_uniform_(fwdhlbl2);
										//	}

										//	didon = !didon;
										//fwdhlbl2.copy_(fwdhlblout.contiguous().clone());
										//	runlr = 0.00001;
										//}
										//lbfgsb_options.eps_f = lssdif * 1e-10;

										//itesrtrain += 1;
										std::stringstream ss;
										ss << "waschanged " << waschanged << std::endl;
										ss << "loss " << loss2 << std::endl;
										ss << "itesrtrain " << itesrtrain << std::endl;
										if (dobetr) {
											ss << "totrainl " << totrainlm << std::endl;
											//ss << "rfgrid " << rfgrid << std::endl;
											//ss << "rfgrdif " << (totrainllst - rfgrid).abs() << std::endl;
											ss << "tolrnll2 " << tolrnl52m << std::endl;
											ss << "abvsgrids " << abvsgrids << std::endl;
											//ss << "abvsgridsvals " << abvsgridsvals << std::endl;
											if (reswillwino.defined()) {
												ss << "reswillwino " << reswillwino << std::endl;
												ss << "reswillwinom " << reswillwino.mean() << std::endl;
											}
											ss << "vbal " << vbal << std::endl;
											ss << "rrvbal " << rrbalv << std::endl;
											//if (itesrwin)
											ss << "rrbal " << (rrbal / 100000000.) << std::endl;
											ss << "rrvbalmin " << rrbalminv << std::endl;
											ss << "rrvbalmax " << rrbalmaxv << std::endl;
											ss << "betsitesrmade400g " << betsitesrmade400g << std::endl;
										}
										else {
											itesrtrain += 1;
											//	ss << "lssdif " << lssdif << std::endl;

										}


										orig_buf->sputn(ss.str().c_str(), ss.str().size());
										//torch::save(test2, "test2.pt");

									} while (1);
								}
								//if (trainedb)
								//	fwdhlbl.copy_(fwdhlblout.contiguous().clone());
								lssdifovg = std::abs(losslstgorig - loss2.item().toFloat());
								{
									std::stringstream ss;
									ss << "lssdifovg " << lssdifovg << std::endl;

									orig_buf->sputn(ss.str().c_str(), ss.str().size());
								}
								bool emti = (lssdifovg < 1e-5 && itesrtrain > 1) && !(loss2.item().toFloat() < 0.5);
								itesrempty += emti;
								if (itesrempty > 0)
								{
									std::stringstream ss;
									ss << "itesrempty " << itesrempty << std::endl;

									orig_buf->sputn(ss.str().c_str(), ss.str().size());
								}
								if (!emti) {
									itesrempty = 0;
									//fwdhlbl.zero_();
								}

								/*auto lstarriter = lstarrnet.begin();
								Net testsmpl;
								testt->ema_update(0, *testsmpl, 50, [&lstarriter](torch::Tensor& w, const torch::Tensor& w1, double decay) {
									//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
									//w.set_data(w1.data());
									//w.set_data(w1.data());
									//*lstarriter = torch::vstack({ *lstarriter, w.data().clone().detach().cuda().unsqueeze(0) });
									(*lstarriter)[0] = w1.data().clone().detach().cuda();
									(*lstarriter) = torch::roll(*lstarriter, 1, 0).cuda();
									w.set_data(torch::sum(*lstarriter, 0).cuda());
									lstarriter += 1;

									//w.set_data(w1.data());
									//w.set_data(w.data() / torch::mean(w.data()));
									//w.set_data(torch::pow(w.data() * (2. / w.data().norm()), (itesr * 2)));
									//torch::nn::LayerNorm()()
									//w.set_data(w.data() * ((10 ) / w.data().norm()));
									//w.set_data(w.data().fmod(w1.data()));

									//if (!dirwlb)
									//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
									//if (shrink)
									//w.set_data(w.data().uniform_(-1, 1));
									//w.clip
									//std::stringstream ss;
						//ss << "rrbal: " << rrbal << std::endl;
						//ss << "triggered at: " << res1[16] << std::endl;
						//ss << torch::mean(w) << std::endl;
									//gr3 = (torch::mean(w) >= 3.0).item().toBool();
									//w.set_data(-w.data());

						//orig_buf->sputn(ss.str().c_str(), ss.str().size());
									}, [&lstarriter](torch::Tensor& w, const torch::Tensor& w1, double decay) {
										//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
										//w.set_data(w1.data());
										//(*lstarriter)[0] = w.data().clone().detach().cuda();
										//(*lstarriter) = torch::roll((*lstarriter), 1);
										//*lstarriter = torch::vstack({ *lstarriter, w.data().clone().detach().cuda().unsqueeze(0) });
										(*lstarriter)[0] = w1.data().clone().detach().cuda();
										(*lstarriter) = torch::roll(*lstarriter, 1, 0).cuda();
										w.set_data(torch::sum(*lstarriter, 0).cuda());
										lstarriter += 1;
										//w.set_data(w1.data());

										//w.set_data(torch::pow(w.data() * (2. / w.data().norm()), 2.));
										//w.set_data((w.data() / (w.data().mean() * w.data().norm()) / itesr));
										//w.set_data(w.data().fmod(w1.data()));
										//if (!dirwlb)
										//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
										//w.set_data(-w.data());
										//if (shrink)
										//w.set_data(w.data() - (0.001 * (1) * (1)) * w.data());
										});

									/*testt->ema_update(0, *testt, !dirw ? 0.01 : -0.01, [](torch::Tensor& w, const torch::Tensor& w1, double decay) {
										//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
										//w.set_data(w.data().fmod(w1.data()));
										//w.set_data(w.data().fmod(w1.data()));
										w.set_data(w1.data() * (2. / w1.data().norm()));
										//if (!dirwlal)
										//	dirwlal = torch::any(w.data() > 2.).item().toBool() || torch::any(w.data() < -2.).item().toBool();
										//if (!dirwlal)
										//	dirwlal = torch::any(w.data() > 2.).item().toBool() || torch::any(w.data() < -2.).item().toBool();
										////if (!dirwlb)
										//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
										//if (shrink)
										//w.set_data(w.data().uniform_(-1, 1));
										//w.clip
										//std::stringstream ss;
							//ss << "rrbal: " << rrbal << std::endl;
							//ss << "triggered at: " << res1[16] << std::endl;
							//ss << torch::mean(w) << std::endl;
										//gr3 = (torch::mean(w) >= 3.0).item().toBool();
										//w.set_data(-w.data());

							//orig_buf->sputn(ss.str().c_str(), ss.str().size());
										}, [](torch::Tensor& w, const torch::Tensor& w1, double decay) {
											//auto ratio = torch::mean(w.data() / w1.data()).item().toFloat();
											//w.set_data(w.data().fmod(w1.data()));
											//w.set_data(w.data().fmod(w1.data()));
											w.set_data(w1.data() * (2. / w1.data().norm()));
											//if (!dirwlal)
											//	dirwlal = torch::any(w.data() > 2.).item().toBool() || torch::any(w.data() < -2.).item().toBool();
											//if (!dirwlal)
											//	dirwlal = torch::any(w.data() > 2.).item().toBool() || torch::any(w.data() < -2.).item().toBool();
											//if (!dirwlb)
											//dirwlb = dirwlb && torch::all(w.data() < 1.).item().toBool() || torch::all(w.data() > -1.).item().toBool();
											//w.set_data(-w.data());
											//if (shrink)
											//w.set_data(w.data() - (0.001 * (1) * (1)) * w.data());
											});*/
											//if (losslstg < 0.5) {

											//
											//else {
											//	loss2.backward();
												//torch::nn::utils::clip_grad_norm_(testt->parameters(), 0.05);
												//if (itesr > 0)
											//	optim[2]->step();
											//}
								itesr = -1;
								//betsitesrmade = -1;

								//toinfermi = torch::roll(toinfermi, 1, 1);
								//for (int i = 0; i < 1000; ++i) {
								//	toinfermi = torch::roll(toinfermi, 1, 1);
								//	dobetlm(0, prabs, 0, true);
								//}
								//fwdhlbl = fwdhlblout.clone().detach();


								//toinfermi = torch::roll(toinfermi, 1, 1);
								//betsitesrmade += 1;
								//totrainl = totrainl / torch::max(totrainl);
							}
							//totrainlr = totrainl.clone().detach();

							//bool haswon = vbal2 > 0;
							//auto& totrainlref = (haswon ? totrainlw : totrainll);
							//totrainlref[0][0][0] = (float)predrightg;
							//totrainl = (totrainlw - totrainll).abs().clone().detach();//totrainlref.clone().detach();
							//totrainlref = torch::roll(totrainlref, -1);
							totrainl = torch::roll(totrainl, 1);
							//torch::save(totrainll, "totrainll.pt");
							//torch::save(totrainlw, "totrainlw.pt");
							//torch::save(totrainlm, "totrainlm.pt");
							//torch::save(tolrnl52m, "tolrnl52m.pt");
							//torch::save(test2, "mdl_bsa.pt");
							//TerminateProcess(GetCurrentProcess(), 0);
							//std::exit(0);
							//if (haswon) {
							//	fwdhlbl2 = fwdhlbl2w;
							//}
							//else {
							//	fwdhlbl2 = fwdhlbl2l;
							//}

							//totrainl = toinferm3.clone().detach();
							//totrainl = torch::roll(toinferm3, (vbal2 > 0 ? 1 : -1) )

							//(void)test2->forward(totrainl, fwdhlbl, &fwdhlbl, 0);
							//fwdhlblout = fwdhlbl.clone().detach();
							//totrainlr = toinferm3.clone().detach();

							betsitesrmade += 1;

							//if ((betsitesrmade % 20) == 0)
							//	betindex += 1;
							//betindex %= 20;

							dobetr = 1;
							//trainedb = itesrtrain > 10;
							}
							//if (loss2.item().toFloat() < 0.1)
							//totrainl = totraintol.clone().detach().cuda();//((totraintol > totrainl).toType(c10::ScalarType::Float)).clone().detach().cuda();
							//tolrnl2[modes] = ((toinferm3l[modes] > toinfero[modes]).toType(c10::ScalarType::Float)).clone().detach().cuda();

						;



						//static bool started = false;



						static bool started = false;

						if (!started && trainedb3) {
							//++thrcount;
							/*res = torch::squeeze(test2.forward(inp.reshape({1, 20, 1}), fwdhl));

							//trainhl = test->layers[0].rnnresh;
							auto lstinpslc1 = lstinp.slice(0, 0, 10);
							auto lstinpslc2 = lstinp.slice(0, 10, 20);
							auto lstinpsum1 = lstinpslc1.sum().item().toFloat();
							auto lstinpsum2 = lstinpslc2.sum().item().toFloat();
							//std::cout << "sum 1" << lstinpsum1;
							//std::cout << "sum 2" << lstinpsum1;
							auto inpmax = torch::argmax(inp).item().toInt();
							resin = res.detach().clone();

							auto lstinpslc1res = resin.slice(0, 0, 10);
							auto lstinpslc2res = resin.slice(0, 10, 20);
							//auto lstinpslc3res = resin.slice(0, 20, 30);
							auto lstinpsum1res = lstinpslc1res.sum().item().toFloat();
							auto lstinpsum2res = lstinpslc2res.sum().item().toFloat();
							auto maxlst = torch::argmax(lstinp).item().toInt();

							auto maxpred = torch::argmax(resin).item().toInt();
							auto minpred = torch::argmin(resin).item().toInt();
							bool cond = (((double)rnd / UINT64_MAX) > 0.5);
							//static std::atomic_bool above;
							//static float aboves;
							bool above = lstinpsum1res > lstinpsum2res;//(lstinpsum1res > lstinpsum2res);*/
							//itersy = 0;
							//std::stringstream ss;
							//for (int i = 0; i < 20; ++i)
							//	ss << std::format("{:x}", (unsigned short)(res[i].item().toFloat() * 0xFFFF));
							//std::stringstream ss;
							//ss << std::setfill('0') << std::setw(30) << std::format("{:x}{:x}", rnd, rndtest);
							//auto str = ss.str();
							//str.resize(30);
							/* << std::format("{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}{:x}",
								resin[0] * 0xFFFF, resin[1] * 0xFFFF, resin[2] * 0xFFFF, resin[3] * 0xFFFF, resin[4] * 0xFFFF, resin[5] * 0xFFFF,
								resin[6] * 0xFFFF, resin[7] * 0xFFFF, resin[8] * 0xFFFF, resin[9] * 0xFFFF, resin[10] * 0xFFFF, resin[11] * 0xFFFF,
								resin[12] * 0xFFFF, resin[13] * 0xFFFF, resin[14] * 0xFFFF, resin[15] * 0xFFFF, resin[16] * 0xFFFF, resin[17] * 0xFFFF,
								resin[18] * 0xFFFF, resin[19] * 0xFFFF);*/
								//auto str = ss.str();
								//str.resize(30);
							std::thread(dobetl2).detach();
							started = true;
							//totrain = torch::tensor({ 0 }).cuda();
							//totrainto = torch::tensor({ 0 }).cuda();
							///started = true;
							//dobetl();
							//itersz = 0;
						}


#if 1
						D3D12_VIEWPORT m_viewport{ 0, 0, 640, 480, 0.0, 1.0 };
						D3D12_RECT m_scissorRect{ 0, 0, 640, 480 };

						// Command list allocators can only be reset when the associated 
						 // command lists have finished execution on the GPU; apps should use 
						 // fences to determine GPU execution progress.
						m_commandAllocator->Reset();

						// However, when ExecuteCommandList() is called on a particular command 
						// list, that command list can then be reset at any time and must be before 
						// re-recording.
						m_commandList->Reset(m_commandAllocator.Get(), m_pipelineState.Get());

						// Set necessary state.
						m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());
						m_commandList->RSSetViewports(1, &m_viewport);
						m_commandList->RSSetScissorRects(1, &m_scissorRect);

						m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

						// Indicate that the back buffer will be used as a render target.
						D3D12_RESOURCE_BARRIER barrier = { .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
							.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
							.Transition = {.pResource = m_renderTargets[m_frameIndex].Get(),
							.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
							.StateBefore = D3D12_RESOURCE_STATE_PRESENT, .StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET,
						} };
						//auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
						m_commandList->ResourceBarrier(1, &barrier);

						D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = { .ptr = m_rtvHeap->GetCPUDescriptorHandleForHeapStart().ptr + m_frameIndex * m_rtvDescriptorSize };
						m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

						// Record commands.


						m_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
						m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
						//m_vertexBufferView.BufferLocation = (size_t)res.data_ptr();
						/*HANDLE sharedHandle;
						SECURITY_ATTRIBUTES windowsSecurityAttributes;
						LPCWSTR name = NULL;
						m_device->CreateSharedHandle(
							m_vertexBuffer.Get(), &windowsSecurityAttributes, GENERIC_ALL, name,
							&sharedHandle);
						D3D12_RESOURCE_ALLOCATION_INFO d3d12ResourceAllocationInfo;
						d3d12ResourceAllocationInfo = m_device->GetResourceAllocationInfo(
							1, 1, &desc);
						size_t actualSize = d3d12ResourceAllocationInfo.SizeInBytes;
						size_t alignment = d3d12ResourceAllocationInfo.Alignment;

						cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
						memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));

						externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
						externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
						externalMemoryHandleDesc.size = actualSize;
						externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

						cudaExternalMemory_t m_externalMemory;

						cudaImportExternalMemory(&m_externalMemory, &externalMemoryHandleDesc);
						//CloseHandle(sharedHandle);

						cudaExternalMemoryBufferDesc externalMemoryBufferDesc;
						memset(&externalMemoryBufferDesc, 0, sizeof(externalMemoryBufferDesc));
						externalMemoryBufferDesc.offset = 0;
						externalMemoryBufferDesc.size = vertexBufferSize;
						externalMemoryBufferDesc.flags = 0;

						void* m_cudaDevVertptr;

						cudaExternalMemoryGetMappedBuffer(
								&m_cudaDevVertptr, m_externalMemory, &externalMemoryBufferDesc);
						cudaMemcpy(m_cudaDevVertptr, res.mutable_data_ptr(), sizeof(float[100]), cudaMemcpyDeviceToDevice);
						std::cout << std::hex << m_vertexBufferView.BufferLocation << std::endl;
						std::cout << m_cudaDevVertptr << std::endl;
						std::cout << res.data_ptr() << std::endl;
						//cudaMemcpy((void*)m_vertexBufferView.BufferLocation, res.data_ptr(), sizeof(float[100 * 100]), cudaMemcpyDeviceToDevice);
						cudaDestroyExternalMemory(m_externalMemory);
						//cudaFree(m_cudaDevVertptr);
						CloseHandle(sharedHandle);
						//m_vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin));
						//memcpy(pVertexDataBegin, res.detach().cuda().data_ptr(), sizeof(float[100 * 100]));
						//m_vertexBuffer->Unmap(0, nullptr);
						//m_vertexBufferView.BufferLocation = (size_t)res.data_ptr();
						//cudaMemcpy((VOID*)m_vertexBufferView.BufferLocation, res.detach().cuda().data_ptr(), sizeof(float[100 * 100]), cudaMemcpyDeviceToDevice);
						//cudaMemcpyDeviceToDevice()* /
						//m_vertexBufferView.BufferLocation = (size_t)res.data_ptr(); */
						// Create the vertex buffer.
						//auto rescpu = res.detach().cuda();

						{

							// Define the geometry for a triangle.




							//static bool __inittriangleVertices = []() {

								//for (int i = 0; i < 50; ++i)
							//if (condt)

							//}
							//}
							//else 
							/* {
								for (int i = 0; i < 50; ++i) {
									triangleVertices[i].vals[7] += 1.0;
								}
							}*/

							// Note: using upload heaps to transfer static data like vert buffers is not 
							// recommended. Every time the GPU needs it, the upload heap will be marshalled 
							// over. Please read up on Default Heap usage. An upload heap is used here for 
							// code simplicity and because there are very few verts to actually transfer.
							D3D12_HEAP_PROPERTIES heapProps{ .Type = D3D12_HEAP_TYPE_UPLOAD,
								.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
								.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
								.CreationNodeMask = 1,
								.VisibleNodeMask = 1 };
							//auto desc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);
							if (!m_vertexBuffer)
								m_device->CreateCommittedResource(
									&heapProps,
									D3D12_HEAP_FLAG_NONE,
									&desc,
									D3D12_RESOURCE_STATE_GENERIC_READ,
									nullptr,
									IID_PPV_ARGS(&m_vertexBuffer));
							if (!m_indexBuffer) {
								m_device->CreateCommittedResource(
									&heapProps,
									D3D12_HEAP_FLAG_NONE,
									&desc,
									D3D12_RESOURCE_STATE_GENERIC_READ,
									nullptr,
									IID_PPV_ARGS(&m_indexBuffer));
							}

							{
								UINT8* pVertexDataBegin;
								D3D12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
								m_indexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin));
								memcpy(pVertexDataBegin, triangleIndices, indexBufferSize);
								m_indexBuffer->Unmap(0, nullptr);
							}

							// Copy the triangle data to the vertex buffer.
							UINT8* pVertexDataBegin;
							D3D12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
							m_vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin));
							memcpy(pVertexDataBegin, triangleVertices, vertexBufferSize);
							m_vertexBuffer->Unmap(0, nullptr);

							// Initialize the vertex buffer view.
							m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
							m_vertexBufferView.StrideInBytes = 32;
							m_vertexBufferView.SizeInBytes = vertexBufferSize;
							m_indexBufferView.BufferLocation = m_indexBuffer->GetGPUVirtualAddress();
							m_indexBufferView.Format = DXGI_FORMAT_R16_UINT;
							m_indexBufferView.SizeInBytes = indexBufferSize;
							m_commandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);
							m_commandList->IASetIndexBuffer(&m_indexBufferView);
							m_commandList->DrawIndexedInstanced(50, 1, 0, 0, 0);
						}

						barrier = { .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
							.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
							.Transition = {.pResource = m_renderTargets[m_frameIndex].Get(),
							.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
							.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET, .StateAfter = D3D12_RESOURCE_STATE_PRESENT,
						} };

						// Indicate that the back buffer will now be used to present.
						//barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
						m_commandList->ResourceBarrier(1, &barrier);

						m_commandList->Close();

						ppCommandLists[0] = { m_commandList.Get() };
						m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

						m_swapChain->Present(1, 0);//*/
#endif
						//m_vertexBuffer->Release();
						/*if (!end) {
							lstinp = inp.detach().slice(0, 0, 10);//.cuda();
							//std::cout << lstinp << std::endl;
							auto max = torch::argmax(lstinp).item().toInt();
							lstinp = torch::zeros({ 10 });
							lstinp[max] = 1.0;
							lstinp = lstinp.cuda();
							lstres = max > 4;
							std::cout << loss << std::endl;
							loss.backward();
							optim.step();
						}*/
#if 1
						MSG msg = { };
						while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE) > 0)
						{
							TranslateMessage(&msg);
							DispatchMessage(&msg);

							/*if (msg.message != WM_KEYDOWN)
								continue;

							if (msg.wParam == VK_UP)
								toinfer = torch::roll(toinfer, -1, 0);
							else if (msg.wParam == VK_DOWN)
								toinfer = torch::roll(toinfer, 1, 0);
							else if (msg.wParam == VK_LEFT)
								toinfer = torch::roll(toinfer, -1, -1);
							else if (msg.wParam == VK_RIGHT)
								toinfer = torch::roll(toinfer, 1, -1);


							if (msg.wParam == 'W')
								tolrn = torch::roll(toinfer, -1, 0);
							else if (msg.wParam == 'S')
								tolrn = torch::roll(toinfer, 1, 0);
							else if (msg.wParam == 'A')
								tolrn = torch::roll(toinfer, -1, -1);
							else if (msg.wParam == 'D')
								tolrn = torch::roll(toinfer, 1, -1);*/
								//inrec.arrive_and_wait();
								//break;
						}
#endif
						//Sleep(INFINITE);
					}
					catch (const c10::Error& e) {
						std::cout << e.msg() << std::endl;
						//optim[optimi] = std::unique_ptr<torch::optim::Optimizer>(factories[1]());
						//return -1;
						//optim[1] = std::unique_ptr<torch::optim::Optimizer>(factories[1]());
						//optim[0] = std::unique_ptr<torch::optim::Optimizer>(factories[0]());
					}
	}
	return 0;

}