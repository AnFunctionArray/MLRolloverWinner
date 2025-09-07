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
static float runlrb = 1e-6;
static float runlrb2 = 0.0005;
static float runlrb3 = 0.0005;
static float lssdifg;
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
static torch::Tensor fwdhlbl = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda().requires_grad_(false), fwdhlb2 = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda(), resall2o, resallo, reswillwino, reswillwino1, reswillwino1lst, reswillwinotr,
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
struct RotaryPositionalEmbeddingsImpl : torch::nn::Cloneable<RotaryPositionalEmbeddingsImpl> {

	int dim;
	int max_seq_len = 4096;
	int seq_len;
	int base = 10000;
	torch::Tensor theta, seq_idx, idx_theta, cache, rope_cache, xshaped, x_out;
	void reset() override {
		rope_init();
	}
	RotaryPositionalEmbeddingsImpl(int dima, int max_seq_lena = 4096, int basea = 10000) {
		base = basea;
		max_seq_len = max_seq_lena;
		dim = dima;
		reset();
	}

	void rope_init() {
		theta = torch::tensor({ 1.0 }) / (
			torch::tensor({ base }).toType(torch::ScalarType::Float).pow(torch::arange(0, dim, 2).index({ torch::indexing::Slice(torch::indexing::None, dim / 2) }).toType(torch::ScalarType::Float) / dim)
			);
		register_buffer("theta", theta);
		build_rope_cache(max_seq_len);
	}

	void build_rope_cache(int max_seq_len = 4096) {

		seq_idx = torch::arange(
			max_seq_len, torch::dtype(torch::ScalarType::Float)
		);


		idx_theta = torch::einsum("i, j -> ij", { seq_idx, theta }).toType(torch::ScalarType::Float);


		cache = torch::stack({ torch::cos(idx_theta), torch::sin(idx_theta) }, -1);
		register_buffer("cache", cache);
	}
	torch::Tensor forward(
		torch::Tensor x, std::optional<torch::Tensor> input_pos) {

		seq_len = x.size(1);

		rope_cache = (
			input_pos ? cache[input_pos.value().item()] : cache.index({ torch::indexing::Slice(torch::indexing::None, seq_len) })// if input_pos is None else self.cache[input_pos]
			);

		auto stakcsizes = x.sizes().vec();
		stakcsizes.back() /= 2;
		stakcsizes.append_range(std::vector{ 2 });

		xshaped = x.toType(torch::ScalarType::Float).reshape(stakcsizes);


		rope_cache = rope_cache.view({ -1, xshaped.size(1), 1, xshaped.size(3), 2 });

		x_out = torch::stack(
			{
				xshaped.index({ "...", 0}) * rope_cache.index({ "...", 0})
					- xshaped.index({ "...", 1}) * rope_cache.index({ "...", 1}),
					xshaped.index({ "...", 1}) * rope_cache.index({ "...", 0})
					+ xshaped.index({ "...", 0}) * rope_cache.index({ "...", 1}),
			},
			-1);


		x_out = x_out.flatten(3);
		return x_out;
	}
};

TORCH_MODULE(RotaryPositionalEmbeddings);
static int64_t rotarypos = 0;
torch::Tensor hybrid_loss(
	torch::Tensor model_output,      
	torch::Tensor sl_target,        
	torch::Tensor validation_matrix 
) {

	torch::Tensor sl_loss = torch::binary_cross_entropy_with_logits(
		model_output,
		sl_target, validation_matrix

	);

	float lambda = 0.5; 
	return sl_loss;
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
torch::Tensor abvsgrids = torch::ones({ 1, 20, 20 }, dtype(torch::ScalarType::Int)).cuda();
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


static bool traingb = false;





//static torch::Device device(torch::kCUDA, 0);

//#define REAL_BAL

static bool REAL_BAL = false;

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
static float lstlsslst = 0;
static float lstlsslstdif = 0;
static float coefminbet = 0;
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
static Vertex triangleVertices[50] = {

};


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

		for (size_t i = 0; i < parameters().size(); ++i) {

			updw(parameters()[i], mdl.parameters()[i]);
		}//);

		return;

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
		//RotaryPositionalEmbeddings embds = nullptr;
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

		std::vector<torch::Tensor> cnvpars;
		for (int i = 0; i < 1; ++i) {
			layers[i].rnn1 = torch::nn::LSTM(torch::nn::LSTMOptions(20, 20).num_layers(6).bidirectional(true));
			//layers[i].embds = RotaryPositionalEmbeddings(20);
			if (0) {

			}
			else {

				layers[i].trans4 = torch::nn::Transformer{ torch::nn::TransformerOptions().d_model(20).nhead(4).dropout(0.0) };

				int di = 0;
				for (int y = 0; y < 1; ++y) {
					int chnls = 20;
					int tchnls = 4;

					di = 0;
					for (int ii = 0; ii < 10; ++ii) {
						layers[i].cnvs1d[y * 8 + ii].cnv = torch::nn::Conv1d{ torch::nn::Conv1dOptions(20, 20, 3).dilation(1 << di).padding((1 << di) - 1) }; // .dilation(1 << di).padding((1 << di) - 1)
						register_module("cnvs1d" + std::to_string(i) + std::to_string(y * 8 + ii), layers[i].cnvs1d[y * 8 + ii].cnv);

						cnvpars.append_range(layers[i].cnvs1d[y * 8 + ii].cnv->parameters());
						if (ii % 2 == 1)
							di += 1;
					}

				}

			}



			register_module("rnn1" + std::to_string(i), layers[i].rnn1);
			//register_module("embds" + std::to_string(i), layers[i].embds);
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

	}



	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input, torch::Tensor key, torch::Tensor val, torch::Tensor hlin, torch::Tensor* hlout, int i = 0) {
		torch::Tensor inputs[2];
		auto mdli = i;
		i = 0;

		std::vector<torch::Tensor> out0, out1, outf;
		out0.resize(N_MODES);
#if 1
		out1.resize(N_MODES);
#endif

		auto batchn = input.size(0);

		auto inputl = val;

		torch::Tensor res = input, totranpsose;

		int ii = 0;
		int y = 0;

		//auto in = layers[i].embds->forward(inputl.unsqueeze(1), torch::tensor({ (int64_t)rotarypos })).squeeze(1);
		auto rnnres = layers[i].rnn1(inputl, std::tuple{ hlin[i][0].toType(c10::ScalarType::Half), hlin[i][1].toType(c10::ScalarType::Half) });

		auto rnno = (std::get<0>(rnnres));//std::get<0>(rnnres).chunk(2, -1)[0];//,

		auto hl0 = std::get<0>(std::get<1>(rnnres));
		auto hl1 = std::get<1>(std::get<1>(rnnres));
		if (hlout) {
			//*hlout = hlout->detach().clone().cuda().contiguous();
			(*hlout)[i][0].copy_((hl0).detach().clone());//.detach().clone().cuda().contiguous();
			(*hlout)[i][1].copy_((hl1).detach().clone());//.detach().clone().cuda().contiguous();
		}


		y = 0;
		inputl = rnno;
		for (ii = 0; ii < 10; ++ii) {

			inputl = (layers[i].cnvs1d[y * 8 + ii].cnv(inputl));

			inputl = torch::relu(inputl);

		}


		for (int i = 0; i < 1; ++i) {


		}

		inputl = layers[i].trans4(inputl, inputl);

		todrawres = inputl;

		auto cnvout = inputl;




		{

			out1[0] = inputl;
			out0[0] = inputl;

		}
		
		return { torch::stack(out0).cuda(), torch::stack(out1).cuda() };
	}

};
TORCH_MODULE(Net2);


static std::vector<torch::Tensor> lstarrnet;

static void clonemodelto(torch::nn::Cloneable<NetImpl>* inm, torch::nn::Cloneable<NetImpl>* outm) {

	torch::serialize::OutputArchive out{};

	std::stringstream iot;


	inm->save(out);

	out.save_to(iot);


	torch::serialize::InputArchive in{};

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
#define DEF_BET_AMNTF 0.0001//0.002//0.00002
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


		args += " " + chm;

	args += " ";
	args += std::to_string(amnt) + "";
	int resi = -1;

	requests += 1;

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



int main(int, char**) {

	cublasContext* cublas_handle;
	cublasStatus_t stat = cublasCreate(&cublas_handle);
	torch::nn::init::xavier_uniform_(fwdhlbl2);
	torch::nn::init::xavier_uniform_(fwdhlbl2w);
	torch::nn::init::xavier_uniform_(fwdhlbl2l);
	fwdhlbl2o = fwdhlbl2.clone();

	static std::array<size_t, 1000> ics = { 0 };

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

	std::cout.rdbuf(NULL);

	torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(c10::ScalarType::Half));

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

		betamntf = ((double)betamnt / 100000000);

		auto betamntt = orbal / 200;

		betamntflss = betamntfd;

		betamntft = betamntt;
#if 1
		HWND hwndcraeted{};
		hwndcraeted = GetConsoleWindow();//hwndtmp;
		if (HWND hwndtmp = CreateWindowEx(0, MAKEINTATOM(32770), "Game", WS_BORDER | WS_CAPTION | WS_POPUP, 0, 0, 640, 480, hwndcraeted, NULL, NULL, NULL)) {
			ShowWindow(hwndtmp, SW_SHOW);//MINNOACTIVE);

			hwndcraeted = hwndtmp;
		}


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


			// Note: using upload heaps to transfer static data like vert buffers is not 
			// recommended. Every time the GPU needs it, the upload heap will be marshalled 
			// over. Please read up on Default Heap usage. An upload heap is used here for 
			// code simplicity and because there are very few verts to actually transfer.
			D3D12_HEAP_PROPERTIES heapProps{ .Type = D3D12_HEAP_TYPE_DEFAULT,
					.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
					.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
					.CreationNodeMask = 1,
					.VisibleNodeMask = 1 };

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



		}
#endif


		static Net2 test2, smpl;








				

					torch::Tensor fwdhlb = torch::zeros({ 5, 2, 2 * 6, 20, 20 }).cuda();

					

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

					toinfermio2 = toinfermi.clone().detach();
					toinfermi2 = toinfermi.clone().detach();
					//torch::nn::init::xavier_uniform_(toinfero);
					//toinferm.uniform_();
					toinfero = toinferm.clone().cuda();
					totrainl = toinferm.clone().cuda();
					totrainl = torch::ones(toinferm.sizes()).cuda();
					totrainll = torch::ones(toinferm.sizes()).cuda();
					totrainlw = torch::ones(toinferm.sizes()).cuda();
					

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


					bool end = false;
					int itersend = 0;

					int testi = 0;
					std::atomic_int64_t iter = 1;
					std::atomic_int64_t curiter = 1;
					bool inited = false;
					int lstresi = -1;

					std::atomic_int iiters = 0;

					std::atomic<float> lossf;


					float mul = 1.0;
					static double lr = 0.0000002;//1. / 500000000; /// 500000000.;
					static double wd = 0.0002;//0.0000002 //0.0002
					lr = wd;
					lrf = wd;
					torch::Tensor loss = torch::empty({ 1 }).cuda();
					bool donetrain = false;
					torch::autograd::variable_list grads;


					std::unique_ptr<torch::optim::Optimizer> optim[4]{};
					std::function<torch::optim::Optimizer* ()> factories[] = {
						[&] {
						
							return nullptr;
							},
							[&] {
							return new torch::optim::NAdam(test2->param_groups);// new torch::optim::SGD(testtb->parameters(), torch::optim::SGDOptions{wd});//torch::optim::SGD(testtb->parameters(), torch::optim::SGDOptions{wd});//torch::optim::Adagrad(testtb->parameters(), torch::optim::AdagradOptions{wd});//NAdam(test->parameters(), torch::optim::NAdamOptions{1.}.weight_decay(1.));
							},
					};

					optim[0] = std::unique_ptr<torch::optim::Optimizer>(factories[0]());
					optim[1] = std::unique_ptr<torch::optim::Optimizer>(factories[1]());



					float yieldav = 0.;

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

					savestuff = [&](bool sv, const torch::nn::Module& net, const char* name) {//itersx > 4) {

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

						return;

						};




					while (true) try {
						
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



						auto dobetl2 = [&]() {
							static torch::Tensor loss2 = torch::empty({});
							static int itesrtrain = 0, itesrtraing = 0;
							static float lss2min;
							auto wcopy = [](torch::Tensor& w, const torch::Tensor& w1, double decay) {
								
								w.set_data(w1.data().clone().detach().cuda());

								};

							dobetr = 1;

							if (std::filesystem::exists("test2.pt"))
								torch::load(test2, "test2.pt");
							if (std::filesystem::exists("fwdhlbl2.pt"))
								torch::load(fwdhlbl2, "fwdhlbl2.pt");
							if (std::filesystem::exists("totrainlm.pt") && std::filesystem::exists("tolrnl52m.pt")) {
								torch::load(totrainlm, "totrainlm.pt");
								torch::load(tolrnl52m, "tolrnl52m.pt");
								dobetr = false;

							}

							totrainl.zero_();
							totrainl[0][0][0] = 1.;

							static int rightiw = 0;
							static int leftiw = -1;
							static int rightil = 0;
							static int leftil = -1;
							static int xorab = 0;

							torch::nn::init::xavier_uniform_(fwdhlbl2);

							if (itesr == 99, vbal2pa > 0, ((vbal2s[-2][0] > vbal2s[-3][0]).item().toBool()), 1) {
								for (int i = 0; i < 1; ++i) {
									float vbaldist = std::abs(vbalmin) + std::abs(vbalmax);
									float vbalceofcur = vbal + std::abs(vbalmin);

									vbalceofcur /= vbaldist;

									test2->train();

									float startinlss = 0.;
									float lssdif = 0.;
									float lstlssdif = 0.;
									float lssdiftrgt = 1e-5;
									bool zrgr = true;
									bool btrain = false;
									bool needregen = true;
									torch::Tensor rfgrid = torch::zeros({ 1, 20, 20 }, dtype(torch::ScalarType::Int)).cuda(), rfgridlst = torch::zeros({ 1, 20, 20 }).cuda(),
										rfmsk = torch::zeros({ 1, 20, 20 }).cuda(), wmsk = torch::zeros({ 1, 20, 20 }).cuda(), wmsklst = torch::zeros({ 1, 20, 20 }).cuda();

									bool optsw = false;
									betsitesrmade = 0;

									runlr = runlrb;//0.05;//0.00000166666 * loss2.item().toFloat();
									runlr2 = runlrb2;
									runlr3 = runlrb3;

									smpl->train();

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

										int ierstr = 0;

										if (itesrtrain == 0) {
											sttteed = false;

											optsw = false;
											wasminned = false;
											waschanged = 0;

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
											
										}



										if (dobetr) {
											if (betsitesrmade == 0) {

												//torch::save(test2, "test2.pt");

											}

												sttteed = false;



												auto indn = (totrainl.flatten().argmax().item().toInt() + 0) % 400;
												volatile bool wasab = reswillwino.defined() ? ((reswillwino)[0][indn] > 0.5).item().toBool() : 0;

												bool actualpred = wasab;

												float coef = reswillwino.defined() ? ((reswillwino)[0][indn]).item().toFloat() : 0.5;
												int numtrgt = std::max(200, std::min((int)(10000 * coef), 9800));
												int numtrgtprob = (wasab ? (10000 - numtrgt) : numtrgt);

												float mul = ((float)10000 / numtrgtprob) * (99. / 100.);
												volatile float ch = ((float)numtrgtprob / 10000) * 100.;
												trainedb = 0;
												prabs[modes] = (float)actualpred;

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

												
												volatile float betamntfl = DEF_BET_AMNTF;//((double)(orbal + rrbal) / 100000000.) * coefminbet;
												if (betamntfl < DEF_BET_AMNTF) {
													betamntfl = DEF_BET_AMNTF;
												}

												if (REAL_BAL) {
													fresir = dobet(wasab, betamntfl, ch, numres);
													resir = !((numres > 4999) == !!actualpred);
												}
												else {
													numres = getRoll(serverSeedl, clientSeedl, noncel);
													resir = !((numres > 4999) == !!actualpred);
													fresir = wasab ? !((numres > numtrgt - 1)) : !((numres < numtrgt));
												}


												

												if (!fresir && reswillwino.defined()) {
													itesrwin += 1;
													acccoef += reswillwino[0][indn].item().toFloat();
												}
												if (trainedb) {
													rrbalv += ((actualdir4 ? !fresir : fresir) ? 1 : -1);
													rrbalmaxv = std::max(rrbalmaxv, rrbalv).load();
													rrbalminv = std::min(rrbalminv, rrbalv).load();


												}

												float avret = !fresir ? mul - 1. : -1.;
												rrbal += betamntfl * 100000000 * avret;
											
												rmaxbal = std::max(rrbal, rmaxbal).load();
												rminbal = std::min(rrbal, rminbal).load();

												vbal2 += avret;//(!fresir ? 1 : -1);

												vbal += (!fresir ? 1 : -1);


												savestuff(false, *test2, 0);



												bool predright = !resir ? actualpred : !actualpred;

												bool needchg = false;



												betsitesrmade += 1;


												for (int y = 0; y < 1; ++y) {

												}
												rfgrid[0].flatten()[indn] = numres;
												wmsk[0].flatten()[indn] = float(vbal2);

												totrainl = torch::roll(totrainl, 1);

												bool aboveres = wasab;

												if (betsitesrmade == 400) {
													//if (trainedb) {
													//	std::exit(0);
													//}
													trainedb = false;
													btrain = totrainlm.defined();
													dobetr = !btrain;
													needregen = vbal2 < 0;
													tolrnll2 = abvsgrids.bitwise_not().clone().detach().toType(c10::ScalarType::Float) / 10000.;//.toType(c10::ScalarType::Bool).bitwise_and(rfgrid.clone().detach().toType(c10::ScalarType::Bool)).toType(c10::ScalarType::Float);
													rfmsk = ((rfgrid * wmsk) / ((rfgrid - 9999).abs() * wmsklst + 1e-6)).sigmoid();
													wmsklst = wmsk.clone().detach();
													if (vbal2 < lstvbal2,1) {
														//trainedb = betsitesrmade400g > 1;
														fwdhlbl2.copy_(fwdhlblout.contiguous());
													}
													if (1) {
														if (0)
															tolrnl52m = torch::vstack({ tolrnl52m, tolrnll2 }).cuda();
														else {

															tolrnl52m = tolrnll2.clone().detach();

														}
													}
													else {
														tolrnl52m = torch::roll(tolrnl52m, -1, 0);
														tolrnl52m[-1] = tolrnll2[0];
													}
													std::swap(lstvbal2, vbal2);
													//vbal2 = 0;
													betsitesrmade400g += 1;
												}


										}
										if (btrain) {

											auto& trgtcrl = stalled;
											test2->train();

											int localiters = 0;

											auto cb = [&](torch::Tensor fwdhlblin, torch::Tensor* fwdhlblout, Net2 mdl) {

												test2->zero_grad();

												auto indn = 400;
											beg:
												auto [resall, reswillwin] = mdl->forward(totrainlm, abvsgridslst, rfgridlst, fwdhlblin, fwdhlblout, currmdl);
												reswillwinotr = reswillwin.squeeze(0);

												if (!torch::all(reswillwinotr.isfinite()).item().toBool()) {

													std::exit(0);

													test2->train();
													goto beg;
												}

												loss2 =
													hybrid_loss(reswillwinotr, tolrnl52m.detach().toType(c10::ScalarType::Half), rfmsk);//.mean(1).flatten());


												float loss = loss2.item().toFloat();
												if (losslstg != FLT_MAX) {
													lssdif = (loss - losslstg);
													if (lssdif == 0.) {
														fsw = false;
													}
												}
												else {
													lssdif = FLT_MIN;
													startinlss = loss;
												}

#if 1
												if (std::min(loss, lss2min) != lss2min) {
													minned = true;
													wasminned = true;

													nonminned = 0;


												}
												else {

													minned = false;

												}
#endif
												lss2min = std::min(loss, lss2min);
												losslstg = loss;

												
												auto& runlrr = (!optsw, true ? runlr : runlr2);

												if (1) {
													if ((std::abs(lssdif) < runlr) != (lssdif < 0.)) {

														runlr += runlr / runlradv;
														runlr2 += runlr2 / runlradv;
														runlr3 += runlr3 / runlradv;

														zrgr = false;

													}
													else {
														runlr -= runlr / runlradv;
														runlr2 -= runlr2 / runlradv;
														runlr3 -= runlr3 / runlradv;

														zrgr = true;

													}
													optim[1]->param_groups()[0].options().set_lr(runlr);
													dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[0].options()).weight_decay(runlr * 100.);
													optim[1]->param_groups()[1].options().set_lr(runlr2);
													dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[1].options()).weight_decay(runlr2 * 100.);
													optim[1]->param_groups()[2].options().set_lr(runlr3);
													dynamic_cast<torch::optim::NAdamOptions&>(optim[1]->param_groups()[2].options()).weight_decay(runlr3 * 100.);

												}

												loss2.backward();

												localiters += 1;
												itesrtraing += 1;
												return loss2;

												};

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

											auto clos = std::bind(cb, fwdhlbl2, &fwdhlblout);

											statecu.m_funcgrad_callback = [&](
												mift* x, mift& f, mift* g,
												const cudaStream_t& stream,
												const LBFGSB_CUDA_SUMMARY<mift>& summary) {
													std::unique_lock lk(trainm);

													auto xt = torch::from_blob(x, { test2->nels }, device);
													auto gt = torch::from_blob(g, { test2->nels }, device);


													float loss;
													torch::Tensor flat_grad;
													std::stringstream ss;
													lststalled = stalled;
													stalled = !torch::any(xt != 0.).item().toBool();
													if (stalled) {

													}
													else {

													}

													auto t = runlr2;
													lbfgsb_options.step_scaling = t;

													if (0)
														test2->add_grad(t, xt.to(device));
													else {
														test2->set_grad(1., xt.to(device).toType(c10::ScalarType::Half));
														optim[1]->step();
													}

													if (0) {

													
														runlr = 0.01;// 1e-6;
														runlr2 = 1.;
													}
													if (lssdif == 0., stalled) {

														losslstgr = loss2.item().toFloat();



													}
													else {

													}

													if (summarycu.num_iteration > 0 && stalled) {

													}

													{

														torch::AutoGradMode enable_grad(true);
														loss = cb(fwdhlbl2, &fwdhlblout, test2).item<float>();

													}

													flat_grad = test2->gather_flat_grad();
													
													
														gt.copy_(flat_grad.to(device).toType(sctpt));


						

													if (1) {
														ss << loss << std::endl;

														ss << "lr " << optim[1]->param_groups()[0].options().get_lr() << std::endl;
														ss << "lr1 " << optim[1]->param_groups()[1].options().get_lr() << std::endl;
														ss << "lr2 " << optim[1]->param_groups()[2].options().get_lr() << std::endl;
														ss << "lssdif " << lssdif << std::endl;

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

													f = loss;

													return (+(lssdif == 0.));
												};

											test2->zero_grad();

											lk.unlock();
											optsw = true;
											if (1) {

												lbfgsbcuda::lbfgsbminimize(test2->nels, statecu, lbfgsb_options, x.data<mift>(), nbd.data<int>(),
													xl.data<mift>(), xu.data<mift>(), summarycu);
											}
											
											else {

												cb(fwdhlbl2, &fwdhlblout, test2).item<float>();
												
												optim[1]->step();

											}


											lk.lock();
											if ((losslstgor - loss2.item().toFloat()) != 0.) {

											}

											losslstgor = loss2.item().toFloat();

											static bool lsttrgtc = false;
											bool trgtc = (losslstgorig - loss2.item().toFloat() == 0.);

											bool optc = false;
											static int minsz = 1;
											static bool minszd = false;
											bool wasminnedl = wasminned;

											if ((std::abs(losslstgorig - loss2.item().toFloat()) < 1e-7)) {

												if (wasminned) {


												}
												
												wasminned = false;

												waschanged += 1;


											}
											else {

												waschanged = 0;

											}

											minned = 0;

											lststc = trgtc;

											losslstgorig = loss2.item().toFloat();

											lssdifg = loss2.item().toFloat() - startinlss;

											/*if ((std::abs(lssdifg) < runlrb) != (lssdifg < 0.)) {

												runlrb += runlrb / 10.;
												runlrb2 += runlrb2 / 10.;
												runlrb3 += runlrb3 / 10.;

												//zrgr = false;

											}
											else {
												runlrb -= runlrb / 10.;
												runlrb2 -= runlrb2 / 10.;
												runlrb3 -= runlrb3 / 10.;

												//zrgr = true;

											}*/

											//if (lssdifg == 0.) {
												//torch::nn::init::xavier_uniform_(fwdhlbl2);
												//rotarypos += 1;
											//	continue;
											//}
											coefminbet = std::max(0.f, loss2.item().toFloat() - lstlsslst);
											lstlsslstdif = loss2.item().toFloat() - lstlsslst;
											lstlsslst = loss2.item().toFloat();

											if (1) {

												lsttrgtc = false;
												if (1) {

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

											if (betsitesrmade == 400) {
												runlr = runlrb;//0.05;//0.00000166666 * loss2.item().toFloat();
												runlr2 = runlrb2;
												runlr3 = runlrb3;//0.00000166666 * loss2.item().toFloat();
												runlradv = 100.;
												rfgridlst = abvsgrids.bitwise_and(rfgrid.clone().detach());//rfgrid.clone().detach();

												totrainllst = test2->mem.detach().clone();
												
												auto totrainllstlst = totrainllst.clone().detach();

												abvsgridslst = abvsgrids.clone().detach();
												//if (lstlsslstdif < 0.)
												abvsgrids = abvsgrids.flatten().bitwise_and(rfgridlst.flatten().flip(0).clone().detach()).bitwise_not().flatten().flip(0).reshape_as(abvsgrids);
												rfgridlst = reswillwino1lst.defined() ? (reswillwino1 - reswillwino1lst).clone().detach() : rfgridlst.bitwise_not().toType(c10::ScalarType::Float) / 10000.;//(tolrnll2 * rfmsk).clone().detach();
#if 1
												test2->eval();
												
												auto [resallpr, reswillwinpr] = test2->forward(totrainllst, abvsgridslst, rfgridlst, fwdhlbl2, nullptr, 0);
										
												reswillwino1lst = reswillwino1.defined() ? reswillwino1.clone().detach() : reswillwino1lst;
												reswillwino1 = resallpr.squeeze(0).clone().detach();
												reswillwino = reswillwino1.sigmoid().flatten(1);


												if (!torch::all(reswillwino.isfinite()).item().toBool()) {
													std::exit(0);
												}

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

												betsitesrmade = 0;

												orand = !orand;
												itesrwin = 0;
												acccoef = 0.;

												if (loss2.item().toFloat() < 0.1, 0) {
													fwdhlbl2.copy_(fwdhlblout.contiguous().detach());
													tolrnl52m.~decltype(tolrnl52m)();
													totrainlm.~decltype(totrainlm)();
													new(&tolrnl52m)decltype(tolrnl52m)();
													new(&totrainlm)decltype(totrainlm)();
												}
												//fwdhlbl.copy_(fwdhlblout.contiguous().detach());
											}



										}

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
											ss << "rfmsk " << rfmsk << std::endl;
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
											ss << "lssdifg " << lssdifg << std::endl;	
											ss << "lstlsslstdif " << lstlsslstdif << std::endl;
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

								itesr = -1;

							}

							totrainl = torch::roll(totrainl, 1);

							betsitesrmade += 1;


							dobetr = 1;

							}

						;







						static bool started = false;

						if (!started && trainedb3) {

							std::thread(dobetl2).detach();
							started = true;

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
						{

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
#if 1
						MSG msg = { };
						while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE) > 0)
						{
							TranslateMessage(&msg);
							DispatchMessage(&msg);
						}
#endif
					}
					catch (const c10::Error& e) {
						std::cout << e.msg() << std::endl;
					}

	return 0;

}
}