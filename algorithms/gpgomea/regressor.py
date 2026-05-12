from pygpg.sk import GPGRegressor as GPGR
import sympy as sp

# Patch: sklearn 1.6+ 添加了 _doc_link_* 内部参数，会被传给 C++ 后端导致报错
_orig_create_cpp = GPGR._create_cpp_option_string
def _patched_create_cpp_option_string(self):
    orig_get = self.get_params
    def _filtered_params():
        return {k: v for k, v in orig_get().items() if not k.startswith('_')}
    self.get_params = _filtered_params
    s = _orig_create_cpp(self)
    self.get_params = orig_get
    return s
GPGR._create_cpp_option_string = _patched_create_cpp_option_string

hyper_params = [
    { # 2
     'd' : (4,), 'rci' : (0.0, ), 'cmp' : (0.0, 0.1),
    },
    { # 2
     'd' : (5,), 'rci' : (0.0, 0.1,),
    },
    { # 2
     'd' : (6,), 'rci' : (0.0, 0.1),  'no_univ_exc_leaves_fos' : (True,),
    },
]

est = GPGR(
    t=2*60*60, # max time limit set to 2 hours
    g=-1, # no generational limit
    e=499500, finetune_max_evals=500, # max evaluations set to 500,000 (499,000 for search, 500 for fine-tuning)
    finetune=True, # enable fine-tuning
    tour=4, d=4, # tournament size & depth of the GP tree
    pop=1024, disable_ims=True, # fixed population size of 1024
    feat_sel=20,  # feature selection
    no_univ_exc_leaves_fos=False, no_large_fos=True, # settings for the FOS (i.e. GOMEA's crossover mask)
    bs=2048,      # max batch size
    fset='+,-,*,/,log,sqrt,sin,cos', # operators to use
    cmp=0.0, # coefficient mutation probability
    rci=0.0, # relative complexity importance (over fitness, for final model selection)
    random_state=0, # seed
    )

def model(estimator):
    return str(estimator.model)

eval_kwargs = {}
