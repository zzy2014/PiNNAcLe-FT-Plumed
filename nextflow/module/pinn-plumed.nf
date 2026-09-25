nextflow.enable.dsl=2

process train {
  label 'pinn'
  publishDir "$params.publish/$name"

  input:
    tuple val(name), path(dataset), path(input, stageAs:'input'), val(flags)

  output:
    tuple val(name), path('model', type:'dir'), emit: model
    tuple val(name), path('pinn.log'), emit: log
    path('append.log')         // publish only
    path('weight-train.csv')   // publish only
    path('weight-eval.csv')    // publish only
    path('*.{tfr,yml}')        // publish only

  script:
    fmaxtol = params.fmaxtol
    emaxtol = params.emaxtol
    frmsetol = params.frmsetol
    ermsetol = params.ermsetol
    convert_flag = "${(flags =~ /--seed[\s,\=]\d+/)[0]}"
    train_flags = "${flags.replaceAll(/--seed[\s,\=]\d+/, '')}"
    dataset = (dataset instanceof Path) ? dataset : dataset[0].baseName+'.yml'
    """

    pinn convert $dataset -o 'train:9,eval:1' $convert_flag

python << 'EOF'
import os
from weight_utils import compute_weight
model_abs_path = os.path.abspath("input")
ds_abs_path = os.path.dirname(os.path.abspath("${dataset}"))
tolarence = {"fmaxtol":$fmaxtol, "emaxtol":$emaxtol, "frmsetol":$frmsetol, "ermsetol":$ermsetol}
compute_weight(model_abs_path, ds_abs_path, "train", tolarence)
EOF

python << 'EOF'
import os
from weight_utils import compute_weight
model_abs_path = os.path.abspath("input")
ds_abs_path = os.path.dirname(os.path.abspath("${dataset}"))
tolarence = {"fmaxtol":$fmaxtol, "emaxtol":$emaxtol, "frmsetol":$frmsetol, "ermsetol":$ermsetol}
compute_weight(model_abs_path, ds_abs_path, "eval", tolarence)
EOF

python << 'EOF'
import os
from weight_utils import add_weight_to_ds
ds_abs_path = os.path.dirname(os.path.abspath("${dataset}"))
add_weight_to_ds(ds_abs_path, "train")
EOF

python << 'EOF'
import os
from weight_utils import add_weight_to_ds
ds_abs_path = os.path.dirname(os.path.abspath("${dataset}"))
add_weight_to_ds(ds_abs_path, "eval")
EOF

    if [ ! -f $input/params.yml ];  then
        mkdir -p model; cp $input model/params.yml
    else
        cp -rL $input model
    fi
    pinn train model/params.yml --model-dir='model'\
        --train-ds='train-weight.yml' --eval-ds='eval-weight.yml'\
        $train_flags
    pinn log model/eval > pinn.log

python << 'EOF'
import os
from weight_utils import analyse_after_train
model_abs_path = os.path.abspath("model")
ds_abs_path = os.path.dirname(os.path.abspath("${dataset}"))
analyse_after_train(model_abs_path, ds_abs_path, "train")
EOF

python << 'EOF'
import os
from weight_utils import analyse_after_train
model_abs_path = os.path.abspath("model")
ds_abs_path = os.path.dirname(os.path.abspath("${dataset}"))
analyse_after_train(model_abs_path, ds_abs_path, "eval")
EOF

    """
}

process md {
  label 'pinn'
  publishDir "$params.publish/$name"

  input:
    tuple val(name), path(model,stageAs:'model*'), path(init, stageAs:'init*'), val(flags)

  output:
    tuple val(name), path('asemd.traj'), emit: traj
    tuple val(name), path('asemd.log'), emit: log

  script:
    """
python3 - <<- 'EOF'

import re
import numpy as np
import pinn
import tensorflow as tf
from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nptberendsen import NPTBerendsen
from ase.md.bussi import Bussi
# from ase.md.nvtberendsen import NVTBerendsen
from tips.bias import EnsembleBiasedCalculator
from ase.calculators.plumed import Plumed

# ------------ patch ase properties to write extra cols --------------------
from ase.calculators.calculator import all_properties
all_properties+=[f'{prop}_{extra}' for prop in ['energy', 'forces', 'stress'] for extra in ['avg','std','bias']]
# --------------------------------------------------------------------------

setup = {
    'ensemble': 'nvt', # ensemble
    'T': 330, # temperature in K
    't': 50, # time in ps
    'dt': 0.5, # timestep is fs
    'taut': 100, # thermostat damping in steps
    'taup': 1000, # barastat dampling in steps
    'log-every': 20, # log interval in steps
    'pressure': 1, # pressure in bar
    'compressibility': 4.57e-4, # compressibility in bar^{-1}
    'bias': None,
    'kb': 0,
    'sigma0': 0,
}

flags = {
    k: v for k,v in
        re.findall('--(.*?)[\\s,\\=]([^\\s]*)', "$flags")
}
setup.update(flags)
ensemble=setup['ensemble']
T=float(setup['T'])
t=float(setup['t'])*units.fs*1e3
dt=float(setup['dt'])*units.fs
taut=int(setup['taut'])
taup=int(setup['taup'])
every=int(setup['log-every'])
pressure=float(setup['pressure'])
compressibility=float(setup['compressibility'])

${(model instanceof Path) ?
"calc = pinn.get_calc('$model')" :
"""
models = ["${model.join('", "')}"]
calcs = [pinn.get_calc(model) for model in models]
if len(calcs) == 1:
    calc =  calcs[0]
else:
    calc = EnsembleBiasedCalculator(calcs,
                                    bias=setup['bias'],
                                    kb=float(setup['kb']),
                                    sigma0=float(setup['sigma0']))
"""}

atoms = read("$init")

setup_pm = [
            f"UNITS LENGTH=A TIME={1/(1000*units.fs)} ENERGY={units.mol/units.kJ}",
            f"o: GROUP ATOMS=15",
            f"h: GROUP ATOMS=192",
            f"d_OH: DISTANCES GROUPA=o GROUPB=h MIN={{BETA=100}}",
            f"metad: METAD ARG=d_OH.min PACE=500 HEIGHT=0.05 SIGMA=0.05 TEMP={T} BIASFACTOR=20 "+
            f"FILE=HILLS GRID_MIN=0.0 GRID_MAX=35", 
            f"PRINT ARG=d_OH.* FILE=colvar STRIDE=1",
          ]

# I don't know why, the Plumed 2.10 need atomic charges
atoms.set_initial_charges([0.0] * len(atoms))

plumed = Plumed(calc=calc, input=setup_pm, timestep=dt, atoms=atoms, kT=T*units.kB, use_charge=True)
atoms.set_calculator(plumed)

MaxwellBoltzmannDistribution(atoms, T*units.kB)

# 0.2 ps was selected based on Table 1 in https://arxiv.org/pdf/0803.4060
dyn = Bussi(atoms, timestep=dt, temperature_K=T, taut=dt * 400)

# running metadynamic on arrhenius, the shape of MACE predicted energy is (1,), rather than a scaler
def log_md(dyn, atoms, logfile):
    epot = float(np.asarray(atoms.get_potential_energy()).squeeze())
    ekin = float(atoms.get_kinetic_energy())
    temp = float(atoms.get_temperature())
    time_ps = dyn.get_time() / (1000 * units.fs)
    with open(logfile, "a") as f:
        f.write(
            f"{time_ps:12.4f} "
            f"{epot + ekin:16.8f} "
            f"{epot:16.8f} "
            f"{ekin:16.8f} "
            f"{temp:10.3f}\\n"
       )

# logging
with open('asemd.log', "w") as f:
    f.write("# Time[ps]       Etot[eV]          Epot[eV]          Ekin[eV]       T[K]\\n")

dyn.attach(lambda: log_md(dyn, atoms, 'asemd.log'), interval=int(every))
dyn.attach(Trajectory('asemd.traj', 'w', atoms).write, interval=int(every))
dyn.run(int(t/dt))

EOF
    """
}
