OPTUNA OPTIMIZATION:

- PPI (no weights) + subloc attributes
- optimize on balanced_accuracy (-m 2)
- same subloc attributes for all tissues
- run on Lucia-Precision, conda env=torch-cuda11.7

$ python main.py -n KIDNEY -l ../../data/Kidney_HELP_pos.csv -p ../../data/Kidney_PPI.csv -s ./data/Sublocs_kidney.csv -t -hy -m ba -dm newmodels -dr newresults -ds studies
$ /home/maurizio/miniconda3/envs/torch-cuda11.7/bin/python main.py -n LUNG -l /home/maurizio/PLOS_CompBiology/HELP/data/Lung_HELP_pos.csv -p /home/maurizio/PLOS_CompBiology/HELP/data/Lung_PPI.csv -s /home/maurizio/PLOS_CompBiology/HELP/data4rev/EPGAT/data/Sublocs_kidney.csv -t -hy -dm newmodels -dr newresults -ds newstudies
$ /home/maurizio/miniconda3/envs/torch-cuda11.7/bin/python main.py -n BRAIN -l /home/maurizio/PLOS_CompBiology/HELP/data/Brain_HELP_pos.csv -p /home/maurizio/PLOS_CompBiology/HELP/data/Brain_PPI.csv -s /home/maurizio/PLOS_CompBiology/HELP/data4rev/EPGAT/data/Sublocs_kidney.csv -t -hy -dm newmodels -dr newresults -ds newstudies
$ /home/maurizio/miniconda3/envs/torch-cuda11.7/bin/python main.py -n HUMAN -l /home/maurizio/PLOS_CompBiology/HELP/data/PanCancer_HELP_pos.csv -p /home/maurizio/PLOS_CompBiology/HELP/data/Human_PPI.csv -s /home/maurizio/PLOS_CompBiology/HELP/data4rev/EPGAT/data/Sublocs_kidney.csv -t -hy -dm newmodels -dr newresults -ds newstudies

EXPERIMENTS

stesse configurazioni per input ma con parametri ottimizzati prima.

$ python main.py -n KIDNEY -l ../../data/Kidney_HELP_pos.csv -p ../../data/Kidney_PPI.csv -s ./data/Sublocs_kidney.csv -t
$ python main.py -n LUNG -l ../../data/Lung_HELP_pos.csv -p ../../data/Lung_PPI.csv -s ./data/Sublocs_kidney.csv -t
$ python main.py -n BRAIN -l ../../data/Brain_HELP_pos.csv -p ../../data/Brain_PPI.csv -s ./data/Sublocs_kidney.csv -t
$ python main.py -n LUNG -l ../../data/PanCancer_HELP_pos.csv -p ../../data/Human_PPI.csv -s ./data/Sublocs_kidney.csv -tSCRIPT Lung_HELP_pos

SCRIPT HELP

usage: main.py [-h] -n <name> -l <labelfile> [-e <exprfile>] [-o <orthofile>]
               [-s <sublocfile>] [-p <ppifile>] [-np] [-w] [-v] [-hy] -dm
               <modelpath> -dr <resultdir> -ds <studydir> [-t] [-r <nruns>]
               [-m <measure>]

PLOS COMPBIO EPGAT

optional arguments:
  -h, --help            show this help message and exit
  -n <name>, --name <name>
                        name of experiment
  -l <labelfile>, --labelfile <labelfile>
                        label filename
  -e <exprfile>, --exprfile <exprfile>
                        expression filename
  -o <orthofile>, --orthofile <orthofile>
                        ortho filename
  -s <sublocfile>, --sublocfile <sublocfile>
                        sublocalization filename
  -p <ppifile>, --ppifile <ppifile>
                        PPI file filename
  -np, --noppi          disable PPI usage
  -w, --weights         use weights in PPI
  -v, --verbose         enable verbosity
  -hy, --hypersearch    enable optuna hyper-search
  -dm <modelpath>, --modelpath <modelpath>
                        models path dir
  -dr <resultdir>, --resultdir <resultdir>
                        results path dir
  -ds <studydir>, --studydir <studydir>
                        study path dir
  -t, --trainmode       enable training mode
  -r <nruns>, --nruns <nruns>
                        n. of runs in experiment (default: 10)
  -m <measure>, --measure <measure>
                        measure for optuna (default: auc, choices: auc, ba,
                        mcc, sens, spec)