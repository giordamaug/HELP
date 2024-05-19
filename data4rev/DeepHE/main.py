# Copy right (c) Xue Zhang, Weijia Xiao, and Wangxin Xiao 2020. All rights reserved.
# February 2020

# importing packages
import time
import os
import logging
import argparse
import numpy as np
seed_value = 12345
import torch
import random
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.use_deterministic_algorithms(True)


from process_data import ProcessDataset
from DNN import DNN, make_folder

logging.basicConfig(format='%(levelname)s: %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger('main')

parser = argparse.ArgumentParser(description='DeepHE implementation!')
parser.add_argument("--expName", default="Experiment", type=str, help="This is used to form the file names of results! Default is 'Experiment'.")
parser.add_argument("--fold", default=4, type=int, help="The fold of nonessential genes versus essential genes. It's max value is <= #ne/#e. Default is 4.")
parser.add_argument("--embedF", default=3, type=int, help="0 for seq features; 1 for network features; other integers for both. Default is 3.")
parser.add_argument("--data_dir", default="data/", type=str, help="The dir for feature files. Default is data/")
parser.add_argument("--trainProp", default=0.8, type=float, help="The proportion of data used for training the model. Default is 0.8.")
parser.add_argument("--repeat", default=10, type=int, help="The times you want to run the experiments. Default is 10.")
parser.add_argument("--result_dir", default="results/", type=str, help="The dir to same results.")
parser.add_argument("--numHiddenLayer", default=3, type=int, help="The number of hidden layers in the DNN model.")

def main():
    args = parser.parse_args()
    expName = args.expName
    result_dir = args.result_dir
    
    # program start time
    start_time = time.time()

    # get features and build training/testing dataset
    pdata = ProcessDataset(data_dir=args.data_dir, trainProp=args.trainProp, ExpName=args.expName, embedF=args.embedF, fold=args.fold)

    # creating file to store evaluation statistics
    make_folder(result_dir)
    
    #Create result file and save some information of the experiments to it
    fn = os.path.join(result_dir, expName + '.txt')
    fw = open(fn, 'w+')
    fw.write("Experiment Name: " + str(expName) + '\n\n')
    
    fw.write("Iteration" + "\t" + "ROC_AUC" + "\t" + "Avg. Precision" + "\t" +
                 "Sensitivity" + "\t" + "Specificity" + "\t" + "PPV" + "\t" + "Accuracy" + "\t" + "Bal. Acc." + "\t" + "MCC" +"\n")

    # dict to store evaluation statistics to calculate average values
    evaluationValueForAvg = {
        'roc_auc': 0.,
        'precision': 0.,
        'sensitivity': 0.,
        'specificity': 0.,
        'PPV': 0.,
        'accuracy': 0.,
        'BA' : 0, 
        'MCC': 0
    }
    evaluationValueForStd = {
        'roc_auc': 0.,
        'precision': 0.,
        'sensitivity': 0.,
        'specificity': 0.,
        'PPV': 0.,
        'accuracy': 0.,
        'BA' : 0, 
        'MCC': 0
    }
    evaluationValueLists = {
        'roc_auc': np.array([], dtype=float),
        'precision': np.array([], dtype=float),
        'sensitivity': np.array([], dtype=float),
        'specificity': np.array([], dtype=float),
        'PPV': np.array([], dtype=float),
        'accuracy': np.array([], dtype=float),
        'BA' : np.array([], dtype=float), 
        'MCC': np.array([], dtype=float)
    }

    #  DNN model
    if os.path.exists(os.path.join(result_dir, expName + '_True_positives.txt')):
        os.remove(os.path.join(result_dir,expName + '_True_positives.txt'))
    if os.path.exists(os.path.join(result_dir, expName + '_False_positives.txt')):
        os.remove(os.path.join(result_dir, expName + '_False_positives.txt'))
    if os.path.exists(os.path.join(result_dir, expName + '_Thresholds.txt')):
        os.remove(os.path.join(result_dir, expName + '_Thresholds.txt'))

    f_tp = open(os.path.join(result_dir, expName + '_True_positives.txt'), 'a')
    f_fp = open(os.path.join(result_dir, expName + '_False_positives.txt'), 'a')
    f_th = open(os.path.join(result_dir, expName + '_Thresholds.txt'), 'a')

    for i in range(0, args.repeat):
        print('Iteration', i)
        model = DNN(pdata, f_tp, f_fp, f_th, expName, i, result_dir=args.result_dir)
        evaluationDict = model.getEvaluationStat()

        print(evaluationDict)

        saveEvaluation(evaluationDict, fw, i + 1)

        evaluationValueForAvg['roc_auc'] += evaluationDict['roc_auc']
        evaluationValueForAvg['precision'] += evaluationDict['precision']
        evaluationValueForAvg['sensitivity'] += evaluationDict['sensitivity']
        evaluationValueForAvg['specificity'] += evaluationDict['specificity']
        evaluationValueForAvg['PPV'] += evaluationDict['PPV']
        evaluationValueForAvg['accuracy'] += evaluationDict['accuracy']
        evaluationValueForAvg['BA'] += evaluationDict['BA']
        evaluationValueForAvg['MCC'] += evaluationDict['MCC']
        evaluationValueLists['roc_auc'] = np.append(evaluationDict['roc_auc'], evaluationValueLists['roc_auc'])
        evaluationValueLists['precision'] = np.append(evaluationDict['precision'], evaluationValueLists['precision'])
        evaluationValueLists['sensitivity'] = np.append(evaluationDict['sensitivity'], evaluationValueLists['sensitivity'])
        evaluationValueLists['specificity'] = np.append(evaluationDict['specificity'], evaluationValueLists['specificity'])
        evaluationValueLists['PPV'] = np.append(evaluationDict['PPV'], evaluationValueLists['PPV'])
        evaluationValueLists['accuracy'] = np.append(evaluationDict['accuracy'], evaluationValueLists['accuracy'])
        evaluationValueLists['BA'] = np.append(evaluationDict['BA'], evaluationValueLists['BA'])
        evaluationValueLists['MCC'] = np.append(evaluationDict['MCC'], evaluationValueLists['MCC'])
       
    for value in evaluationValueForAvg:
        #evaluationValueForAvg[value] = float(evaluationValueForAvg[value]) / args.repeat
        evaluationValueForAvg[value] = np.mean(evaluationValueLists[value])
        evaluationValueForStd[value] = np.std(evaluationValueLists[value])

    #saveEvaluation(evaluationValueForAvg, fw, 'Avg.')
    for value in evaluationValueForAvg:
        fw.write(f"{value}: "+str(evaluationValueForAvg[value])+"±"+str(evaluationValueForStd[value])+"\n")

    fw.write("\n")
    fw.write("Number of training samples: " + str(evaluationDict['numTrain']) + '\n')
    fw.write("Number of validation samples: " + str(evaluationDict['numValidation']) + '\n')
    fw.write("Number of testing samples: " + str(evaluationDict['numTest']) + '\n')
    fw.write("Number of features: " + str(evaluationDict['numFeature']) + '\n')
    fw.write('Batch size:' + str(evaluationDict['batch_size']) + '\n')
    fw.write('Activation:' + str(evaluationDict['activation']) + '\n')
    fw.write('Dropout:' + str(evaluationDict['dropout']) + '\n')

    end_time = time.time()
    fw.write("Execution time: " + str(end_time - start_time) + " sec.")
    fw.close()
    # f_imp.close()

    f_tp.close()
    f_fp.close()
    f_th.close()


# writes the evaluation statistics
def saveEvaluation(evaluationDict, fw, iteration):
    fw.write(str(iteration) + "\t" + str(evaluationDict['roc_auc']) + "\t" +
                 str(evaluationDict['precision']) + '\t' + str(evaluationDict['sensitivity']) + '\t' +
                 str(evaluationDict['specificity']) + '\t' + str(evaluationDict['PPV']) + '\t' +
                 str(evaluationDict['accuracy']) + '\t' + str(evaluationDict['BA'])  + '\t' + str(evaluationDict['MCC']) + '\n')


if __name__ == "__main__":
    main()
