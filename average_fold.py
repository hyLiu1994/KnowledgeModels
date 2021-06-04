import os
import sys
import json
sys.path.append("./DataProcessor/")
from public import *

TmpDir = "./data/LCData/"

def getAverage(dataset,model):

    if dataset == "bridge_algebra06":
        dataset_name = "bridge_algebra06_U_[20,1e9]_P_[F,1e9]_T_[2006-10-05 08-26-16,2007-06-20 13-36-57]_D_T_R_T"
    elif dataset == "algebra05" :
        dataset_name = "algebra05_U_[20,1e9]_P_[F,1e9]_T_[2005-08-30 09-50-35,2006-06-07 11-12-38]_D_T_R_T"
    elif(dataset == "hdu"):
        dataset_name = "hdu_U_[30,3600,0.1,T]_P_[30,1e9,F,T]_T_[2018-06-01 00-00-00,2018-11-29 00-00-00]_R_T_D_T"

    if(model == "AFM"):
        model_name = "AFM"
        file_name = "results_D_5_K_5_T_0.8_I_300_A_sa.json"
    elif(model == "DAS3H"):   
        model_name = "DAS3H"
        file_name = "results_D_5_K_5_T_0.8_I_300_A_uiswat1.json"
    elif(model == "DASH"):  
        model_name = "DASH"
        file_name = "results_D_5_K_5_T_0.8_I_300_A_uiwat2.json"
    elif(model == "IRT"):  
        model_name = "IRT"
        file_name = "results_D_F_K_5_T_0.8_I_300_A_ui.json"
    elif(model == "KTM"):  
        model_name = "KTM"
        file_name = "results_D_5_K_5_T_0.8_I_300_A_iswf.json"
    elif(model == "MIRTb"):  
        model_name = "MIRTb"
        file_name = "results_D_5_K_5_T_0.8_I_300_A_ui.json"
    elif(model == "PFA"):  
        model_name = "PFA"
        file_name = "results_D_5_K_5_T_0.8_I_300_A_swf.json"

    file_dir = os.path.join(TmpDir,dataset_name+"/"+"das3h/results_KT/"+model_name+"/"+file_name)


    with open(file_dir, 'r') as f:
        a = json.load(f)    #此时a是一个字典对象
        maxfold = 0
        minfold = 1
        results = a["results"]
        fold_0 = results["0"]["AUC"]
        fold_1 = results["1"]["AUC"]
        fold_2 = results["2"]["AUC"]
        fold_3 = results["3"]["AUC"]
        fold_4 = results["4"]["AUC"]
        average_fold = (fold_0 + fold_1 + fold_2 + fold_3 + fold_4) / 5
        maxfold = max(maxfold,fold_0)
        maxfold = max(maxfold,fold_1)
        maxfold = max(maxfold,fold_2)
        maxfold = max(maxfold,fold_3)
        maxfold = max(maxfold,fold_4)
        minfold = min(minfold,fold_0)
        minfold = min(minfold,fold_1)
        minfold = min(minfold,fold_2)
        minfold = min(minfold,fold_3)
        minfold = min(minfold,fold_4)
        # print(average_fold,maxfold,minfold)

        
        if os.path.exists("./average_AUC.json") == False:
            a = {'0':{}}
            average_value_dict = a['0']
        else:
            a = loadDict('./', 'average_AUC.json')
            num = len(a.keys())
            a[str(num)] = {}
            average_value_dict = a[str(num)]
            printDict(average_value_dict)
        average_value_dict["dataset"] = dataset
        average_value_dict["model"] = model
        average_value_dict["AUC_average"] = average_fold
        average_value_dict["AUCdiffer_max"] = maxfold - average_fold
        average_value_dict["AUCdiffer_min"] = average_fold - minfold

        saveDict(a, './','average_AUC.json')


        '''with open("./average_AUC.json","a",encoding='utf8') as f1:
            json.dump(average_value_dict,f1,indent=4,ensure_ascii=False)
            f1.write('\n')
            #f1.close()
            #print("写入文件完成...")
        '''

'''
        if os.path.exists("./average_AUC.json") == False:
            with open("./average_AUC.json","w",encoding='utf8') as f1:
                json.dump(average_value_dict,f1,indent=4,ensure_ascii=False)
                f1.write('\n')
                #f1.close()
                #print("写入文件完成...")
        elif os.path.exists("./average_AUC.json"):
            with open("./average_AUC.json","a",encoding='utf8') as f1:
                json.dump(average_value_dict,f1,indent=4,ensure_ascii=False)
                f1.write('\n')
                #f1.close()
                #print("写入文件完成...")
'''

if __name__ == "__main__":
    getAverage("bridge_algebra06","DAS3H")
    getAverage("bridge_algebra06","DASH")
    getAverage("bridge_algebra06","IRT")
    getAverage("bridge_algebra06","MIRTb")
    getAverage("bridge_algebra06","PFA")
    getAverage("bridge_algebra06","KTM")
    getAverage("bridge_algebra06","AFM")

    getAverage("algebra05","DAS3H")
    getAverage("algebra05","DASH")
    getAverage("algebra05","IRT")
    getAverage("algebra05","MIRTb")
    getAverage("algebra05","PFA")
    getAverage("algebra05","KTM")
    getAverage("algebra05","AFM")

    getAverage("hdu","DAS3H")
    getAverage("hdu","DASH")
    getAverage("hdu","IRT")
    getAverage("hdu","MIRTb")
    getAverage("hdu","PFA")
    getAverage("hdu","KTM")
    getAverage("hdu","AFM")
    
