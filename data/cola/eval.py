import glob
import argparse
from sklearn.metrics import matthews_corrcoef, accuracy_score

def evaluate(goldDict, predDict):
	goldList =[]; predList =[]
	for key,value in goldDict.items():
		goldList.append(value)
		predList.append(predDict.get(key))
	assert len(goldList)==len(predList)	
	print('Matthew\'s Corr : ', matthews_corrcoef(goldList, predList))
	print('Accuracy : ', accuracy_score(goldList, predList))

def preprocessingFile(fileName):
	readDatas = open(fileName,'r').read().strip().split("\n")
	idxLabelDict = dict()
	for readData in readDatas:
		data_format = len(readData.split("\t"))
		if data_format ==3:
			idx = readData.split("\t")[0]
			label = readData.split("\t")[1]
			if idx not in idxLabelDict.keys(): idxLabelDict[idx] = [label]
			else: 
				print ("Although this idx ({}) is unique information, duplicate idx exists.\n \
						You need to check the idx of the file.".format(idx))
		else:
			print ("The evaluation data is incorrectly formatted.\n \
					You should check the corresponding evaluation data format.\n \
					The corresponding evaluation data format is \"index<tab>acceptability_label<tab>sentence\".")
	return idxLabelDict

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-p', '--predFile', required=True, help='path of prediction file for evaluation.')
	parser.add_argument('-g', '--goldFile', default = './NIKL_CoLA_gold.txt', help='path of gold file for evaluation.')
	args=parser.parse_args()

	# gold File preprocessing
	goldIdxLabelDict = preprocessingFile(args.goldFile)
	# prediction File preprocessing	
	predIdxLabelDict= preprocessingFile(args.predFile)

	print('Evaluation data size = {}\n'.format(len(goldIdxLabelDict.keys())))
	evaluate(goldIdxLabelDict, predIdxLabelDict)
