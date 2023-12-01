#!/usr/bin/env python
# coding: utf-8

# # Basic Imports

# In[1]:


import numpy as np
import pickle
import os
from haversine import haversine
from math import floor
from collections import defaultdict
import random
import gzip
from tqdm import tqdm 
import pandas as pd
import time
import multiprocess as mp
from tqdm import tqdm


# # Setting up initial variables

# In[2]:


# Cleaned endomondo dataset.... (simply cleaned for cases with abnormal readings..)
data_path = "endomondoHR_proper.json"

# Attribute embedding features....
attrFeatures = ['userId', 'sport', 'gender']
    
# Percentage of data splits....
trainValidTestSplit = [0.8, 0.1, 0.1]
targetAtts = ["derived_speed"]
    
# Time sequnce inputs.... (contextual features...)
inputAtts = ["distance", "altitude", "time_elapsed"]

# splits file
trainValidTestFN = "./endomondoHR_proper_temporal_dataset.pkl"


# # Data reading helper functions

# In[3]:


# dataset already been preprocessed
def parse(path):
    if 'gz' in path:
        f = gzip.open(path, 'rb')
        for l in f.readlines():
            yield(eval(l.decode('ascii')))
    else:
        f = open(path, 'rb')
        for l in f.readlines():
            yield(eval(l))

def process(line):
    return eval(line)


# # Pre-processing classes

# In[4]:


class dataInterpreter(object):
    
    # Basic constructor 
    def __init__(self, 
                 inputAtts, 
                 targetAtts=['derived_speed'], 
                 includeUser=True, 
                 includeSport=False, 
                 includeGender=False, 
                 includeTemporal=False, 
                 fn="endomondoHR_proper.json", 
                 scaleVals=True, 
                 trimmed_workout_len=450, 
                 scaleTargets="scaleVals", 
                 trainValidTestSplit=[.8,.1,.1], 
                 zMultiple=5, 
                 trainValidTestFN=None):

        # Loading the proper dataset...   
        self.filename = fn
        self.data_path = "./"
        self.metaDataFn = fn.split(".")[0] + "_metaData.pkl"

        # z-normalizing the features...
        self.scaleVals = scaleVals

        # fixing the number of timesteps for the workout 
        self.trimmed_workout_len = trimmed_workout_len
        
        # set to false when scale only inputs
        if scaleTargets == "scaleVals":
            scaleTargets = scaleVals
        self.scale_targets = scaleTargets 
        
        # window size = 1 means no smoothing (using contextual information from adjacent points...)
        self.smooth_window = 1 
        self.perform_target_smoothing = True

        # Nominal features with fixed set of values....
        self.isNominal = ['gender', 'sport']

        # Features that are derived....
        self.isDerived = ['time_elapsed', 'distance', 'derived_speed', 'since_begin', 'since_last']

        # sequence features...
        self.isSequence = ['altitude', 'heart_rate', 'latitude', 'longitude'] + self.isDerived
        
        # setting up all other feature list 
        self.inputAtts = inputAtts
        self.includeUser = includeUser
        self.includeSport = includeSport
        self.includeGender = includeGender
        self.includeTemporal = includeTemporal

        # Target variable....
        self.targetAtts = ["tar_" + tAtt for tAtt in targetAtts]

        print("input attributes: ", self.inputAtts)
        print("target attributes: ", self.targetAtts)

        # Setting up data split parameters....
        self.trainValidTestSplit = trainValidTestSplit
        self.trainValidTestFN = trainValidTestFN
        self.zMultiple = zMultiple

    # Main function to create new features.....
    def preprocess_data(self):
 
        self.original_data_path = self.data_path + "/" + self.filename 
        self.processed_path = self.data_path + "/processed_" + self.filename.split(".")[0] + ".npy"

        # load index for train/valid/test (already created from the splits notebook)
        self.loadTrainValidTest()
        
        # Checking if already a processed version of the dataset exists or not
        if os.path.exists(self.processed_path):
            print("{} exists".format(self.processed_path))
            self.original_data = np.load(self.processed_path)[0]
            self.map_workout_id()
            
        else:
            
            # not preprocessed yet, load raw data and preprocess
            print("load original data")
            pool = mp.Pool(5) 
            with open(self.original_data_path, 'r') as f:
                self.original_data = pool.map(process, f)
            pool.close()
            pool.join()
            
            print("Original dataset with {} points loaded".format(len(self.original_data)))
            
            # Create a new index map of workout ids
            self.map_workout_id()
            print("Updated mapping workout id..")
            print("##############")
            
            # Create new derived features
            self.derive_data()
            print("Created new features..")
            print("##############")
            
            # build meta
            self.buildMetaData()
            print("Meta Data file generated....")
            print("##############")
            
            # scale data
            self.scale_data()
            print("Data is scaled....")
            print("##############")
        
        self.load_meta()
        self.input_dim = len(self.inputAtts)
        self.output_dim = len(self.targetAtts) # each continuous target has dimension 1, so total length = total dimension
      
    def map_workout_id(self):
        
        # convert workout id (session ID) to original data row number
        self.idxMap = defaultdict(int)
        
        # Id to numeric index mapping..... (mapping each session to a number)
        for idx, d in enumerate(self.original_data):  
            self.idxMap[d['id']] = idx

        # splitting the dataset based on session IDS
        self.trainingSet = [self.idxMap[wid] for wid in self.trainingSet]
        self.validationSet = [self.idxMap[wid] for wid in self.validationSet]
        self.testSet = [self.idxMap[wid] for wid in self.testSet]
        
        # update workout id to index in original_data
        contextMap2 = {} 
        
        # previous workout information...
        for wid in self.contextMap:
            context = self.contextMap[wid]
            contextMap2[self.idxMap[wid]] = (context[0], context[1], [self.idxMap[wid] for wid in context[2]])
        
        # updating context map
        self.contextMap = contextMap2 
    
    
    def load_meta(self): 
        self.buildMetaData() 

    def randomizeDataOrder(self, dataIndices):
        return np.random.permutation(dataIndices)

    
    def generateByIdx(self, index):
        targetAtts = self.targetAtts
        inputAtts = self.inputAtts

        inputDataDim = self.input_dim
        targetDataDim = self.output_dim
        
        current_input = self.original_data[index] 
        workoutid = current_input['id']

        inputs = np.zeros([inputDataDim, self.trimmed_workout_len])
        outputs = np.zeros([targetDataDim, self.trimmed_workout_len])
        for idx, att in enumerate(inputAtts):
            if att == 'time_elapsed':
                inputs[idx, :] = np.ones([1, self.trimmed_workout_len]) * current_input[att][self.trimmed_workout_len-1] # given the total workout length
            else:
                inputs[idx, :] = current_input[att][:self.trimmed_workout_len]
        for att in targetAtts:
            outputs[0, :] = current_input[att][:self.trimmed_workout_len]
        inputs = np.transpose(inputs)
        outputs = np.transpose(outputs)

        if self.includeUser:
            user_inputs = np.ones([self.trimmed_workout_len, 1]) * self.oneHotMap['userId'][current_input['userId']]
        if self.includeSport:
            sport_inputs = np.ones([self.trimmed_workout_len, 1]) * self.oneHotMap['sport'][current_input['sport']]
        if self.includeGender:
            gender_inputs = np.ones([self.trimmed_workout_len, 1]) * self.oneHotMap['gender'][current_input['gender']]

        # build context input    
        if self.includeTemporal:
            context_idx = self.contextMap[idx][2][-1] # index of previous workouts
            context_input = self.original_data[context_idx]

            context_since_last = np.ones([1, self.trimmed_workout_len]) * self.contextMap[idx][0]
            # consider what context?
            context_inputs = np.zeros([inputDataDim, self.trimmed_workout_len])
            context_outputs = np.zeros([targetDataDim, self.trimmed_workout_len])
            for idx, att in enumerate(inputAtts):
                if att == 'time_elapsed':
                    context_inputs[idx, :] = np.ones([1, self.trimmed_workout_len]) * context_input[att][self.trimmed_workout_len-1]
                else:
                    context_inputs[idx, :] = context_input[att][:self.trimmed_workout_len]
            for att in targetAtts:
                context_outputs[0, :] = context_input[att][:self.trimmed_workout_len]
            context_input_1 = np.transpose(np.concatenate([context_inputs, context_since_last], axis=0))
            context_input_2 = np.transpose(context_outputs)

        inputs_dict = {'input':inputs}
        if self.includeUser:       
            inputs_dict['user_input'] = user_inputs
        if self.includeSport:       
            inputs_dict['sport_input'] = sport_inputs
        if self.includeGender:
            inputs_dict['gender_input'] = gender_inputs
        if self.includeTemporal:
            inputs_dict['context_input_1'] = context_input_1
            inputs_dict['context_input_2'] = context_input_2

        return (inputs_dict, outputs, workoutid)
    
    # yield input and target data
    def dataIteratorSupervised(self, trainValidTest):
        targetAtts = self.targetAtts
        inputAtts = self.inputAtts

        inputDataDim = self.input_dim
        targetDataDim = self.output_dim

        # run on train, valid or test?
        if trainValidTest == 'train':
            indices = self.trainingSet
        elif trainValidTest == 'valid':
            indices = self.validationSet
        elif trainValidTest == 'test':
            indices = self.testSet
        else:
            raise (Exception("invalid dataset type: must be 'train', 'valid', or 'test'"))

        # loop each data point
        for idx in indices:
            current_input = self.original_data[idx] 
            workoutid = current_input['id']
 
            inputs = np.zeros([inputDataDim, self.trimmed_workout_len])
            outputs = np.zeros([targetDataDim, self.trimmed_workout_len])
            for i, att in enumerate(inputAtts):
                if att == 'time_elapsed':
                    inputs[i, :] = np.ones([1, self.trimmed_workout_len]) * current_input[att][self.trimmed_workout_len-1] # given the total workout length
                else:
                    inputs[i, :] = current_input[att][:self.trimmed_workout_len]
            for att in targetAtts:
                outputs[0, :] = current_input[att][:self.trimmed_workout_len]
            inputs = np.transpose(inputs)
            outputs = np.transpose(outputs)

            if self.includeUser:
                user_inputs = np.ones([self.trimmed_workout_len, 1]) * self.oneHotMap['userId'][current_input['userId']]
            if self.includeSport:
                sport_inputs = np.ones([self.trimmed_workout_len, 1]) * self.oneHotMap['sport'][current_input['sport']]
            if self.includeGender:
                gender_inputs = np.ones([self.trimmed_workout_len, 1]) * self.oneHotMap['gender'][current_input['gender']]
   
            # build context input    
            if self.includeTemporal:
                context_idx = self.contextMap[idx][2][-1] # index of previous workouts
                context_input = self.original_data[context_idx]

                context_since_last = np.ones([1, self.trimmed_workout_len]) * self.contextMap[idx][0]
                # consider what context?
                context_inputs = np.zeros([inputDataDim, self.trimmed_workout_len])
                context_outputs = np.zeros([targetDataDim, self.trimmed_workout_len])
                for i, att in enumerate(inputAtts):
                    if att == 'time_elapsed':
                        context_inputs[i, :] = np.ones([1, self.trimmed_workout_len]) * context_input[att][self.trimmed_workout_len-1]
                    else:
                        context_inputs[i, :] = context_input[att][:self.trimmed_workout_len]
                for att in targetAtts:
                    context_outputs[0, :] = context_input[att][:self.trimmed_workout_len]
                context_input_1 = np.transpose(np.concatenate([context_inputs, context_since_last], axis=0))
                context_input_2 = np.transpose(context_outputs)
            
            inputs_dict = {'input':inputs}
            if self.includeUser:       
                inputs_dict['user_input'] = user_inputs
            if self.includeSport:       
                inputs_dict['sport_input'] = sport_inputs
            if self.includeGender:
                inputs_dict['gender_input'] = gender_inputs
            if self.includeTemporal:
                inputs_dict['context_input_1'] = context_input_1
                inputs_dict['context_input_2'] = context_input_2
                
            yield (inputs_dict, outputs, workoutid)


    # feed into Keras' fit_generator (automatically resets)
    def generator_for_autotrain(self, batch_size, num_steps, trainValidTest):
        print("batch size = {}, num steps = {}".format(batch_size, num_steps))
        print("start new generator epoch: " + trainValidTest)

        # get the batch generator based on mode: train/valid/test
        if trainValidTest=="train":
            data_len = len(self.trainingSet)
        elif trainValidTest=="valid":
            data_len = len(self.validationSet)
        elif trainValidTest=="test":
            data_len = len(self.testSet)
        else:
            raise(ValueError("trainValidTest is not a valid value"))
        batchGen = self.dataIteratorSupervised(trainValidTest)
        epoch_size = int(data_len / batch_size)
        inputDataDim = self.input_dim
        targetDataDim = self.output_dim

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
            
        for i in range(epoch_size):
            inputs = np.zeros([batch_size, num_steps, inputDataDim])
            outputs = np.zeros([batch_size, num_steps, targetDataDim])
            workoutids = np.zeros([batch_size])

            if self.includeUser:
                user_inputs = np.zeros([batch_size, num_steps, 1])
            if self.includeSport:
                sport_inputs = np.zeros([batch_size, num_steps, 1])
            if self.includeGender:
                gender_inputs = np.zeros([batch_size, num_steps, 1])
            if self.includeTemporal:
                context_input_1 = np.zeros([batch_size, num_steps, inputDataDim + 1])
                context_input_2 = np.zeros([batch_size, num_steps, targetDataDim])

            # inputs_dict = {'input':inputs}
            inputs_dict = {'input':inputs, 'workoutid':workoutids}
            for j in range(batch_size):
                current = next(batchGen)
                inputs[j,:,:] = current[0]['input']
                outputs[j,:,:] = current[1]
                workoutids[j] = current[2]

                if self.includeUser:
                    user_inputs[j,:,:] = current[0]['user_input']
                    inputs_dict['user_input'] = user_inputs
                if self.includeSport:
                    sport_inputs[j,:,:] = current[0]['sport_input']
                    inputs_dict['sport_input'] = sport_inputs
                if self.includeGender:
                    gender_inputs[j,:,:] = current[0]['gender_input']
                    inputs_dict['gender_input'] = gender_inputs
                if self.includeTemporal:
                    context_input_1[j,:,:] = current[0]['context_input_1']
                    context_input_2[j,:,:] = current[0]['context_input_2']
                    inputs_dict['context_input_1'] = context_input_1
                    inputs_dict['context_input_2'] = context_input_2
            # yield one batch
            yield (inputs_dict, outputs)

    # please run the splits creation notebook before....
    def loadTrainValidTest(self):
        with open(self.trainValidTestFN, "rb") as f:
            self.trainingSet, self.validationSet, self.testSet, self.contextMap = pickle.load(f)
            print("train/valid/test set size = {}/{}/{}".format(len(self.trainingSet), len(self.validationSet), len(self.testSet)))
            print("******Dataset split loaded******") 
            
            
    # derive 'time_elapsed', 'distance', 'new_workout', 'derived_speed'
    def deriveData(self, att, currentDataPoint, idx):
        
        if att == 'time_elapsed':
            # Derive the time elapsed from the start
            timestamps = currentDataPoint['timestamp']
            initialTime = timestamps[0]
            # Total time elapsed from the start time (better feature to use) 
            return [x - initialTime for x in timestamps]

        elif att == 'distance':
            # Derive the distance
            lats = currentDataPoint['latitude']
            longs = currentDataPoint['longitude']
            indices = range(1, len(lats)) 
            distances = [0]
            # Gets distance traveled since last time point in kilometers
            distances.extend([haversine([lats[i-1],longs[i-1]], [lats[i],longs[i]]) for i in indices]) 
            
            # returning a sequential feature of distance...
            return distances

        # derive the new_workout list
        elif att == 'new_workout': 
            workoutLength = self.trimmed_workout_len
            
            # trimmed number of points
            newWorkout = np.zeros(workoutLength)
            
            # Add the signal at start
            newWorkout[0] = 1 
            
            return newWorkout

        elif att == 'derived_speed':
            # accesing the computed distance from lat-long
            distances = self.deriveData('distance', currentDataPoint, idx)
            timestamps = currentDataPoint['timestamp']
            indices = range(1, len(timestamps))
            # there is no time before 0th step
            times = [0]
            times.extend([timestamps[i] - timestamps[i-1] for i in indices])
            
            # 0 speed at 0th step
            derivedSpeeds = [0]
            for i in indices:
                try:
                    curr_speed = 3600 * distances[i] / times[i]
                    derivedSpeeds.append(curr_speed)
                except:
                    # handle outlier exception cases (as speed of the previous step)
                    derivedSpeeds.append(derivedSpeeds[i-1])
                    
            return derivedSpeeds

        elif att == 'since_last':
            if idx in self.contextMap:
                total_time = self.contextMap[idx][0]
            else:
                # since we always drop the first workout info of each user
                total_time = 0
            
            # feature multiplied with total time value
            return np.ones(self.trimmed_workout_len) * total_time

        elif att == 'since_begin':
            if idx in self.contextMap:
                total_time = self.contextMap[idx][1]
            else:
                total_time = 0
            return np.ones(self.trimmed_workout_len) * total_time
        else:
            # If a random un-expected derived attribute is demanded
            raise(Exception("No such derived data attribute"))

        
    # computing z-scores and multiplying them based on a scaling paramater
    # produces zero-centered data, which is important for the drop-in procedure
    def scaleData(self, data, att, zMultiple=2):
        mean, std = self.variableMeans[att], self.variableStds[att]
        diff = [d - mean for d in data]
        zScore = [d / std for d in diff] 
        return [x * zMultiple for x in zScore]

    # perform fixed-window median smoothing on a sequence
    def median_smoothing(self, seq, context_size):
        # seq is a list
        if context_size == 1: # if the window is 1, no smoothing should be applied
            return seq
        seq_len = len(seq)
        if context_size % 2 == f0:
            raise(exception("Context size must be odd for median smoothing"))

        smoothed_seq = []
        # loop through sequence and smooth each position
        for i in range(seq_len): 
            cont_diff = (context_size - 1) / 2
            context_min = int(max(0, i-cont_diff))
            context_max = int(min(seq_len, i+cont_diff))
            median_val = np.median(seq[context_min:context_max])
            smoothed_seq.append(median_val)

        return smoothed_seq
    
    def buildEncoder(self, classLabels):
        # Constructs a dictionary that maps each class label to a list 
        # where one entry in the list is 1 and the remainder are 0
        encodingLength = len(classLabels)
        encoder = {}
        mapper = {}
        for i, label in enumerate(classLabels):
            encoding = [0] * encodingLength
            encoding[i] = 1
            encoder[label] = encoding
            mapper[label] = i
        return encoder, mapper
    
    
    def writeSummaryFile(self):
        metaDataForWriting=metaDataEndomondo(self.numDataPoints, self.encodingLengths, self.oneHotEncoders,  
                                             self.oneHotMap, self.isSequence, self.isNominal, self.isDerived, 
                                             self.variableMeans, self.variableStds)
        with open(self.metaDataFn, "wb") as f:
            pickle.dump(metaDataForWriting, f)
        print("Summary file written")
        
    def loadSummaryFile(self):
        try:
            print("Loading metadata")
            with open(self.metaDataFn, "rb") as f:
                metaData = pickle.load(f)
        except:
            raise(IOError("Metadata file: " + self.metaDataFn + " not in valid pickle format"))
        self.numDataPoints = metaData.numDataPoints
        self.encodingLengths = metaData.encodingLengths
        self.oneHotEncoders = metaData.oneHotEncoders
        self.oneHotMap = metaData.oneHotMap
        self.isSequence = metaData.isSequence 
        self.isNominal = metaData.isNominal
        self.variableMeans = metaData.variableMeans
        self.variableStds = metaData.variableStds
        print("Metadata loaded")

        
    def derive_data(self):
        print("Creating Derived Data")
        
        # Looping every row of the original data
        for idx, d in tqdm(enumerate(self.original_data), position = 0,leave = True):
            # Looping across the derived features....
            for att in self.isDerived:
                # add derived attribute to each row
                self.original_data[idx][att] = self.deriveData(att, d, idx) 
            
        
    # Generate meta information about data
    def buildMetaData(self):
        if os.path.isfile(self.metaDataFn):
            self.loadSummaryFile()
        else:
            # create a new file if it does not exist....
            
            print("Building data schema")
            # other than categoriacl, all are continuous
            # categorical to one-hot: gender, sport
            # categorical to embedding: userId  
            
            # continuous attributes
            print("is sequence: {}".format(self.isSequence))  
            
            # sum of variables? 
            variableSums = defaultdict(float)
            
            # number of categories for each categorical variable
            classLabels = defaultdict(set)
        
            # consider all data to first get the max, min, etc...   
            # Looping on entire dataset
            for currData in self.original_data:
                # update number of users
                att = 'userId'
                user = currData[att]
                
                # Unique users list...
                classLabels[att].add(user)
                
                # update categorical attribute (adding all other nominal variables)
                for att in self.isNominal:
                    val  = currData[att]
                    classLabels[att].add(val)
                    
                # update continuous attribute
                for att in self.isSequence: 
                    variableSums[att] += sum(currData[att])

            # One hot encoded variables for categorical features...
            oneHotEncoders = {}
            oneHotMap = {}
            encodingLengths = {}
            for att in self.isNominal:
                oneHotEncoders[att], oneHotMap[att] = self.buildEncoder(classLabels[att]) 
                encodingLengths[att] = len(classLabels[att])
            
            # Creating a last one for userID
            att = 'userId'
            oneHotEncoders[att], oneHotMap[att] = self.buildEncoder(classLabels[att]) 
            encodingLengths[att] = 1
            
            # For sequential features...
            for att in self.isSequence:
                encodingLengths[att] = 1 # single datapoints
            
            # summary information
            self.numDataPoints=len(self.original_data)
            
            # normalize continuous: altitude, heart_rate, latitude, longitude, speed and all derives            
            self.computeMeanStd(variableSums, self.numDataPoints, self.isSequence)
    
            self.oneHotEncoders=oneHotEncoders
            self.oneHotMap = oneHotMap
            self.encodingLengths = encodingLengths
            
            #Save that summary file so that it can be used next time
            self.writeSummaryFile()

 
    def computeMeanStd(self, varSums, numDataPoints, attributes):
        print("Computing variable means and standard deviations")
        
        # assume each data point has 500 time step?! is it correct?
        numSequencePoints = numDataPoints * 500 
        
        variableMeans = {}
        for att in varSums:
            variableMeans[att] = varSums[att] / numSequencePoints
        
        varResidualSums = defaultdict(float)
        
        for numDataPoints, currData in enumerate(self.original_data):
            # loop each continuous attribute
            for att in attributes:
                dataPointArray = np.array(currData[att])
                # add to the variable running sum of squared residuals
                diff = np.subtract(dataPointArray, variableMeans[att])
                sq = np.square(diff)
                varResidualSums[att] += np.sum(sq)

        variableStds = {}
        for att in varResidualSums:
            variableStds[att] = np.sqrt(varResidualSums[att] / numSequencePoints)
            
        self.variableMeans = variableMeans
        self.variableStds = variableStds
        
        
    # scale continuous data
    def scale_data(self, scaling=True): 
        print("scale data")
        targetAtts = ['heart_rate', 'derived_speed']

        for idx, currentDataPoint in tqdm(enumerate(self.original_data),position = 0, leave = True):
            # target attribute, add to dict 
            for tAtt in targetAtts:         
                if self.perform_target_smoothing:
                    tar_data = self.median_smoothing(currentDataPoint[tAtt], self.smooth_window)
                else:
                    tar_data = currentDataPoint[tAtt]
                if self.scale_targets:
                    tar_data = self.scaleData(tar_data, tAtt, self.zMultiple) 
                self.original_data[idx]["tar_" + tAtt] = tar_data
                    
            # continuous input attribute, update dict
            for att in self.isSequence: 
                if scaling:
                    in_data = currentDataPoint[att]
                    self.original_data[idx][att] = self.scaleData(in_data, att, self.zMultiple) 
        
        for d in self.original_data:
            key = 'url'
            del d[key]
            key = 'speed'
            if key in d:
                del d[key]
        
        print("Saving data")
        
        # write to disk
        #np.save(self.processed_path,[self.original_data])


# # MetaData Creator class

# In[5]:


class metaDataEndomondo(object):
    def __init__(self, 
                 numDataPoints, 
                 encodingLengths, 
                 oneHotEncoders, 
                 oneHotMap, 
                 isSequence, 
                 isNominal, 
                 isDerived,
                 variableMeans, 
                 variableStds):
        
        self.numDataPoints = numDataPoints
        self.encodingLengths = encodingLengths
        self.oneHotEncoders = oneHotEncoders
        self.oneHotMap = oneHotMap
        self.isSequence = isSequence
        self.isNominal = isNominal
        self.isDerived = isDerived
        self.variableMeans = variableMeans
        self.variableStds = variableStds


# # Setting up the entire dataset

# In[7]:


# Reading the dataset....
endo_reader = dataInterpreter(inputAtts = inputAtts,
                              trainValidTestFN = trainValidTestFN)
    
endo_reader.preprocess_data()


# # Embeddings creation

# In[10]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, concatenate, Reshape, Concatenate, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate, Add, Dot, concatenate, add, dot, multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from keras import initializers
from keras import backend as K
import numpy as np
import random
import sys, argparse
import pandas as pd
#from data_interpreter_Keras_aux import dataInterpreter, metaDataEndomondo
import pickle
from math import floor
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import datetime
from tqdm import tqdm
import os 
from keras.optimizers import Adagrad, Adam, SGD, RMSprop


# In[11]:


# defining args
def parse():
    parser = argparse.ArgumentParser(description='context2seq-NAT')
    parser.add_argument('--patience', default=10, type=int, help='patience for early stop') # [3,5,10,20]
    parser.add_argument('--epoch', default=50, type=int, help='max epoch') # [50,100]
    parser.add_argument('--attributes', default="userId,sport,gender", help='input attributes')
    parser.add_argument('--input_attributes', default="distance,altitude,time_elapsed", help='input attributes')
    parser.add_argument('--pretrain', action='store_true', help='use pretrain model')
    parser.add_argument('--temporal', action='store_true', help='use temporal input')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size') # 
    parser.add_argument('--attr_dim', default=5, type=int, help='attribute dimension') # 
    parser.add_argument('--hidden_dim', default=64, type=int, help='rnn hidden dimension') # 
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate') # 0.001 for fine tune; 0.005 for general
    parser.add_argument('--user_reg', default=0.0, type=float, help='user attribute reg') 
    parser.add_argument('--sport_reg', default=0.01, type=float, help='sport attribute reg') 
    parser.add_argument('--gender_reg', default=0.05, type=float, help='gender attribute reg') 
    parser.add_argument('--out_reg', default=0.0, type=float, help='final output layer reg') 
    parser.add_argument('--pretrain_file', default="", help='pretrain file') 

    args = parser.parse_args()
    return args


# # Helper Functions

# In[14]:


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 


# In[15]:


class args_holder(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='context2seq-NAT')
        
        self.patience = 10
        self.epoch = 50
        self.attributes = "userId,sport,gender"
        self.input_attributes = "distance,altitude,time_elapsed"
        self.pretrain = True
        self.temporal = True
        self.batch_size = 256
        self.attr_dim = 5
        self.hidden_dim = 64
        self.lr = 0.005
        self.user_reg = 0.0
        self.sport_reg = 0.01
        self.gender_reg = 0.05
        self.out_reg = 0.0
        self.pretrain_file = ""
        
args = args_holder()   


# # Main LSTM model

# In[ ]:


class keras_endoLSTM(object):
    def __init__(self, args, newModel):
        
        # Model training from scratch....
        if newModel:
            
            # Logging & save directory.....
            self.model_save_location = "./fitrec/model_states/"
            self.summaries_dir = path + "./fitrec/logs/"
            self.data_path = "endomondoHR_proper.json"
            self.trainValidTestFN = self.data_path.split(".")[0] + "_temporal_dataset.pkl"
            
            # Training details......
            self.patience = args.patience # [3,5,10]
            self.max_epochs = args.epoch # [50,100]
            print("patience={}".format(self.patience))
            print("max_epochs={}".format(self.max_epochs))
            
            # Normalising properties & other hyper-parameters...
            self.zMultiple = 5
            self.attrFeatures = args.attributes.split(',')
            self.user_dim = args.attribute_dim
            self.sport_dim = args.attribute_dim
            self.gender_dim = args.attribute_dim
            self.includeUser = 'userId' in self.attrFeatures
            self.includeSport = 'sport' in self.attrFeatures
            self.includeGender = 'gender' in self.attrFeatures

            self.pretrain = args.pretrain
            self.includeTemporal = args.temporal
            self.pretrain_model_file_name = args.pretrain_file

            self.lr = args.lr
            print("RMSprop lr = {}".format(self.lr))
            
            # status of feature inclusion
            print("include pretrain/user/sport/gender/temporal = {}/{}/{}/{}/{}".format(self.pretrain, 
                                                                                        self.includeUser, 
                                                                                        self.includeSport, 
                                                                                        self.includeGender, 
                                                                                        self.includeTemporal))

            self.model_file_name = []
            self.model_file_name.extend(self.attrFeatures)
            if self.includeTemporal:
                self.model_file_name.append("context")
            print(self.model_file_name)

            self.trainValidTestSplit = [0.8, 0.1, 0.1]
            self.targetAtts = ['heart_rate']
            self.inputAtts = args.input_attributes
            
            self.trimmed_workout_len = 450
            self.num_steps = self.trimmed_workout_len
            self.batch_size_m = args.batch_size
            # Should the data values be scaled to their z-scores with the z-multiple?
            self.scale = True
            self.scaleTargets = False 

            self.endo_reader = dataInterpreter(self.inputAtts, 
                                               self.targetAtts, 
                                               self.includeUser,
                                               self.includeSport, 
                                               self.includeGender,
                                               self.includeTemporal,  
                                               fn=self.data_path, 
                                               scaleVals=self.scale,
                                               trimmed_workout_len=self.trimmed_workout_len, 
                                               scaleTargets=self.scaleTargets, 
                                               trainValidTestSplit=self.trainValidTestSplit, 
                                               zMultiple = self.zMultiple, 
                                               trainValidTestFN=self.trainValidTestFN)

            # preprocess data: scale
            self.endo_reader.preprocess_data()
            self.input_dim = self.endo_reader.input_dim 
            self.output_dim = self.endo_reader.output_dim 
            self.train_size = len(self.endo_reader.trainingSet)
            self.valid_size = len(self.endo_reader.validationSet)
            self.test_size = len(self.endo_reader.testSet)
            # build model
            self.model = self.build_model(args)


    def build_model(self, args):
        print('Build model...')
        self.num_users = len(self.endo_reader.oneHotMap['userId'])
        self.num_sports = len(self.endo_reader.oneHotMap['sport'])

        self.hidden_dim = args.hidden_dim
        user_reg = args.user_reg
        sport_reg = args.sport_reg
        gender_reg = args.gender_reg
        output_reg = args.output_reg
        print("user/sport/output regularizer = {}/{}/{}".format(user_reg, sport_reg, output_reg))
        
        # Embedding layer...
        inputs = Input(shape=(self.num_steps,self.input_dim), name='input')
        self.layer1_dim = self.input_dim

        if self.includeUser:
            user_inputs = Input(shape=(self.num_steps,1), name='user_input')
            User_Embedding = Embedding(input_dim=self.num_users, output_dim=self.user_dim, name='user_embedding', 
                                       embeddings_initializer=initializers.random_normal(stddev=0.01), embeddings_regularizer=l2(user_reg))
            user_embedding = User_Embedding(user_inputs)
            user_embedding = Lambda(lambda y: K.squeeze(y, 2))(user_embedding) 
            self.layer1_dim += self.user_dim

        if self.includeSport:
            sport_inputs = Input(shape=(self.num_steps,1), name='sport_input')
            Sport_Embedding = Embedding(input_dim=self.num_sports, output_dim=self.sport_dim, name='sport_embedding', 
                                   embeddings_initializer=initializers.random_normal(stddev=0.01), embeddings_regularizer=l2(sport_reg))
            sport_embedding = Sport_Embedding(sport_inputs)
            sport_embedding = Lambda(lambda y: K.squeeze(y, 2))(sport_embedding) 
            self.layer1_dim += self.sport_dim

        if self.includeGender:
            gender_inputs = Input(shape=(self.num_steps,1), name='gender_input')
            Gender_Embedding = Embedding(input_dim=self.num_users, output_dim=self.gender_dim, name='gender_embedding', 
                                       embeddings_initializer=initializers.random_normal(stddev=0.01), embeddings_regularizer=l2(gender_reg))
            gender_embedding = Gender_Embedding(gender_inputs)
            gender_embedding = Lambda(lambda y: K.squeeze(y, 2))(gender_embedding) 
            self.layer1_dim += self.gender_dim

        if self.includeTemporal:
            context_input_1 = Input(shape=(self.num_steps,self.input_dim + 1), name='context_input_1') # add 1 for since_last
            context_input_2 = Input(shape=(self.num_steps,self.output_dim), name='context_input_2')

        predict_vector = inputs
        if self.includeUser:
            predict_vector = concatenate([predict_vector, user_embedding])

        if self.includeSport:
            predict_vector = concatenate([predict_vector, sport_embedding])

        if self.includeGender:
            predict_vector = concatenate([predict_vector, gender_embedding]) 

        if self.includeTemporal:
            self.context_dim = self.hidden_dim 
            context_layer_1 = LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.num_steps, self.input_dim), name='context_layer_1')
            context_layer_2 = LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.num_steps, self.output_dim), name='context_layer_2')
            context_embedding_1 = context_layer_1(context_input_1)
            context_embedding_2 = context_layer_2(context_input_2)
            context_embedding = concatenate([context_embedding_1, context_embedding_2])
            context_embedding = Dense(self.context_dim, activation='selu', name='context_projection')(context_embedding)
            predict_vector = concatenate([context_embedding, predict_vector]) 
            self.layer1_dim += self.context_dim

        
        # Main prediction moodel....
        layer1 = LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.num_steps, self.layer1_dim), name='layer1')(predict_vector)
        dropout1 = Dropout(0.2, name='dropout1')(layer1)
        layer2 = LSTM(self.hidden_dim, return_sequences=True, name='layer2')(dropout1)
        dropout2 = Dropout(0.2, name='dropout2')(layer2)
        output = Dense(self.output_dim, name='output', kernel_regularizer=l2(output_reg))(dropout2)
        predict = Activation('selu', name='selu_activation')(output)
        #predict = Activation('linear', name='linear_activation')(output)

        inputs_array = [inputs]
        if self.includeUser:
            inputs_array.append(user_inputs)
        if self.includeSport:
            inputs_array.append(sport_inputs)
        if self.includeGender:
            inputs_array.append(gender_inputs)
        if self.includeTemporal:
            inputs_array.extend([context_input_1, context_input_2])
        model = Model(inputs=inputs_array, outputs=[predict])

        # compile model
        model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=self.lr), metrics=['mae', root_mean_squared_error])

        print("Endomodel Built!")
        model.summary()

        if self.pretrain == True:
            print("pretrain model: {}".format(self.pretrain_model_file_name))
            filepath = "./"+self.pretrain_model_file_name+"_bestValidScore"

            custom_ob = {'root_mean_squared_error':root_mean_squared_error}
            pretrain_model = keras.models.load_model(self.model_save_location+self.pretrain_model_file_name+"/"+self.pretrain_model_file_name+"_bestValidScore", custom_objects=custom_ob) 

            layer_dict = dict([(layer.name, layer) for layer in pretrain_model.layers])
            for layer_name in layer_dict:
                weights = layer_dict[layer_name].get_weights()
    
                if layer_name=='layer1':
                    weights[0] = np.vstack([weights[0],
                                            np.zeros((self.layer1_dim - self.input_dim - self.user_dim - self.sport_dim, self.hidden_dim * 4)).astype(np.float32)])

                model.get_layer(layer_name).set_weights(weights)
            del pretrain_model
        
        return model

    def run_model(self, model):

        modelRunIdentifier = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.model_file_name.append(modelRunIdentifier) # Applend a unique identifier to the filenames
        self.model_file_name = "_".join(self.model_file_name)

        self.model_save_location += self.model_file_name + "/"
        self.summaries_dir += self.model_file_name + "/"
        os.mkdir(self.model_save_location) 
        os.mkdir(self.summaries_dir) 

        best_valid_score = 9999999999
        best_epoch = 0
      
        train_steps_per_epoch = int(self.train_size * self.trimmed_workout_len / (self.num_steps * self.batch_size_m))
        valid_steps_per_epoch = int(self.valid_size * self.trimmed_workout_len / (self.num_steps * self.batch_size_m))
        test_steps_per_epoch = int(self.test_size * self.trimmed_workout_len / (self.num_steps * self.batch_size_m))

        # avoid process data in each iterator?
        for iteration in range(1, self.max_epochs):
            print()
            print('-' * 50)
            print('Iteration', iteration)

            trainDataGen = self.endo_reader.generator_for_autotrain(self.batch_size_m, self.num_steps, "train")  
            
            model_save_fn = self.model_save_location + self.model_file_name + "_epoch_"+str(iteration)
            checkpoint = ModelCheckpoint(model_save_fn, verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

            history = model.fit_generator(trainDataGen, train_steps_per_epoch, epochs=1, verbose=1, callbacks=[checkpoint])
            try:
                del history.model
                with open(self.summaries_dir+"model_history_"+self.model_file_name+"_epoch_"+str(iteration), "wb") as f:
                    pickle.dump(history, f)
                print("Model history saved")
            except:
                pass

            validDataGen = self.endo_reader.generator_for_autotrain(self.batch_size_m, self.num_steps, "valid")
            valid_score = model.evaluate_generator(validDataGen, valid_steps_per_epoch)
            print("Valid score: ", valid_score)
            try:
                with open(self.summaries_dir+"model_valid_score_"+self.model_file_name+"_epoch_"+str(iteration), "wb") as f:
                    pickle.dump(valid_score, f)
                print("Model validation score saved")
            except:
                pass

            if valid_score[0] <= best_valid_score:
                best_valid_score = valid_score[0]
                best_epoch = iteration
            elif (iteration-best_epoch < self.patience):
                pass
            else:
                print("Stopped early at epoch: " + str(iteration))
                break

        # load best model
        custom_ob = {'root_mean_squared_error':root_mean_squared_error}
        best_model = keras.models.load_model(self.model_save_location+self.model_file_name+"_epoch_"+str(best_epoch), custom_objects=custom_ob)
        best_model.save(self.model_save_location+self.model_file_name+"_bestValidScore")

        print("Best epoch: " + str(best_epoch) + " validation score: " + str(best_valid_score))

        print("Testing")
        testDataGen = self.endo_reader.generator_for_autotrain(self.batch_size_m, self.num_steps, "test")
        test_score = best_model.evaluate_generator(testDataGen, test_steps_per_epoch)
        print("Test score: " + str(test_score))
        print("Done!!!")

