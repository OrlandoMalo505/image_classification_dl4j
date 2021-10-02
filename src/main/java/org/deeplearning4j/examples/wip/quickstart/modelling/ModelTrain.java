package org.deeplearning4j.examples.wip.quickstart.modelling;

import org.apache.commons.io.IOUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.nn.conf.BackpropType;


import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.*;
import java.util.Scanner;
import org.datavec.image.recordreader.BaseImageRecordReader;


public class ModelTrain {




    public static void main(String[] args) throws Exception {

        int numOfChannels=3;
        int width=300;
        int height=300;

        int numOfOutput=10;
        int batchSize=10;
        int numOfEpochs=10 ;

        int numOfIterations=3;
        int seed=123;
        Random random=new Random();

        File trainData= new File("C:\\Users\\User\\Downloads\\TRAINFOLDER");
        File testData= new File("C:\\Users\\User\\Downloads\\TESTFOLDER");

        FileSplit train= new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS,random);
        FileSplit test=new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,random);

        ParentPathLabelGenerator labelMaker= new ParentPathLabelGenerator();
        ImageRecordReader recordReader= new ImageRecordReader(height,width,numOfChannels,labelMaker);
        recordReader.initialize(train);

        DataSetIterator dataIter=new RecordReaderDataSetIterator(recordReader,batchSize,1,numOfOutput);


        DataNormalization scaler= new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        DataSet ds=dataIter.next();

        System.out.println(dataIter.getLabels());

        MultiLayerConfiguration configuration= new NeuralNetConfiguration.Builder()
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
            .updater(new Sgd(0.01))
            .l2(0.0005)
            .list()

            .layer(0, new ConvolutionLayer.Builder(5,5)
                .nIn(numOfChannels)
                .stride(1, 1)
                .nOut(20)
                .activation(Activation.RELU)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build())
            .layer(2, new DenseLayer.Builder()
                .activation(Activation.RELU)
                .nOut(500)
                .build())
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(numOfOutput)
                .activation(Activation.SOFTMAX)
                .build())

            .setInputType(InputType.convolutionalFlat(300,300,3))

            .build();

        MultiLayerNetwork neuralNetwork= new MultiLayerNetwork(configuration);
        neuralNetwork.init();

        neuralNetwork.setListeners(new ScoreIterationListener(1));

        for (int i=0;i<numOfEpochs;i++){
            neuralNetwork.fit(dataIter);
            Evaluation evaluation= neuralNetwork.evaluate(dataIter);
            System.out.println(evaluation.stats());
            dataIter.reset();
        }
        File locationToSave= new File("modeli.zip");
        boolean saveUpdater= false;
        ModelSerializer.writeModel(neuralNetwork,locationToSave,saveUpdater);






    }

    }

