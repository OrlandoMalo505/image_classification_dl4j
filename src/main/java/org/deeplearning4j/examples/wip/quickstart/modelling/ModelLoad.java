package org.deeplearning4j.examples.wip.quickstart.modelling;

import org.nd4j.linalg.api.buffer.DataBuffer;
import java.util.Arrays;
import org.apache.arrow.flatbuf.Int;
import org.apache.commons.io.IOUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import java.sql.Array;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class ModelLoad {

    public static float maximum(float[] t) {
        float maximum = t[0];   // start with the first value
        for (int i=1; i<t.length; i++) {
            if (t[i] > maximum) {
                maximum = t[i];   // new maximum
            }
        }
        return maximum;
    }

    public static int getIndexOfLargest( float[] array )
    {
        if ( array == null || array.length == 0 ) return -1; // null or empty

        int largest = 0;
        for ( int i = 1; i < array.length; i++ )
        {
            if ( array[i] > array[largest] ) largest = i;
        }
        return largest; // position of the first largest found
    }






    public static void main(String[] args) throws Exception {

        int numOfChannels = 3;
        int width = 300;
        int height = 300;

        int numOfOutput = 10;
        int batchSize = 10;
        int numOfEpochs = 10;

        int numOfIterations = 3;
        int seed = 123;
        Random random = new Random();

        File trainData = new File("C:\\Users\\User\\Downloads\\TRAINFOLDER");
        File testData = new File("C:\\Users\\User\\Desktop\\testim_imazhi");

        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, random);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, random);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, numOfChannels, labelMaker);
        recordReader.initialize(train);


        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numOfOutput);


        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        DataSet ds = dataIter.next();


        System.out.println(dataIter.getLabels());


        File locationToSave = new File("modeli.zip");

        MultiLayerNetwork neuralNetwork = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        ParentPathLabelGenerator labelMakerTest = new ParentPathLabelGenerator();
        ImageRecordReader recordReaderTest = new ImageRecordReader(height, width, numOfChannels, labelMakerTest);
        recordReaderTest.initialize(test);

        DataSetIterator dataIterTest = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, numOfOutput);

        DataNormalization scalerTest = new ImagePreProcessingScaler(0, 1);
        scalerTest.fit(dataIterTest);
        dataIterTest.setPreProcessor(scalerTest);

        String[] vektori=new String[]{"Dele","Elefant","Fluturë","Kal","Ketër","Lopë","Mace","Merimangë","Pulë","Qen"};


        DataBuffer dataBuffer = neuralNetwork.output(dataIterTest).data();
        float[] array = dataBuffer.asFloat();

        System.out.println("*************************\n\n");
        System.out.println("Kategorite e imazhit:"+Arrays.toString(vektori)+"\n\n");
        System.out.println("Probabiliteti sipas klasave:" + Arrays.toString(array)+"\n\n");
        System.out.println("Probabiliteti maksimal:"+ maximum(array)+"\n\n");
        int nr=getIndexOfLargest(array);
        System.out.println("Imazhi klasifikohet si: "+vektori[nr]+"\n\n");
        System.out.println("************************");




    }

}




