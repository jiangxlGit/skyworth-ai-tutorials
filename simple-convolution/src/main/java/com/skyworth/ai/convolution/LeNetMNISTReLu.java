package com.skyworth.ai.convolution;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * 在 MNIST 数据集上实现 LeNet-5 手写数字图像分类（准确率 99%）
 */
@Slf4j
public class LeNetMNISTReLu {

    // 存放文件的地址
    private static final String BASE_PATH = "/Users/jloved/github_repo/skyworth-ai-tutorials";

    public static void main(String[] args) throws Exception {
        // 图片像素高
        int height = 28;
        // 图片像素宽
        int width = 28;

        // 因为是黑白图像，所以颜色通道只有一个
        int channels = 1;

        // 分类结果，0-9，共十种数字
        int outputNum = 10;

        // 批大小
        int batchSize = 54;

        // 循环次数
        int nEpochs = 1;

        // 初始化伪随机数的种子
        int seed = 1234;

        // 随机数工具
        Random randNumGen = new Random(seed);

        log.info("检查数据集文件夹是否存在：{}", BASE_PATH + "/mnist_png");

        if (!new File(BASE_PATH + "/mnist_png").exists()) {
            log.info("数据集文件不存在，请下载压缩包并解压到：{}", BASE_PATH);
            return;
        }

        // 标签生成器，将指定文件的父目录作为标签
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        // 归一化配置(像素值从0-255变为0-1)
        DataNormalization imageScalar = new ImagePreProcessingScaler();

        // 不论训练集还是测试集，初始化操作都是相同套路：
        // 1. 读取图片
        // 2. 根据批大小创建的迭代器
        // 3. 将归一化器作为预处理器

        log.info("===== 训练集的初始化操作...");
        // 初始化训练集
        File trainData = new File(BASE_PATH + "/mnist_png/training");
        // 分批读取训练集文件夹中图片，并对图片进行初始化
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(trainSplit);
        /*
         * 数据集迭代器
         * 形参:
         * recordReader – RecordReader：提供数据源
         * batchSize – 输出 DataSet 对象的批次大小（示例数）
         * labelIndex – 标签 Writable 的索引（通常是 IntWritable），由 recordReader. next() 获取
         * outputNum – 分类的类别数（可能的标签）
         */
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        // 拟合数据(实现类中实际上什么也没做)
//        imageScalar.fit(trainIter);
        // 调用迭代器，进行图片预处理（主要进行归一化）
        dataSetIterator.setPreProcessor(imageScalar);
        log.info("===== 训练集的初始化完成 =====");


        log.info("===== 测试集的初始化操作...");
        // 初始化测试集，与前面的训练集操作类似
        File testData = new File(BASE_PATH + "/mnist_png/testing");
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testSplit);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
        testIter.setPreProcessor(imageScalar);
        log.info("===== 测试集的初始化完成 =====");

        log.info("===== 配置神经网络 =====");

        // 在训练中，将学习率配置为随着迭代阶梯性下降
        Map<Integer, Double> learningRateSchedule = new HashMap<>();
        learningRateSchedule.put(0, 0.06);
        learningRateSchedule.put(200, 0.05);
        learningRateSchedule.put(600, 0.028);
        learningRateSchedule.put(800, 0.0060);
        learningRateSchedule.put(1000, 0.001);

        // 配置参数
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

                // 初始化随机数的种子
                .seed(seed)

                // L2正则化系数
                .l2(0.0005)

                // 梯度下降的学习率设置
                .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))

                // 权重初始化（均值为 0、方差为 2.0/(fanIn + fanOut) 的高斯分布）
                .weightInit(WeightInit.XAVIER)

                // 准备分层
                .list()

                // INPUT图片数据的输入类型 (高度 * 宽度 * 通道)
                .setInputType(InputType.convolutionalFlat(height, width, channels))

                // C1卷积层
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels) // 颜色通道, 如果是黑白图像，则颜色通道只有一个
                        .stride(1, 1) // 步进数
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())

                // S2下采样，即池化
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // 最大池化，取最大值
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())

                // C3卷积层
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())

                // S4下采样，即池化
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())

                // F6全连接，激活函数设置为ReLU
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500)
                        .build())

                // OUTPUT输出，激活函数为归一化指数函数
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // 每十个迭代打印一次损失函数值
        net.setListeners(new ScoreIterationListener(10));

        log.info("===== 神经网络共[{}]个参数", net.numParams());

        long startTime = System.currentTimeMillis();
        // 循环操作，进行训练和测试
        for (int i = 0; i < nEpochs; i++) {
            log.info("------ 第[{}]个循环", i + 1);
            net.fit(dataSetIterator);
            Evaluation eval = net.evaluate(testIter);
            log.info(eval.stats());
            dataSetIterator.reset();
            testIter.reset();
        }
        log.info("===== 完成{}次训练和测试，耗时[{}]毫秒 =====", nEpochs, System.currentTimeMillis() - startTime);

        // 保存模型
        File mnistModelPath = new File(BASE_PATH + "/mnist-model.zip");
        ModelSerializer.writeModel(net, mnistModelPath, true);
        log.info("@@@@@@@@ 最新的MNIST模型保存在[{}] @@@@@@@@@", mnistModelPath.getPath());
    }
}
