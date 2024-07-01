package com.skyworth.ai.predictnumber.service.impl;

import com.skyworth.commons.utils.ImageFileUtil;
import com.skyworth.ai.predictnumber.service.PredictService;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.PostConstruct;
import java.io.File;

/**
 * @author jiangxl
 * @version V1.0
 * @ClassName PredictServiceImpl
 * @Description 数字识别服务的接口实现类
 * @Date 2024/7/1 下午2:07
 */
@Service
@Slf4j
public class PredictServiceImpl implements PredictService {

    /**
     * -1表示识别失败
     */
    private static final int RLT_INVALID = -1;

    /**
     * 模型文件的位置
     */
    private final String modelPath = System.getProperty("user.dir") + "/mnist-model.zip";

    /**
     * 处理图片文件的目录
     */
    private final String imageFilePath = System.getProperty("user.dir");

    /**
     * 神经网络
     */
    private MultiLayerNetwork net;

    /**
     * bean实例化成功就加载模型
     */
    @PostConstruct
    private void loadModel() {
        log.info("load model from [{}]", modelPath);

        // 加载模型
        try {
            net = ModelSerializer.restoreMultiLayerNetwork(new File(modelPath));
            log.info("module summary\n{}", net.summary());
        } catch (Exception exception) {
            log.error("loadModel error", exception);
        }
    }

    @Override
    public int predict(MultipartFile file, boolean isNeedRevert) throws Exception {
        log.info("start predict, file [{}], isNeedRevert [{}]", file.getOriginalFilename(), isNeedRevert);

        // 先存文件
        String rawFileName = ImageFileUtil.save(imageFilePath, file);

        if (null == rawFileName) {
            return RLT_INVALID;
        }

        // 反色处理后的文件名
        String revertFileName = null;

        // 调整大小后的文件名
        String resizeFileName;

        // 是否需要反色处理
        if (isNeedRevert) {
            // 把原始文件做反色处理，返回结果是反色处理后的新文件
            revertFileName = ImageFileUtil.colorRevert(imageFilePath, rawFileName);

            // 把反色处理后调整为28*28大小的文件
            resizeFileName = ImageFileUtil.resize(imageFilePath, revertFileName);
        } else {
            // 直接把原始文件调整为28*28大小的文件
            resizeFileName = ImageFileUtil.resize(imageFilePath, rawFileName);
        }

        // 现在已经得到了结果反色和调整大小处理过后的文件，
        // 那么原始文件和反色处理过的文件就可以删除了
        ImageFileUtil.clear(imageFilePath, rawFileName, revertFileName);

        // 取出该黑白图片的特征
        INDArray features = ImageFileUtil.getGrayImageFeatures(imageFilePath, resizeFileName);

        // 将特征传给模型去识别
        return net.predict(features)[0];
    }
}