package com.skyworth.ai.predictnumber.service;

import org.springframework.web.multipart.MultipartFile;

/**
 * @author jiangxl
 * @version V1.0
 * @ClassName PredictService
 * @Description 数字识别接口
 * @Date 2024/7/1 下午1:31
 */
public interface PredictService {

    /**
     * 取得上传的图片，做转换后识别成数字
     *
     * @param file         上传的文件
     * @param isNeedRevert 是否要做反色处理
     * @return
     */
    int predict(MultipartFile file, boolean isNeedRevert) throws Exception;
}
