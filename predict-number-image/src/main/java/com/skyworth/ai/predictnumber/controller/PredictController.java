package com.skyworth.ai.predictnumber.controller;

import com.skyworth.ai.predictnumber.service.PredictService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * @ClassName   PredictController
 * @Description shou'xie
 * @author      jiangxl
 * @Date        2024/7/1 下午1:40
 * @version     V1.0
 */
@RequestMapping("/pni")
@RestController
public class PredictController {

    @Autowired
    private PredictService predictService;
    
    // 训练模型的时候，用的数字是白字黑底，
    // 因此如果上传白字黑底的图片，可以直接拿去识别，而无需反色处理
    @PostMapping("/predict-with-black-background")
    @ResponseBody
    public int predictWithBlackBackground(@RequestParam("file") MultipartFile file) throws Exception {
        return predictService.predict(file, false);
    }
}
