package bandrol_training.model.Detectors;

import bandrol_training.Constants;
import bandrol_training.model.Detection;
import bandrol_training.model.HOGExtractor;
import bandrol_training.model.Utils;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public abstract class DetectionMethod {

    protected Table<Integer, Integer, Mat> extractFeatures(Mat img, int sliding_window_width, int sliding_window_height,
                                                           double sourceImgWidth)
    {
        Table<Integer, Integer, Mat> hogTable = HashBasedTable.create();
        double upperRatio = sourceImgWidth * Constants.QR_RATIO;
        for(int i=0;i<img.rows();i++) {
            for (int j = 0; j < img.cols(); j++) {
                if (i < upperRatio || i + sliding_window_height - 1 >= img.rows())
                    continue;
                if (j + sliding_window_width - 1 >= img.cols())
                    continue;
                Mat imgRect = img.submat(i, i + sliding_window_height, j, j + sliding_window_width);
                Mat hogFeatureT = HOGExtractor.extractOpenCVHogFeature(imgRect, sliding_window_width,
                        sliding_window_height);
                Mat hogFeature = new Mat();
                Core.transpose(hogFeatureT, hogFeature);
                Mat hogFeature32f = new Mat();
                hogFeature.convertTo(hogFeature32f, CvType.CV_32F);
                hogTable.put(i,j,hogFeature32f);
            }
        }
        return hogTable;
    }

    public abstract void detect(Mat img, int sliding_window_width, int sliding_window_height,
                                double sourceImgWidth, double nms_iou_threshold);

    public static List<Detection> nonMaximaSuppression(List<Detection> preListOfDetections, double nms_iou_threshold)
    {
//        List<Detection> sortedList = preListOfDetections.stream().
//                sorted(Comparator.comparing(Detection::getResponse)).collect(Collectors.toList());
        List<Detection> sortedList = preListOfDetections.stream().
                sorted((d0,d1) -> -d0.getResponse().compareTo(d1.getResponse())).collect(Collectors.toList());
        List<Detection> maxima = new ArrayList<>();
        while (sortedList.size() > 0)
        {
            Detection mostConfidentDetection = sortedList.get(0);
            List<Detection> survivedList = new ArrayList<>();
            maxima.add(mostConfidentDetection);
            for(int i=1;i<sortedList.size();i++)
            {
                Detection candidate = sortedList.get(i);
                double iou = Utils.calculateIoU(mostConfidentDetection.getRect(), candidate.getRect());
                if(iou < nms_iou_threshold)
                    survivedList.add(candidate);
            }
            sortedList = survivedList;
        }
        return maxima;
    }

}
