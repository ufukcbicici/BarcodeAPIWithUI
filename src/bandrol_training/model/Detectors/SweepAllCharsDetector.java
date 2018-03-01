package bandrol_training.model.Detectors;

import bandrol_training.Constants;
import bandrol_training.model.ClassifierType;
import bandrol_training.model.Detection;
import bandrol_training.model.Ensembles.EnsembleModel;
import bandrol_training.model.Ensembles.SVMEnsemble;
import bandrol_training.model.Utils;
import com.google.common.collect.Table;
import jdk.jshell.spi.ExecutionControl;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import org.opencv.ml.StatModel;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static bandrol_training.Constants.DETECTIONPATH;

public class SweepAllCharsDetector extends DetectionMethod {

    private Map<String, EnsembleModel> ensembleMap;
    private ClassifierType classifierType;

    public SweepAllCharsDetector(ClassifierType classifierType)
    {
        ensembleMap = new HashMap<>();
        this.classifierType = classifierType;
    }

    public void init() throws ExecutionControl.NotImplementedException {
        for(String label : Constants.CURR_LABELS)
        {
            switch (this.classifierType)
            {
                case SVM:
                    SVMEnsemble svmEnsemble = new SVMEnsemble(false);
                    svmEnsemble.loadEnsemble(label);
                    ensembleMap.put(label, svmEnsemble);
                    break;
                case MLP:
                    throw new ExecutionControl.NotImplementedException("MLP not implemented");
            }
        }
    }

    private List<Detection> getLabelDetections(
            String label,
            Table<Integer, Integer, Mat> featureTable,
            int sliding_window_width,
            int sliding_window_height,
            double nms_iou_threshold) {
        EnsembleModel ensemble = ensembleMap.get(label);
        List<Detection> detectionList = new ArrayList<>();
        for (Table.Cell c : featureTable.cellSet()) {
            Mat feature = (Mat) c.getValue();
            Mat predictedLabels = ensemble.predictLabels(feature);
            Mat predictedMargins = ensemble.predictConfidences(feature);
            double totalMarginResponse = 0.0;
            double totalVote = 0.0;
            assert predictedLabels.cols() == predictedMargins.cols();
            for (int j = 0; j < predictedLabels.cols(); j++) {
                double predictedLabel = predictedLabels.get(0, j)[0];
                totalVote += predictedLabel;
                totalMarginResponse += Math.abs(predictedMargins.get(0, j)[0]) * predictedLabel;
            }
            double avgMarginResponse = totalMarginResponse / (double) ensemble.getModelCount();
            if (totalVote > 0) {
                Detection detection = new Detection(
                        new Rect((int) c.getColumnKey(), (int) c.getRowKey(),
                                sliding_window_width, sliding_window_height), avgMarginResponse, label);
                detectionList.add(detection);
            }
        }
        List<Detection> maxima = nonMaximaSuppression(detectionList, nms_iou_threshold);
        return maxima;
    }

    public void detect(Mat img, int sliding_window_width, int sliding_window_height,
                                double sourceImgWidth, double nms_iou_threshold)
    {
        Mat canvasImg = img.clone();
        Table<Integer, Integer, Mat> featureTable =
                extractFeatures(img, sliding_window_width, sliding_window_height, sourceImgWidth);
        List<Detection> listOfMaximaForAllLabels = new ArrayList<>();
        for(String label : Constants.CURR_LABELS) {
            System.out.println("Processing Label:"+label);
            if (!ensembleMap.containsKey(label))
                continue;
            listOfMaximaForAllLabels.addAll(getLabelDetections(label, featureTable,
                    sliding_window_width, sliding_window_height, nms_iou_threshold));
        }
        List<Detection> ultimateMaxima = nonMaximaSuppression(listOfMaximaForAllLabels, nms_iou_threshold);
        for (Detection dtc : ultimateMaxima) {
            Rect r = dtc.getRect();
            Imgproc.rectangle(canvasImg, new Point(r.x, r.y),
                    new Point(r.x + r.width - 1, r.y + r.height - 1),
                    new Scalar(0, 0, 255));
            int font = Core.FONT_HERSHEY_PLAIN;
            Imgproc.putText(canvasImg, dtc.getLabel(), new Point(r.x + 5, r.y + 10), font,
                    1.0,new Scalar(0,255,0),1);
        }
        String fileName = Utils.getNonExistingFileName(DETECTIONPATH + "final_detection", ".png");
        Imgcodecs.imwrite(fileName, canvasImg);
        Utils.showImageInPopup(Utils.matToBufferedImage(canvasImg, null));
    }
}
