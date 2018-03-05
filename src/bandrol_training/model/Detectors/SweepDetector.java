package bandrol_training.model.Detectors;

import bandrol_training.model.Algorithms.NonMaximaSuppression;
import bandrol_training.model.Detection;
import bandrol_training.model.Ensembles.EnsembleModel;
import com.google.common.collect.Table;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class SweepDetector extends Thread {
    private List<String> characters;
    private List<Detection> detections;
    private Map<String, EnsembleModel> ensembleMap;
    private Table<Integer, Integer, Mat> featureTable;
    private int sliding_window_width;
    private int sliding_window_height;
    private double nms_iou_threshold;

//    public SweepDetector(List<String> characters,
//                         EnsembleModel ensemble, Table<Integer, Integer, Mat> featureTable,
//                         int sliding_window_width, int sliding_window_height, double nms_iou_threshold)
//    {
//        SweepDetector(ensemble, featureTable, sliding_window_width, sliding_window_height, nms_iou_threshold);
//        this.characters = characters;
//    }

    public SweepDetector(Map<String, EnsembleModel> ensembleMap, Table<Integer, Integer, Mat> featureTable,
                         int sliding_window_width, int sliding_window_height, double nms_iou_threshold)
    {
        this.ensembleMap = ensembleMap;
        this.featureTable = featureTable;
        this.sliding_window_width = sliding_window_width;
        this.sliding_window_height = sliding_window_height;
        this.nms_iou_threshold = nms_iou_threshold;
        characters = new ArrayList<>();
        detections = new ArrayList<>();
    }

    private List<Detection> getLabelDetections(String label) {
        List<Detection> detectionList = new ArrayList<>();
        EnsembleModel ensemble = ensembleMap.get(label);
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
        List<Detection> maxima = NonMaximaSuppression.run(detectionList, nms_iou_threshold);
        return maxima;
    }

    public void addCharacter(String label)
    {
        characters.add(label);
    }

    public void run()
    {
        for(String label : characters)
        {
            System.out.println("Processing Label:"+label);
            detections.addAll(getLabelDetections(label));
        }
    }

    public List<Detection> getDetections() {
        return detections;
    }

    public List<String> getCharacters() {
        return characters;
    }
}
